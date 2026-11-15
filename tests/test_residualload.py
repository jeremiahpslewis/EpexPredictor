"""Tests for the residual load forecast."""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from predictor.model.auxdatastore import AuxDataStore
from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
from predictor.model.residualload import (
    GenerationForecastStore,
    GridWeatherStore,
    ResidualLoad,
    generation_features,
)

START = datetime(2025, 6, 1, tzinfo=timezone.utc)
DAYS = 80


@pytest.fixture
def region():
    return PriceRegionName.DE.to_region()


def locations(region) -> int:
    return len(region.generation_latitudes) + len(region.generation_offshore_latitudes)


def grid_weather(region, start=START, days=DAYS) -> pd.DataFrame:
    """Hourly weather whose wind differs from hour to hour, the same at all locations."""
    index = pd.date_range(start, start + timedelta(days=days), freq="1h", name="time")
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 50, len(index))
    radiation = np.clip(np.sin((index.hour.to_numpy() - 6) / 12 * np.pi), 0, None) * 600
    data = {}
    for i in range(locations(region)):
        data[f"wind_{i}"] = wind
        data[f"radiation_{i}"] = radiation
        data[f"temp_{i}"] = 15.0
    return pd.DataFrame(data, index=index)


def published_figures(weather: pd.DataFrame) -> pd.DataFrame:
    """What the TSOs would publish for that weather, on a 15 minute index."""
    index = pd.date_range(weather.index[0], weather.index[-1], freq="15min", name="time")
    hourly = weather.reindex(index).interpolate(method="time")
    return pd.DataFrame({
        "tso_wind": np.clip(hourly["wind_0"] / 43.2, 0, 1) ** 3 * 50,
        "tso_solar": hourly["radiation_0"] / 600 * 40,
        "tso_load": 55.0,
    }, index=index)


def forecaster(region, weather: pd.DataFrame, published: pd.DataFrame) -> ResidualLoad:
    residual_load = ResidualLoad(region, None, AuxDataStore(region))
    residual_load.LGB_ROUNDS = 60  # enough for the simple relations of the test data, and much faster
    residual_load.gridweather.get_data = AsyncMock(side_effect=lambda start, end: weather.loc[start:end])
    residual_load.generation.get_data = AsyncMock(side_effect=lambda start, end: published.loc[start:end])
    return residual_load


class TestGridWeatherStore:
    def test_covers_the_generation_grid(self, region):
        store = GridWeatherStore(region)

        assert len(store.latitudes()) == len(store.longitudes()) == locations(region)
        assert store.latitudes()[-1] == region.generation_offshore_latitudes[-1]
        assert len(store.expected_columns()) == 3 * locations(region)
        assert "wind_0" in store.expected_columns()

    def test_is_stored_apart_from_the_region_weather(self, region, temp_storage_dir):
        store = GridWeatherStore(region, temp_storage_dir)

        assert "gridweather_v1" in store.get_storage_file()

    @pytest.mark.asyncio
    async def test_requests_hourly_values_of_the_grid(self, region):
        store = GridWeatherStore(region)
        requested = []

        class FailingSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def get(self, url):
                requested.append(url)
                raise RuntimeError("no network in tests")

        with patch("predictor.model.weatherstore.aiohttp.ClientSession", return_value=FailingSession()):
            with pytest.raises(RuntimeError):
                await store.refresh_range(START, START + timedelta(days=1))

        assert "&hourly=wind_speed_120m,shortwave_radiation,temperature_2m&" in requested[0]
        assert f"latitude={','.join(map(str, store.latitudes()))}&" in requested[0]

    def test_fetches_in_pieces_of_a_month(self, region):
        store = GridWeatherStore(region)

        ranges = store.gen_missing_date_ranges(START, START + timedelta(days=100))

        assert len(ranges) == 4
        assert all(end - start <= timedelta(days=31) for start, end in ranges)

    @pytest.mark.asyncio
    async def test_reports_an_error_of_the_api(self, region):
        store = GridWeatherStore(region)
        store.REQUEST_PAUSE_SECONDS = 0

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def text(self):
                return json.dumps({"error": True, "reason": "Daily API request limit exceeded."})

        class Session(Response):
            def get(self, url):
                return Response()

        with patch("predictor.model.weatherstore.aiohttp.ClientSession", return_value=Session()):
            with pytest.raises(RuntimeError, match="limit exceeded"):
                await store.refresh_range(START, START + timedelta(days=1))


class TestGenerationForecastStore:
    @pytest.mark.asyncio
    async def test_adds_up_wind_and_converts_to_gw(self, region):
        store = GenerationForecastStore(region)
        index = pd.date_range(START, periods=4, freq="15min")
        megawatts = {"wind_onshore": 10000.0, "wind_offshore": 2000.0, "solar": 5000.0, "load": 50000.0}
        store.fetch_forecast = AsyncMock(side_effect=lambda kind, start, end: pd.Series(megawatts[kind], index=index))

        df = await store.fetch_range(START, START + timedelta(hours=1))

        assert df is not None
        assert list(df.columns) == ["tso_wind", "tso_solar", "tso_load"]
        assert (df["tso_wind"] == 12.0).all()
        assert (df["tso_solar"] == 5.0).all()
        assert (df["tso_load"] == 50.0).all()

    @pytest.mark.asyncio
    async def test_keeps_nothing_when_one_figure_cannot_be_fetched(self, region):
        store = GenerationForecastStore(region)
        index = pd.date_range(START, periods=4, freq="15min")
        store.fetch_forecast = AsyncMock(side_effect=lambda kind, start, end: None if kind == "solar" else pd.Series(1.0, index=index))

        assert await store.fetch_range(START, START + timedelta(hours=1)) is None
        assert await store.fetch_missing_data(START, START + timedelta(hours=1)) is False
        assert store.data.empty

    @pytest.mark.asyncio
    async def test_stores_what_was_fetched(self, region):
        store = GenerationForecastStore(region)
        index = pd.date_range(START, START + timedelta(hours=1), freq="15min")
        store.fetch_forecast = AsyncMock(side_effect=lambda kind, start, end: pd.Series(1000.0, index=index))

        assert await store.fetch_missing_data(START, START + timedelta(hours=1)) is True
        assert len(store.data) == 5
        # nothing is missing any more, so nothing is fetched again
        store.fetch_forecast.reset_mock()
        assert await store.fetch_missing_data(START, START + timedelta(hours=1)) is False
        store.fetch_forecast.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_stored_data_is_updated_right_after_a_restart(self, region, temp_storage_dir):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        old = pd.date_range(now - timedelta(days=5), now - timedelta(days=3), freq="15min", name="time")
        store = GenerationForecastStore(region, temp_storage_dir)
        store.data = pd.DataFrame({"tso_wind": 1.0, "tso_solar": 1.0, "tso_load": 1.0}, index=old)
        await store.serialize()
        three_days_ago = (now - timedelta(days=3)).timestamp()
        os.utime(store.get_storage_file(), (three_days_ago, three_days_ago))

        restarted = await GenerationForecastStore(region, temp_storage_dir).load()
        new = pd.date_range(now - timedelta(days=3), now, freq="15min")
        restarted.fetch_forecast = AsyncMock(side_effect=lambda kind, start, end: pd.Series(1000.0, index=new))
        result = await restarted.get_data(now - timedelta(days=5), now)

        restarted.fetch_forecast.assert_called()
        assert result.index.max() == now

    @pytest.mark.asyncio
    async def test_fresh_stored_data_is_not_fetched_again_before_the_next_check(self, region, temp_storage_dir):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        index = pd.date_range(now - timedelta(days=2), now, freq="15min", name="time")
        store = GenerationForecastStore(region, temp_storage_dir)
        store.data = pd.DataFrame({"tso_wind": 1.0, "tso_solar": 1.0, "tso_load": 1.0}, index=index)
        await store.serialize()

        restarted = await GenerationForecastStore(region, temp_storage_dir).load()
        restarted.fetch_forecast = AsyncMock()
        await restarted.get_data(now - timedelta(days=1), now + timedelta(days=2))

        restarted.fetch_forecast.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_nothing_for_a_region_without_forecasts(self):
        store = GenerationForecastStore(PriceRegionName.AT.to_region())
        store.fetch_forecast = AsyncMock()

        assert await store.fetch_missing_data(START, START + timedelta(hours=1)) is False
        store.fetch_forecast.assert_not_called()


class TestGenerationFeatures:
    @pytest.mark.asyncio
    async def test_features_on_the_quarter_hour_index(self, region):
        weather = grid_weather(region, days=2)
        aux = await AuxDataStore(region).get_data(START, START + timedelta(days=1))

        features = generation_features(region, weather, aux)

        assert features.index.equals(aux.index)
        assert not features.isna().any().any()
        assert features["wind_power_onshore"].between(0, 1).all()
        # between two hours, the weather is interpolated
        hour = features["radiation_all"].loc[START + timedelta(hours=8)]
        next_hour = features["radiation_all"].loc[START + timedelta(hours=9)]
        assert features["radiation_all"].loc[START + timedelta(hours=8, minutes=30)] == pytest.approx((hour + next_hour) / 2)

    @pytest.mark.asyncio
    async def test_no_features_without_weather(self, region):
        weather = grid_weather(region, days=1)
        aux = await AuxDataStore(region).get_data(START, START + timedelta(days=3))

        features = generation_features(region, weather, aux)

        assert features.loc[START + timedelta(days=2), "wind_power_onshore"] != features.loc[START + timedelta(days=2), "wind_power_onshore"]
        assert features.loc[START + timedelta(hours=12)].notna().all()


class TestResidualLoad:
    def test_configured_for_germany_only(self, monkeypatch):
        monkeypatch.undo()

        assert ResidualLoad.is_configured(PriceRegionName.DE.to_region())
        assert not ResidualLoad.is_configured(PriceRegionName.AT.to_region())

    @pytest.mark.asyncio
    async def test_published_figure_where_there_is_one_and_forecast_elsewhere(self, region):
        weather = grid_weather(region)
        truth = published_figures(weather)
        last_published = START + timedelta(days=DAYS - 2)
        residual_load = forecaster(region, weather, truth.loc[:last_published].copy())
        end = START + timedelta(days=DAYS - 1)

        await residual_load.train(START, end)
        result = await residual_load.get_data(START, end)

        assert result is not None
        assert list(result.columns) == ["residual_load"]
        expected = truth["tso_load"] - truth["tso_wind"] - truth["tso_solar"]
        known = result.loc[:last_published, "residual_load"]
        pd.testing.assert_series_equal(known, expected.loc[:last_published], check_names=False, check_freq=False)
        forecast = result.loc[last_published + timedelta(minutes=15):, "residual_load"]
        assert not forecast.isna().any()
        assert (forecast - expected.loc[forecast.index]).abs().mean() < 3.0

    @pytest.mark.asyncio
    async def test_no_value_without_weather_and_published_figure(self, region):
        weather = grid_weather(region)
        truth = published_figures(weather)
        residual_load = forecaster(region, weather, truth)
        end = START + timedelta(days=DAYS + 2)

        await residual_load.train(START, end)
        result = await residual_load.get_data(START, end)

        assert result is not None
        assert result.loc[START + timedelta(days=DAYS + 1):, "residual_load"].isna().all()
        assert not result.loc[:START + timedelta(days=DAYS), "residual_load"].isna().any()

    @pytest.mark.asyncio
    async def test_not_trained_with_too_little_data(self, region):
        weather = grid_weather(region, days=10)
        residual_load = forecaster(region, weather, published_figures(weather))

        await residual_load.train(START, START + timedelta(days=10))

        assert residual_load.models is None
        assert await residual_load.get_data(START, START + timedelta(days=10)) is None

    @pytest.mark.asyncio
    async def test_weather_service_down_leaves_the_price_model_without_this_input(self, region):
        weather = grid_weather(region)
        residual_load = forecaster(region, weather, published_figures(weather))
        residual_load.gridweather.get_data = AsyncMock(side_effect=RuntimeError("Daily API request limit exceeded."))

        await residual_load.train(START, START + timedelta(days=DAYS))

        assert residual_load.models is None
        assert await residual_load.get_data(START, START + timedelta(days=DAYS)) is None

    @pytest.mark.asyncio
    async def test_weather_service_down_after_training_uses_stored_weather(self, region):
        weather = grid_weather(region)
        truth = published_figures(weather)
        last_published = START + timedelta(days=DAYS - 2)
        residual_load = forecaster(region, weather, truth.loc[:last_published].copy())
        end = START + timedelta(days=DAYS + 1)
        await residual_load.train(START, end)

        residual_load.gridweather.data = weather
        residual_load.gridweather.get_data = AsyncMock(side_effect=RuntimeError("no connection"))
        result = await residual_load.get_data(START, end)

        assert result is not None
        assert not result.loc[:START + timedelta(days=DAYS), "residual_load"].isna().any()
        assert result.loc[START + timedelta(days=DAYS, hours=1):, "residual_load"].isna().all()

    @pytest.mark.asyncio
    async def test_shared_stores_but_own_models(self, region):
        weather = grid_weather(region)
        residual_load = forecaster(region, weather, published_figures(weather))
        await residual_load.train(START, START + timedelta(days=DAYS))

        other = residual_load.with_shared_stores()

        assert other.gridweather is residual_load.gridweather
        assert other.generation is residual_load.generation
        assert other.models is None
        assert residual_load.models is not None


class TestPricePredictorWithResidualLoad:
    def test_only_for_configured_regions(self, monkeypatch):
        monkeypatch.undo()

        assert PricePredictor(PriceRegionName.DE.to_region()).residual_load is not None
        assert PricePredictor(PriceRegionName.AT.to_region()).residual_load is None

    def test_stage_2_model_shares_the_stores(self, monkeypatch):
        monkeypatch.undo()
        base = PricePredictor(PriceRegionName.DE.to_region())
        stage2 = PricePredictor(PriceRegionName.DE.to_region(), cross_predictors=[base])

        stage2.use_datastores_from(base)

        assert stage2.residual_load is not base.residual_load
        assert stage2.residual_load.generation is base.residual_load.generation
        assert stage2.residual_load.auxstore is base.auxstore

    def test_horizon_cutoff_hides_published_figures(self, monkeypatch):
        monkeypatch.undo()
        predictor = PricePredictor(PriceRegionName.DE.to_region())

        predictor.set_horizon_cutoff(START)

        assert predictor.pricestore.horizon_cutoff == START
        assert predictor.gasstore.horizon_cutoff == START
        assert predictor.residual_load.generation.horizon_cutoff == START
        assert predictor.residual_load.gridweather.horizon_cutoff is None

    @pytest.mark.asyncio
    async def test_residual_load_is_an_input_of_the_price_model(self, monkeypatch, region, sample_gas_data):
        monkeypatch.undo()
        weather = grid_weather(region)
        truth = published_figures(weather)
        end = START + timedelta(days=DAYS)
        index = pd.date_range(START, end, freq="15min", name="time")

        predictor = PricePredictor(region)
        predictor.residual_load = forecaster(region, weather, truth)
        predictor.residual_load.auxstore = predictor.auxstore
        region_weather = pd.DataFrame({"wind_0": 5.0, "temp_0": 10.0}, index=index)
        # the price follows the residual load, and nothing else
        prices = pd.DataFrame({"price": (truth["tso_load"] - truth["tso_wind"] - truth["tso_solar"]) / 5}, index=index)
        gas = pd.DataFrame({"gasprice": 30.0}, index=index)
        predictor.weatherstore.get_data = AsyncMock(side_effect=lambda s, e: region_weather.loc[s:e])
        predictor.pricestore.get_data = AsyncMock(side_effect=lambda s, e: prices.loc[s:e])
        predictor.gasstore.get_data = AsyncMock(side_effect=lambda s, e: gas.loc[s:e])

        await predictor.train(START, end)
        assert predictor.traindata is not None
        assert "residual_load" in predictor.traindata.columns

        check_start = end - timedelta(days=2)
        predicted = await predictor.predict(check_start, end, fill_known=False)
        error = (predicted["price"] - prices.loc[check_start:end, "price"]).abs().mean()
        assert error < 0.5


class TestRefreshFailures:
    @pytest.mark.asyncio
    async def test_grid_weather_failure_does_not_stop_the_refresh(self, monkeypatch):
        monkeypatch.undo()
        predictor = PricePredictor(PriceRegionName.DE.to_region())
        predictor.weatherstore.refresh_range = AsyncMock(return_value=True)
        predictor.entsoestore.refresh_range = AsyncMock(return_value=True)
        predictor.residual_load.gridweather.refresh_range = AsyncMock(side_effect=RuntimeError("Daily API request limit exceeded."))

        await predictor.refresh_forecasts(START, START + timedelta(days=1))

        predictor.weatherstore.refresh_range.assert_awaited_once()
        predictor.entsoestore.refresh_range.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_grid_weather_failure_does_not_stop_the_base_model_update(self, monkeypatch):
        monkeypatch.undo()
        from predictor.api import priceapi

        coordinator = priceapi.BaseModelCoordinator()
        for p in coordinator.base.values():
            p.pricestore.get_data = AsyncMock(return_value=pd.DataFrame())
            p.gasstore.get_data = AsyncMock(return_value=pd.DataFrame())
            p.weatherstore.refresh_range = AsyncMock(return_value=False)
            p.entsoestore.refresh_range = AsyncMock(return_value=False)
            p.train = AsyncMock()
            if p.residual_load is not None:
                p.residual_load.gridweather.refresh_range = AsyncMock(side_effect=RuntimeError("no connection"))
        monkeypatch.setattr(priceapi.asyncio, "sleep", AsyncMock())

        await coordinator._do_update()

        assert all(p.train.await_count == 1 for p in coordinator.base.values())


class TestFallbackWithoutResidualLoad:
    def price_predictor(self, monkeypatch, region, weather_days=DAYS + 2):
        monkeypatch.undo()
        weather = grid_weather(region)
        truth = published_figures(weather)
        end = START + timedelta(days=weather_days)
        index = pd.date_range(START, end, freq="15min", name="time")
        predictor = PricePredictor(region)
        predictor.residual_load = forecaster(region, weather, truth)
        predictor.residual_load.auxstore = predictor.auxstore
        region_weather = pd.DataFrame({"wind_0": 5.0, "temp_0": 10.0}, index=index)
        residual = (truth["tso_load"] - truth["tso_wind"] - truth["tso_solar"]).reindex(index).ffill()
        prices = pd.DataFrame({"price": residual / 5}, index=index)
        gas = pd.DataFrame({"gasprice": 30.0}, index=index)
        predictor.weatherstore.get_data = AsyncMock(side_effect=lambda s, e: region_weather.loc[s:e])
        predictor.pricestore.get_data = AsyncMock(side_effect=lambda s, e: prices.loc[s:e])
        predictor.gasstore.get_data = AsyncMock(side_effect=lambda s, e: gas.loc[s:e])
        return predictor

    @pytest.mark.asyncio
    async def test_forecast_without_grid_weather_where_the_rest_is_there(self, monkeypatch, region):
        predictor = self.price_predictor(monkeypatch, region)
        await predictor.train(START, START + timedelta(days=DAYS - 1))
        assert predictor.fallback_predictor is not None

        # grid weather and published figures end, the region's weather goes on for two more days
        end = START + timedelta(days=DAYS + 2)
        predicted = await predictor.predict(START + timedelta(days=DAYS - 2), end, fill_known=False)

        residual_load = await predictor.residual_load.get_data(START + timedelta(days=DAYS - 2), end)
        assert residual_load is not None
        without = residual_load["residual_load"].isna()
        assert without.any()
        assert predicted["price"].notna().all()

    @pytest.mark.asyncio
    async def test_no_forecast_where_other_inputs_are_missing_too(self, monkeypatch, region):
        predictor = self.price_predictor(monkeypatch, region, weather_days=DAYS)
        await predictor.train(START, START + timedelta(days=DAYS - 1))

        predicted = await predictor.predict(START + timedelta(days=DAYS - 2), START + timedelta(days=DAYS + 2), fill_known=False)

        assert predicted.loc[START + timedelta(days=DAYS, hours=1):, "price"].isna().all()
        assert predicted.loc[:START + timedelta(days=DAYS - 1), "price"].notna().all()

    @pytest.mark.asyncio
    async def test_no_fallback_models_without_residual_load(self, sample_region, sample_weather_data, sample_price_data,
                                                           sample_aux_data, sample_entsoe_data, sample_gas_data):
        predictor = PricePredictor(sample_region)
        assert predictor.residual_load is None
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
        predictor.entsoestore.get_data = AsyncMock(return_value=sample_entsoe_data)
        predictor.gasstore.get_data = AsyncMock(return_value=sample_gas_data)

        await predictor.train(datetime(2025, 11, 1, tzinfo=timezone.utc), datetime(2025, 11, 2, tzinfo=timezone.utc))

        assert predictor.predictor is not None
        assert predictor.fallback_predictor is None
