"""Tests for predictor.model.pricepredictor module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName


class TestPricePredictorInit:
    """Tests for PricePredictor initialization."""

    def test_init_creates_stores(self, sample_region):
        """Test that initialization creates all required stores."""
        predictor = PricePredictor(sample_region)
        assert predictor.region == sample_region
        assert predictor.weatherstore is not None
        assert predictor.pricestore is not None
        assert predictor.auxstore is not None
        assert predictor.entsoestore is not None

    def test_init_with_storage_dir(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        predictor = PricePredictor(sample_region, temp_storage_dir)
        assert predictor.weatherstore.storage_dir == temp_storage_dir
        assert predictor.pricestore.storage_dir == temp_storage_dir


class TestPricePredictorGetLastKnownPrice:
    """Tests for get_last_known_price method."""

    def test_get_last_known_price_empty_store(self, sample_region):
        """Test get_last_known_price with empty price store."""
        predictor = PricePredictor(sample_region)
        result = predictor.pricestore.get_last_known()
        assert result is None

    def test_get_last_known_price_with_data(self, sample_region):
        """Test get_last_known_price with data in store."""
        predictor = PricePredictor(sample_region)

        # Add price data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        result = predictor.pricestore.get_last_known()
        assert result is not None
        assert isinstance(result, datetime)


class TestPricePredictorToPriceDict:
    """Tests for to_price_dict method."""

    def test_to_price_dict(self, sample_region):
        """Test conversion of DataFrame to price dictionary."""
        predictor = PricePredictor(sample_region)

        dates = pd.date_range(
            start="2025-11-01", end="2025-11-01T01:00", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0, 9.0, 10.0, 11.0, 12.0]}, index=dates)
        df.index.name = "time"

        result = predictor.to_price_dict(df)

        assert isinstance(result, dict)
        assert len(result) == len(df)
        for dt, price in result.items():
            assert isinstance(dt, datetime)
            assert isinstance(price, float)


class TestPricePredictorPrepareDataframe:
    """Tests for prepare_dataframe method."""

    @pytest.mark.asyncio
    async def test_prepare_dataframe_combines_data(
        self, sample_region, sample_weather_data, sample_price_data, sample_aux_data, sample_entsoe_data,
        sample_gas_data
    ):
        """Test that prepare_dataframe combines all data sources."""
        predictor = PricePredictor(sample_region)

        # Mock the stores to return our sample data
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
        predictor.entsoestore.get_data = AsyncMock(return_value=sample_entsoe_data)
        predictor.gasstore.get_data = AsyncMock(return_value=sample_gas_data)

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        result = await predictor.prepare_dataframe(start, end)

        assert result is not None
        assert not result.empty
        assert "gasprice" in result.columns
        assert not result.loc[start:end, "gasprice"].isna().any()


class TestPricePredictorTrain:
    """Tests for train method."""

    @pytest.mark.asyncio
    async def test_train_creates_model(self, mocked_predictor):
        """Test that training creates a model."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await mocked_predictor.train(start, end)

        assert mocked_predictor.predictor is not None


class TestPricePredictorPredict:
    """Tests for predict method."""

    @pytest.mark.asyncio
    async def test_predict_returns_dataframe(self, mocked_predictor):
        """Test that predict returns a DataFrame with predictions."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict
        result = await mocked_predictor.predict(start, end)

        assert result is not None
        assert not result.empty
        assert "price" in result.columns

    @pytest.mark.asyncio
    async def test_predict_fill_known_true(self, mocked_predictor):
        """Test that predict with fill_known=True uses known prices."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict with fill_known=True
        result = await mocked_predictor.predict(start, end, fill_known=True)

        assert result is not None
        assert not result.empty


class TestPricePredictorGasRelativeModel:
    """Tests for the second model, which predicts price / gas price."""

    def test_gas_scale_without_gas_price(self, sample_region, sample_weather_data):
        """Test that there is nothing to divide by as long as no gas price is available."""
        assert PricePredictor(sample_region).get_gas_scale(sample_weather_data) is None

    def test_gas_scale_is_never_zero(self, sample_region, sample_gas_data):
        """Test that the divisor has a lower bound."""
        gas = sample_gas_data.copy()
        gas["gasprice"] = 0.0
        scale = PricePredictor(sample_region).get_gas_scale(gas)
        assert scale is not None
        assert (scale > 0).all()

    def test_to_gas_relative_converts_cross_prices_only(self, sample_region, sample_gas_data):
        """Test that only cross-region prices change their unit."""
        predictor = PricePredictor(sample_region)
        params = sample_gas_data.copy()
        params["cross_price_FR"] = 12.0
        params["wind_0"] = 5.0

        result = predictor.to_gas_relative(params, predictor.get_gas_scale(params))

        assert (result["cross_price_FR"] == 12.0 / 30.0).all()
        assert (result["wind_0"] == 5.0).all()
        assert (result["gasprice"] == 30.0).all()
        assert (params["cross_price_FR"] == 12.0).all()  # input is left untouched

    @pytest.mark.asyncio
    async def test_trains_both_models_with_gas_price(self, mocked_predictor):
        """Test that a region with gas price data gets the gas-relative model in addition."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await mocked_predictor.train(start, end)

        assert mocked_predictor.predictor is not None
        assert mocked_predictor.gas_predictor is not None

    @pytest.mark.asyncio
    async def test_trains_single_model_without_gas_price(
        self, sample_weather_data, sample_price_data, sample_aux_data, sample_entsoe_data
    ):
        """Test that a region without gas price only models the price itself."""
        region = PriceRegionName.FR.to_region()
        assert not region.use_de_nat_gas_price
        predictor = PricePredictor(region)
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
        predictor.entsoestore.get_data = AsyncMock(return_value=sample_entsoe_data)

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)
        await predictor.train(start, end)
        result = await predictor.predict(start, end, fill_known=False)

        assert predictor.predictor is not None
        assert predictor.gas_predictor is None
        assert not result["price"].isna().any()

    @pytest.mark.asyncio
    async def test_prediction_follows_gas_price(self, mocked_predictor, sample_gas_data):
        """Test that a gas price level unseen in training moves the forecast, but less than proportionally."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await mocked_predictor.train(start, end)
        before = await mocked_predictor.predict(start, end, fill_known=False)

        mocked_predictor.gasstore.get_data = AsyncMock(return_value=sample_gas_data * 2)
        after = await mocked_predictor.predict(start, end, fill_known=False)

        # the gas-relative half of the forecast doubles, the other half stays
        assert (before["price"] > 0).all()
        assert (after["price"] > before["price"]).all()
        assert (after["price"] < before["price"] * 2).all()


class TestPricePredictorCrossFeatures:
    """Tests for the features a model gets from other regions."""

    @staticmethod
    def _weather(wind, irradiance, temp):
        dates = pd.date_range(start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC")
        data = {}
        for i, w in enumerate(wind):
            data[f"wind_{i}"] = [w] * len(dates)
            data[f"irradiance_{i}"] = [irradiance] * len(dates)
            data[f"temp_{i}"] = [temp] * len(dates)
        return pd.DataFrame(data, index=dates)

    def _cross_predictor(self, name, weather, price=10.0):
        cp = PricePredictor(name.to_region())
        cp.weatherstore.get_data = AsyncMock(return_value=weather)
        cp.predict = AsyncMock(return_value=pd.DataFrame({"price": [price] * len(weather)}, index=weather.index))
        return cp

    @pytest.mark.asyncio
    async def test_stage2_dataframe(self, mocked_predictor):
        """Test that a stage-2 model gets the price forecasts of the stage-1 models."""
        mocked_predictor.cross_predictors = [
            self._cross_predictor(PriceRegionName.FR, self._weather([22.5], 300.0, 20.0), price=12.0)
        ]

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)
        result = await mocked_predictor.prepare_dataframe(start, end)

        assert result is not None
        assert result["cross_price_FR"].iloc[0] == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_dataframe_without_other_regions(self, mocked_predictor):
        """Test that a standalone model only uses its own region's data."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)
        result = await mocked_predictor.prepare_dataframe(start, end)

        assert result is not None
        assert not any(c.startswith("cross_price_") for c in result.columns)

class TestPricePredictorCleanup:
    """Tests for cleanup method."""

    def test_cleanup_removes_old_data(self, sample_region):
        """Test that cleanup removes data older than 5 years."""
        predictor = PricePredictor(sample_region)

        # Add old data (more than 5 years ago)
        old_dates = pd.date_range(
            start="2020-01-01", end="2020-01-02", freq="15min", tz="UTC"
        )
        old_df = pd.DataFrame({"price": [8.0] * len(old_dates)}, index=old_dates)
        old_df.index.name = "time"
        predictor.pricestore._update_data(old_df)

        # Add recent data
        recent_dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        recent_df = pd.DataFrame({"price": [9.0] * len(recent_dates)}, index=recent_dates)
        recent_df.index.name = "time"
        predictor.pricestore._update_data(recent_df)

        # Cleanup
        predictor.cleanup()

        # Old data (2020) should be removed, recent data (2025) kept
        assert predictor.pricestore.data.index.min() > pd.Timestamp("2021-01-01", tz="UTC")


class TestPricePredictorRefreshMethods:
    """Tests for refresh_prices and refresh_forecasts methods."""


    @pytest.mark.asyncio
    async def test_refresh_forecasts(self, sample_region):
        """Test refresh_forecasts method."""
        predictor = PricePredictor(sample_region)

        # Mock the weather store's refresh_range method (not fetch_missing_data)
        predictor.weatherstore.refresh_range = AsyncMock()

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 8, tzinfo=timezone.utc)

        await predictor.refresh_forecasts(start, end)

        # Should have called refresh_range
        assert predictor.weatherstore.refresh_range.called


