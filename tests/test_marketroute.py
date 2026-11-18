"""Tests for the market model as forecaster (with a stand-in for Timestral)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from predictor.model import marketmodel
from predictor.model.marketroute import CARBON_PRICE, EUR_PER_MWH, OTHER_PLANT, MarketRoute
from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
from predictor.model.timestralblend import SLOT, TimestralBlend

LAST_KNOWN = datetime(2025, 11, 1, 22, 45, tzinfo=timezone.utc)
GAS = 30.0


def input_frame(future_slots: int = MarketRoute.HORIZON) -> pd.DataFrame:
    index = pd.date_range(LAST_KNOWN - (MarketRoute.CONTEXT - 1) * SLOT, LAST_KNOWN + future_slots * SLOT, freq="15min", name="time")
    df = pd.DataFrame({"weekday": index.weekday, "gasprice": GAS, "residual_load": 30.0 + 10.0 * np.sin(np.arange(len(index)) / 15.0)}, index=index)
    # prices of a market that had 2 GW more supply than the curves say
    df["price"] = marketmodel.clearing_price(df["residual_load"] - OTHER_PLANT, GAS, CARBON_PRICE, shift=2.0) / EUR_PER_MWH
    df.loc[LAST_KNOWN + SLOT:, "price"] = np.nan
    return df


def route_with_stand_in(df: pd.DataFrame, shift: float = 2.0) -> tuple[MarketRoute, PricePredictor, list]:
    predictor = PricePredictor(PriceRegionName.DE.to_region())
    predictor.pricestore.get_last_known = lambda: pd.Timestamp(LAST_KNOWN)
    predictor.prepare_dataframe = AsyncMock(side_effect=lambda start, end: df.loc[start:end])
    timestral = TimestralBlend("stand-in")
    calls = []

    def forecast(series, covariates, weekday):
        calls.append((series, covariates, weekday))
        return np.full(covariates.shape[1] - len(series), shift)

    timestral._forecast = forecast
    return MarketRoute(timestral), predictor, calls


class TestMarketRoute:
    def test_follows_the_timestral_setting(self, monkeypatch):
        monkeypatch.delenv("EPEXPREDICTOR_MARKET_MODEL", raising=False)
        assert MarketRoute.from_env(None) is None
        assert MarketRoute.from_env(TimestralBlend("stand-in")) is not None
        monkeypatch.setenv("EPEXPREDICTOR_MARKET_MODEL", "false")
        assert MarketRoute.from_env(TimestralBlend("stand-in")) is None

    def test_gas_price_is_the_one_of_two_days_before(self):
        df = input_frame()
        df["gasprice"] = np.arange(len(df), dtype=float)
        gas = MarketRoute.gas_price(df)
        assert (gas[192:] == np.arange(len(df) - 192)).all()
        # nothing older in the first two days: the earliest price there is
        assert (gas[:192] == 0.0).all()

        # a gas price that ends at the last known price still covers all of the forecast
        df.loc[LAST_KNOWN + SLOT:, "gasprice"] = np.nan
        assert np.isfinite(MarketRoute.gas_price(df)).all()

    def test_supply_shift_is_read_off_the_known_prices(self):
        df = input_frame()
        shift = MarketRoute.supply_shift(df.loc[:LAST_KNOWN])
        # on the step at zero price the shift is not pinned down
        clear = df.loc[:LAST_KNOWN, "price"].abs().to_numpy() > 0.1
        assert shift[clear] == pytest.approx(2.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_forecast_clears_the_market_with_the_forecast_shift(self):
        df = input_frame()
        route, predictor, calls = route_with_stand_in(df, shift=2.0)

        forecast = await route.forecast(predictor)

        assert forecast is not None
        assert forecast.index[0] == LAST_KNOWN + SLOT
        assert len(forecast) == MarketRoute.HORIZON
        series, covariates, weekday = calls[0]
        assert len(series) == MarketRoute.CONTEXT
        assert covariates.shape == (1, MarketRoute.CONTEXT + MarketRoute.HORIZON)
        # with the same shift as in the history, the prices continue as the same market gives them
        expected = marketmodel.clearing_price(df.loc[forecast.index, "residual_load"] - OTHER_PLANT, GAS, CARBON_PRICE, shift=2.0) / EUR_PER_MWH
        assert forecast.to_numpy() == pytest.approx(expected, abs=1e-9)

    @pytest.mark.asyncio
    async def test_more_supply_gives_a_lower_price(self):
        df = input_frame()
        low = await route_with_stand_in(df, shift=6.0)[0].forecast(route_with_stand_in(df, shift=6.0)[1])
        high = await route_with_stand_in(df, shift=-6.0)[0].forecast(route_with_stand_in(df, shift=-6.0)[1])
        assert (low < high).all()

    @pytest.mark.asyncio
    async def test_forecast_ends_where_the_residual_load_ends(self):
        df = input_frame()
        df.loc[LAST_KNOWN + 120 * SLOT:, "residual_load"] = np.nan
        route, predictor, _ = route_with_stand_in(df)
        forecast = await route.forecast(predictor)
        assert forecast is not None and len(forecast) == 119

    @pytest.mark.asyncio
    @pytest.mark.parametrize("column", ["residual_load", "gasprice"])
    async def test_no_forecast_without_its_inputs(self, column):
        route, predictor, calls = route_with_stand_in(input_frame().drop(columns=[column]))
        assert await route.forecast(predictor) is None
        assert calls == []

    @pytest.mark.asyncio
    async def test_no_forecast_when_the_residual_load_is_not_forecast(self):
        df = input_frame()
        df.loc[LAST_KNOWN + 10 * SLOT:, "residual_load"] = np.nan
        route, predictor, calls = route_with_stand_in(df)
        assert await route.forecast(predictor) is None
        assert calls == []

    @pytest.mark.asyncio
    async def test_non_finite_output_is_not_used(self):
        route, predictor, _ = route_with_stand_in(input_frame(), shift=float("nan"))
        assert await route.forecast(predictor) is None


class TestBlendWithTheMarketModel:
    @pytest.mark.asyncio
    async def test_three_forecasts_count_a_third_each(self):
        df = input_frame()
        df["wind_0"], df["irradiance_0"], df["temp_0"], df["holiday"] = 20.0, 100.0, 10.0, 0.0
        route, predictor, _ = route_with_stand_in(df, shift=2.0)
        route.timestral.forecast = AsyncMock(return_value=pd.Series(30.0, index=df.index[df.index > LAST_KNOWN][:MarketRoute.HORIZON]))
        market = await route.forecast(predictor)
        prediction = pd.DataFrame({"price": 12.0}, index=df.index)

        assert await route.timestral.blend(predictor, [prediction], {"market": route}) is True

        first = LAST_KNOWN + SLOT
        assert prediction.loc[first, "price"] == pytest.approx((12.0 + 30.0 + market[first]) / 3)
        assert (prediction.loc[:LAST_KNOWN, "price"] == 12.0).all()
        assert set(route.timestral.last_forecasts) == {"timestral", "market"}

    @pytest.mark.asyncio
    async def test_a_failing_market_model_leaves_the_average_of_two(self):
        df = input_frame()
        route, predictor, _ = route_with_stand_in(df)
        route.timestral.forecast = AsyncMock(return_value=pd.Series(30.0, index=df.index[df.index > LAST_KNOWN][:MarketRoute.HORIZON]))
        route.forecast = AsyncMock(side_effect=RuntimeError("no market today"))
        prediction = pd.DataFrame({"price": 12.0}, index=df.index)

        assert await route.timestral.blend(predictor, [prediction], {"market": route}) is True

        assert prediction.loc[LAST_KNOWN + SLOT, "price"] == pytest.approx(21.0)
        assert set(route.timestral.last_forecasts) == {"timestral"}

    @pytest.mark.asyncio
    async def test_the_market_model_alone_counts_half(self):
        df = input_frame()
        route, predictor, _ = route_with_stand_in(df)
        route.timestral.forecast = AsyncMock(return_value=None)
        market = await route.forecast(predictor)
        prediction = pd.DataFrame({"price": 12.0}, index=df.index)

        assert await route.timestral.blend(predictor, [prediction], {"market": route}) is True
        first = LAST_KNOWN + SLOT
        assert prediction.loc[first, "price"] == pytest.approx((12.0 + market[first]) / 2)
