"""Tests for the optional Timestral blend (with a stand-in for the model itself)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
from predictor.model.timestralblend import SLOT, TimestralBlend

LAST_KNOWN = datetime(2025, 11, 1, 22, 45, tzinfo=timezone.utc)


def input_frame(future_slots: int) -> pd.DataFrame:
    index = pd.date_range(LAST_KNOWN - (TimestralBlend.CONTEXT - 1) * SLOT, LAST_KNOWN + future_slots * SLOT, freq="15min", name="time")
    df = pd.DataFrame({
        "wind_0": 20.0, "wind_1": 30.0, "wind_6": 40.0, "irradiance_0": 100.0, "temp_0": 10.0,
        "holiday": 0.0, "weekday": index.weekday, "eu_wind": 0.3, "gasprice": 30.0, "price": 10.0,
    }, index=index)
    df.loc[LAST_KNOWN + SLOT:, "price"] = np.nan
    return df


def blend_with_stand_in(df: pd.DataFrame, value: float = 20.0) -> tuple[TimestralBlend, PricePredictor, list]:
    predictor = PricePredictor(PriceRegionName.DE.to_region())
    predictor.pricestore.get_last_known = lambda: pd.Timestamp(LAST_KNOWN)
    predictor.prepare_dataframe = AsyncMock(side_effect=lambda start, end: df.loc[start:end])
    blend = TimestralBlend("stand-in")
    calls = []

    def forecast(prices, covariates, weekday):
        calls.append((prices, covariates, weekday))
        return np.full(covariates.shape[1] - len(prices), value)

    blend._forecast = forecast
    return blend, predictor, calls


class TestTimestralBlend:
    def test_off_without_setting(self, monkeypatch):
        monkeypatch.delenv("EPEXPREDICTOR_TIMESTRAL_MODEL", raising=False)

        assert TimestralBlend.from_env() is None

    def test_off_when_the_package_is_missing(self, monkeypatch):
        monkeypatch.setenv("EPEXPREDICTOR_TIMESTRAL_MODEL", "mistralai/Timestral-123M")
        monkeypatch.setitem(__import__("sys").modules, "timestral", None)

        assert TimestralBlend.from_env() is None

    def test_weights_fade_out(self):
        weights = TimestralBlend.weights()

        assert len(weights) == TimestralBlend.HORIZON
        assert (weights[:TimestralBlend.FULL_WEIGHT_SLOTS] == 0.5).all()
        assert weights[-1] == 0.0
        assert (np.diff(weights) <= 0).all()

    @pytest.mark.asyncio
    async def test_forecast_covers_the_time_after_the_last_known_price(self):
        blend, predictor, calls = blend_with_stand_in(input_frame(TimestralBlend.HORIZON))

        forecast = await blend.forecast(predictor)

        assert forecast is not None
        assert forecast.index[0] == LAST_KNOWN + SLOT
        assert len(forecast) == TimestralBlend.HORIZON
        prices, covariates, weekday = calls[0]
        assert len(prices) == TimestralBlend.CONTEXT
        assert not np.isnan(prices).any()
        assert covariates.shape[1] == len(weekday) == TimestralBlend.CONTEXT + TimestralBlend.HORIZON
        assert not np.isnan(covariates).any()

    @pytest.mark.asyncio
    async def test_forecast_ends_where_the_weather_forecast_ends(self):
        df = input_frame(TimestralBlend.HORIZON)
        df.loc[LAST_KNOWN + 120 * SLOT:, "wind_0"] = np.nan
        blend, predictor, _ = blend_with_stand_in(df)

        forecast = await blend.forecast(predictor)

        assert forecast is not None
        assert len(forecast) == 119

    @pytest.mark.asyncio
    async def test_no_forecast_without_weather_forecast(self):
        df = input_frame(TimestralBlend.HORIZON)
        df.loc[LAST_KNOWN + 10 * SLOT:, "wind_0"] = np.nan
        blend, predictor, calls = blend_with_stand_in(df)

        assert await blend.forecast(predictor) is None
        assert calls == []

    @pytest.mark.asyncio
    async def test_blend_averages_after_the_last_known_price_only(self):
        blend, predictor, _ = blend_with_stand_in(input_frame(TimestralBlend.HORIZON), value=20.0)
        index = pd.date_range(LAST_KNOWN - timedelta(days=1), LAST_KNOWN + timedelta(days=3), freq="15min")
        prediction = pd.DataFrame({"price": 10.0}, index=index)
        prediction.loc[LAST_KNOWN + 5 * SLOT, "price"] = np.nan

        assert await blend.blend(predictor, [prediction]) is True

        assert (prediction.loc[:LAST_KNOWN, "price"] == 10.0).all()
        assert prediction.loc[LAST_KNOWN + SLOT, "price"] == 15.0
        assert prediction.loc[LAST_KNOWN + TimestralBlend.FULL_WEIGHT_SLOTS * SLOT, "price"] == 15.0
        assert 10.0 < prediction.loc[LAST_KNOWN + (TimestralBlend.FULL_WEIGHT_SLOTS + 20) * SLOT, "price"] < 15.0
        assert (prediction.loc[LAST_KNOWN + TimestralBlend.HORIZON * SLOT:, "price"] == 10.0).all()
        # a time without forecast of the tree models stays without forecast
        assert np.isnan(prediction.loc[LAST_KNOWN + 5 * SLOT, "price"])

    @pytest.mark.asyncio
    async def test_failure_leaves_the_forecast_alone(self):
        blend, predictor, _ = blend_with_stand_in(input_frame(TimestralBlend.HORIZON))

        def failing(prices, covariates, weekday):
            raise RuntimeError("out of memory")

        blend._forecast = failing
        prediction = pd.DataFrame({"price": 10.0}, index=pd.date_range(LAST_KNOWN, periods=50, freq="15min"))

        assert await blend.blend(predictor, [prediction]) is False
        assert (prediction["price"] == 10.0).all()
