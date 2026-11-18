"""Tests for the residual load forecast of Timestral (with a stand-in for Timestral)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
from predictor.model.timestralblend import SLOT, TimestralBlend
from predictor.model.timestralload import TimestralLoad
from tests.test_residualload import forecaster, grid_weather, published_figures

START = datetime(2025, 6, 1, tzinfo=timezone.utc)
DAYS = 95
LAST_PUBLISHED = pd.Timestamp(START + timedelta(days=90)) - SLOT


@pytest.fixture
def region():
    return PriceRegionName.DE.to_region()


def load_with_stand_in(region, weather: pd.DataFrame | None = None, values=(10.0, -1.0, 50.0)):
    """Timestral answers with a constant for wind, solar (negative: to be clipped) and load"""
    weather = grid_weather(region, START, DAYS) if weather is None else weather
    published = published_figures(grid_weather(region, START, DAYS)).loc[:LAST_PUBLISHED]
    residual_load = forecaster(region, weather, published)
    residual_load.generation.get_last_known = lambda: LAST_PUBLISHED
    timestral = TimestralBlend("stand-in")
    calls = []

    def forecast(series, covariates, weekday):
        calls.append((series, covariates, weekday))
        return np.tile(np.array(values)[:, None], (1, covariates.shape[1] - series.shape[1]))

    timestral._forecast_series = forecast
    return TimestralLoad(timestral), residual_load, calls


class TestTimestralLoad:
    def test_follows_the_timestral_setting(self, monkeypatch):
        timestral = TimestralBlend("stand-in")
        assert TimestralLoad.from_env(None) is None
        assert TimestralLoad.from_env(timestral) is not None
        monkeypatch.setenv("EPEXPREDICTOR_TIMESTRAL_LOAD", "false")
        assert TimestralLoad.from_env(timestral) is None

    @pytest.mark.asyncio
    async def test_continues_the_published_figures(self, region):
        load, residual_load, calls = load_with_stand_in(region)
        figures = await load.figures(residual_load)

        assert figures is not None and len(figures) == TimestralLoad.HORIZON
        assert figures.index[0] == LAST_PUBLISHED + SLOT
        assert (figures["tso_wind"] == 10.0).all() and (figures["tso_load"] == 50.0).all()
        # solar power cannot be negative
        assert (figures["tso_solar"] == 0.0).all()

        series, covariates, weekday = calls[0]
        assert series.shape == (3, TimestralLoad.CONTEXT)
        # wind, solar and load as published, in this order
        assert series[2] == pytest.approx(55.0)
        assert series[0, -1] == pytest.approx(published_figures(grid_weather(region, START, DAYS)).loc[LAST_PUBLISHED, "tso_wind"])
        # onshore all / north / south, North Sea, Baltic Sea, radiation all / south, temperature, holiday
        assert covariates.shape == (9, TimestralLoad.CONTEXT + TimestralLoad.HORIZON)
        assert np.isfinite(covariates).all()
        assert covariates[7] == pytest.approx(15.0)
        assert len(weekday) == covariates.shape[1]

        residual = await load.residual_load(residual_load)
        assert (residual == 40.0).all() and residual.name == "residual_load"

    @pytest.mark.asyncio
    async def test_is_made_once_for_each_state_of_the_data(self, region):
        load, residual_load, calls = load_with_stand_in(region)
        await load.figures(residual_load)
        await load.figures(residual_load)
        assert len(calls) == 1

        residual_load.gridweather.last_updated = datetime.now(timezone.utc)
        await load.figures(residual_load)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_ends_where_the_weather_forecast_ends(self, region):
        weather = grid_weather(region, START, DAYS).loc[:LAST_PUBLISHED + timedelta(hours=30)]
        load, residual_load, _ = load_with_stand_in(region, weather)
        figures = await load.figures(residual_load)
        # the last published figure is for a quarter to the hour, the hourly weather ends on the hour
        assert figures is not None and len(figures) == 30 * 4 - 3

    @pytest.mark.asyncio
    async def test_no_forecast_without_weather_forecast(self, region):
        weather = grid_weather(region, START, DAYS).loc[:LAST_PUBLISHED]
        load, residual_load, calls = load_with_stand_in(region, weather)
        assert await load.figures(residual_load) is None
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_unusable_answer_is_not_used(self, region):
        load, residual_load, _ = load_with_stand_in(region, values=(10.0, np.nan, 50.0))
        assert await load.residual_load(residual_load) is None

    @pytest.mark.asyncio
    async def test_failure_leaves_the_tree_forecast(self, region):
        load, residual_load, _ = load_with_stand_in(region)
        load.timestral._forecast_series = lambda *args: 1 / 0
        predictor = PricePredictor(region)
        predictor.residual_load = residual_load
        index = pd.date_range(LAST_PUBLISHED - 3 * SLOT, periods=20, freq="15min")
        df = pd.DataFrame({"residual_load": 33.0}, index=index)
        assert (await load.apply(predictor, df))["residual_load"].eq(33.0).all()

    @pytest.mark.asyncio
    async def test_replaces_the_residual_load_after_the_last_published_figure(self, region):
        load, residual_load, _ = load_with_stand_in(region)
        predictor = PricePredictor(region)
        predictor.residual_load = residual_load
        index = pd.date_range(LAST_PUBLISHED - 3 * SLOT, periods=TimestralLoad.HORIZON + 20, freq="15min")
        df = pd.DataFrame({"residual_load": 33.0, "price": 1.0}, index=index)

        result = await load.apply(predictor, df)
        assert (result.loc[:LAST_PUBLISHED, "residual_load"] == 33.0).all()
        # the average of Timestral's 40 and the 33 of the tree models, fading to the tree models at the end
        after = result.loc[LAST_PUBLISHED + SLOT:, "residual_load"].to_numpy()
        steady = TimestralLoad.HORIZON - TimestralLoad.FADE_OUT
        assert after[:steady + 1] == pytest.approx(36.5)
        assert (np.diff(after[steady:TimestralLoad.HORIZON]) < 0).all()
        assert after[TimestralLoad.HORIZON - 1] < 33.1
        assert (after[TimestralLoad.HORIZON:] == 33.0).all()

        # without a forecast of the tree models, Timestral's stands alone
        df.loc[LAST_PUBLISHED + 10 * SLOT, "residual_load"] = np.nan
        assert (await load.apply(predictor, df)).loc[LAST_PUBLISHED + 10 * SLOT, "residual_load"] == 40.0
        df.loc[LAST_PUBLISHED + 10 * SLOT, "residual_load"] = 33.0
        # the frame that was passed in is left as it was
        assert (df["residual_load"] == 33.0).all()

    @pytest.mark.asyncio
    async def test_the_price_forecasters_get_it(self, region):
        load, residual_load, _ = load_with_stand_in(region)
        predictor = PricePredictor(region)
        predictor.residual_load = residual_load
        index = pd.date_range(LAST_PUBLISHED - 3 * SLOT, periods=40, freq="15min")
        predictor.prepare_dataframe = AsyncMock(return_value=pd.DataFrame({"residual_load": 33.0}, index=index))

        timestral = load.timestral
        assert (await timestral.inputs(predictor, index[0], index[-1]))["residual_load"].eq(33.0).all()
        timestral.load = load
        assert (await timestral.inputs(predictor, index[0], index[-1])).loc[LAST_PUBLISHED + SLOT, "residual_load"] == 36.5

    @pytest.mark.asyncio
    async def test_comparison_for_the_log(self, region):
        load, residual_load, _ = load_with_stand_in(region)
        residual_load.LGB_ROUNDS = 20
        await residual_load.train(START, START + timedelta(days=DAYS))
        last, columns = await load.comparison(residual_load)
        assert last == LAST_PUBLISHED
        assert (columns["timestral"] == 40.0).all()
        assert len(columns["trees"]) == TimestralLoad.HORIZON
        # the tree forecast ends with the hourly weather, on the full hour
        assert columns["trees"].iloc[:-3].notna().all()
