"""
Optional: the residual load of the next day from Timestral, for the two forecasters that run on Timestral.

The transmission system operators publish their forecasts of wind power, solar power and load one day ahead. For
the time after that, residualload.py reads the three figures off the weather forecast with tree models. Here,
Timestral continues the three published series, jointly, with a few aggregates of the same weather forecast as
inputs known for the future. It starts from the last published level, which the tree models do not know.

Right after the last published figure Timestral is the better of the two, later neither is, and they make
different mistakes. So the two are mixed: Timestral alone at the last published figure, half and half from a day
later on, with a linear change in between. In backtests of three seasons (146 days, forecasts made at midnight
for the day that follows) this was better than either alone and than a plain average where it matters most for
the price forecast, in the night hours: against the plain average, the price forecast's error at night was 8 %
lower in winter and 4 % lower in summer, and unchanged in autumn, with no clear change over the whole day. How
much weight the tree models deserve depends on how old their weather forecast is, which the backtests could only
approximate - the log of both forecasts (see priceapi.py) shows it for live operation.

Used by Timestral on the price (timestralblend.py) and by the market model (marketroute.py). The tree models of
the price keep the residual load of residualload.py.

Runs with the Timestral model that is switched on with EPEXPREDICTOR_TIMESTRAL_MODEL, unless
EPEXPREDICTOR_TIMESTRAL_LOAD is set to false.
"""

import asyncio
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .pricepredictor import PricePredictor
from .residualload import ResidualLoad
from .timestralblend import SLOT, TimestralBlend

log = logging.getLogger(__name__)

# Offshore locations west of this longitude are in the North Sea, the others in the Baltic Sea
NORTH_SEA_LONGITUDE = 10.0


class TimestralLoad:
    # Quarter hours of published figures the model looks at (about 85 days)
    CONTEXT = 8192
    # Quarter hours after the last published figure that Timestral is asked for (two and a half days, as backtested)
    HORIZON = 240
    # Weight of Timestral against the tree models: START at the last published figure, falling to WEIGHT over the
    # first LEAD quarter hours. Over the last FADE_OUT quarter hours of the horizon it falls to zero, so that there
    # is no step where the tree models carry on alone.
    START = 1.0
    WEIGHT = 0.5
    LEAD = 96
    FADE_OUT = 48
    TARGETS = ResidualLoad.TARGETS

    timestral: TimestralBlend
    # the last forecast and what it was made from
    cached: tuple[tuple, pd.DataFrame | None] | None = None
    lock: asyncio.Lock

    def __init__(self, timestral: TimestralBlend):
        self.timestral = timestral
        self.lock = asyncio.Lock()

    @staticmethod
    def from_env(timestral: TimestralBlend | None) -> "TimestralLoad | None":
        if timestral is None:
            return None
        if os.getenv("EPEXPREDICTOR_TIMESTRAL_LOAD", "true").strip().lower() in ("no", "false", "f", "0"):
            return None
        return TimestralLoad(timestral)

    @staticmethod
    def covariates(residual_load: ResidualLoad, features: pd.DataFrame) -> np.ndarray:
        """What is known about the future, one row per covariate: wind power, radiation and temperature by area"""
        region = residual_load.region
        onshore = len(region.generation_latitudes)
        north_sea = np.array(region.generation_offshore_longitudes) < NORTH_SEA_LONGITUDE
        offshore = features[[f"wind_power_{onshore + i}" for i in range(len(north_sea))]].to_numpy(dtype=float)
        rows = [features[name].to_numpy(dtype=float) for name in ("wind_power_onshore", "wind_power_north", "wind_power_south")]
        for area in (north_sea, ~north_sea):
            if area.any():
                rows.append(offshore[:, area].mean(axis=1))
        rows += [features[name].to_numpy(dtype=float) for name in ("radiation_all", "radiation_south", "temp_all")]
        rows.append((features["holiday"].to_numpy(dtype=float) >= 0.5).astype(float))
        return np.vstack(rows)

    async def figures(self, residual_load: ResidualLoad) -> pd.DataFrame | None:
        """
        Wind power, solar power and load [GW] for the quarter hours after the last published figures, or None if
        the forecast cannot be made. Made once for each state of the published figures and the weather forecast.
        """
        last: datetime | None = residual_load.generation.get_last_known()
        if last is None:
            return None
        async with self.lock:
            key = (last, residual_load.gridweather.last_updated)
            if self.cached is not None and self.cached[0] == key:
                return self.cached[1]
            figures = await self._figures(residual_load, pd.Timestamp(last))
            self.cached = (key, figures)
            return figures

    async def _figures(self, residual_load: ResidualLoad, last: pd.Timestamp) -> pd.DataFrame | None:
        zone = residual_load.region.bidding_zone_entsoe
        first = last + SLOT
        # the weather is hourly: asked for up to the next full hour, so that the last quarter hours are covered as well
        features = await residual_load.get_features(first - self.CONTEXT * SLOT, last + (self.HORIZON + 3) * SLOT)
        published = await residual_load.generation.get_data(first - self.CONTEXT * SLOT, last)
        history = features.loc[:last]
        published = published.reindex(index=history.index, columns=list(self.TARGETS))

        covariates = self.covariates(residual_load, features)
        horizon = min(TimestralBlend.known_horizon(covariates, len(history)), self.HORIZON)
        missing = float(published.isna().any(axis=1).mean()) if len(history) > 0 else 1.0
        if horizon < 4 or len(history) < self.CONTEXT // 2 or missing > 0.1:
            log.warning(f"{zone}: not enough data for a Timestral forecast of the residual load ({len(history)} quarter hours "
                        f"of history, {missing:.0%} of them without published figures, weather known for {horizon} quarter hours "
                        f"after the last published figure) - the forecast of the tree models is used")
            return None

        rows = len(history) + horizon
        values = await asyncio.to_thread(
            self.timestral._forecast_series,
            published.to_numpy(dtype=float).T,
            covariates[:, :rows],
            features["weekday"].to_numpy(dtype=float)[:rows],
        )
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.TARGETS), horizon) or not np.isfinite(values).all():
            log.warning(f"{zone}: Timestral returned an unusable forecast of the residual load (shape {values.shape}, "
                        f"{int(np.size(values) - np.isfinite(values).sum())} non-finite values) - the forecast of the tree models is used")
            return None

        figures = pd.DataFrame(values.T, index=features.index[len(history):rows], columns=list(self.TARGETS))
        # wind and solar power cannot be negative
        figures[["tso_wind", "tso_solar"]] = figures[["tso_wind", "tso_solar"]].clip(lower=0.0)
        log.info(f"{zone}: Timestral forecast of the residual load for {horizon} quarter hours after {last.isoformat()}")
        return figures

    async def residual_load(self, residual_load: ResidualLoad) -> pd.Series | None:
        """The residual load [GW] after the last published figures, or None. Never raises."""
        try:
            figures = await self.figures(residual_load)
        except Exception:
            log.exception(f"{residual_load.region.bidding_zone_entsoe}: Timestral forecast of the residual load failed - "
                          f"the forecast of the tree models is used")
            return None
        if figures is None:
            return None
        return (figures["tso_load"] - figures["tso_wind"] - figures["tso_solar"]).rename(ResidualLoad.FEATURE)

    @classmethod
    def weights(cls, slots: int) -> np.ndarray:
        """Weight of Timestral for each of the quarter hours after the last published figure"""
        ahead = np.arange(slots)
        near = cls.START + (cls.WEIGHT - cls.START) * np.clip(ahead / cls.LEAD, 0.0, 1.0)
        return near * np.clip((cls.HORIZON - ahead) / cls.FADE_OUT, 0.0, 1.0)

    async def apply(self, predictor: PricePredictor, df: pd.DataFrame) -> pd.DataFrame:
        """
        df with its residual load after the last published figures mixed with this forecast, where there is one.
        What is in df there is the forecast of the tree models.
        """
        if predictor.residual_load is None or ResidualLoad.FEATURE not in df.columns:
            return df
        forecast = await self.residual_load(predictor.residual_load)
        if forecast is None:
            return df
        weight = pd.Series(self.weights(len(forecast)), index=forecast.index)
        common = forecast.index[weight.to_numpy() > 0].intersection(df.index)
        if len(common) == 0:
            return df
        df = df.copy()
        trees = df.loc[common, ResidualLoad.FEATURE]
        mixed = weight[common] * forecast[common] + (1.0 - weight[common]) * trees
        # without a forecast of the tree models, Timestral's stands alone
        df.loc[common, ResidualLoad.FEATURE] = mixed.where(trees.notna(), forecast[common])
        return df

    async def comparison(self, residual_load: ResidualLoad) -> tuple[datetime, dict[str, pd.Series]] | None:
        """
        For the forecast log: the last time with published figures, and the residual load after it as forecast by
        the tree models and by Timestral. None if there is no Timestral forecast. Never raises.
        """
        try:
            last = residual_load.generation.get_last_known()
            forecast = await self.residual_load(residual_load)
            if last is None or forecast is None:
                return None
            trees = await residual_load.get_data(forecast.index[0], forecast.index[-1])
            columns = {"timestral": forecast}
            if trees is not None:
                columns["trees"] = trees[ResidualLoad.FEATURE]
            return last, columns
        except Exception:
            log.exception(f"{residual_load.region.bidding_zone_entsoe}: could not put together the residual load forecasts for the log")
            return None
