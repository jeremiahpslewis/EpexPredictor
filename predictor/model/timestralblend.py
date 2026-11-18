"""
Optional second opinion: the Timestral time series foundation model (https://github.com/mistralai/timestral).

It forecasts the price from its recent course plus the weather and the residual load as known-future covariates,
without any training.
Where the tree models read the price off the weather, it continues the last known prices, so it is best right
after them, and the two make different mistakes. In backtests for Germany, their average improved on the tree models
for the night hours in autumn and in spring/summer, and for the whole first day in autumn only; on its own,
Timestral was not better than the tree models.

Switched on with EPEXPREDICTOR_TIMESTRAL_MODEL (e.g. mistralai/Timestral-123M) and needs the packages of
requirements-timestral.txt.
"""

import asyncio
import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from .europeweather import RATED_WIND_SPEED
from .pricepredictor import PricePredictor

log = logging.getLogger(__name__)

SLOT = timedelta(minutes=15)

# Model weights the backtests were run with. Other models, or other revisions, are loaded as published now.
TESTED_REVISIONS = {
    "mistralai/Timestral-28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "mistralai/Timestral-123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "mistralai/Timestral-209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


class TimestralBlend:
    # Quarter hours of price history the model looks at (about 85 days)
    CONTEXT = 8192
    # Quarter hours after the last known price that get the full weight, and where the weight has faded to zero.
    # Backtested up to 38 hours ahead - further out, the tree models are on their own.
    FULL_WEIGHT_SLOTS = 152
    HORIZON = 192
    WEIGHT = 0.5

    model_name: str
    revision: str | None
    forecaster = None
    # the forecasts of the last blend, by name
    last_forecasts: dict[str, pd.Series] = {}

    def __init__(self, model_name: str, revision: str | None = None):
        self.model_name = model_name
        self.revision = revision if revision else TESTED_REVISIONS.get(model_name)

    @staticmethod
    def from_env() -> "TimestralBlend | None":
        model_name = os.getenv("EPEXPREDICTOR_TIMESTRAL_MODEL", "").strip()
        if len(model_name) == 0:
            return None
        try:
            import timestral  # noqa: F401
        except ImportError:
            log.error("EPEXPREDICTOR_TIMESTRAL_MODEL is set, but the timestral package is not installed "
                      "(see requirements-timestral.txt). Continuing without it.")
            return None
        return TimestralBlend(model_name, os.getenv("EPEXPREDICTOR_TIMESTRAL_REVISION", "").strip() or None)

    @classmethod
    def weights(cls) -> np.ndarray:
        """Weight of the Timestral forecast for each quarter hour after the last known price"""
        fade = np.linspace(1.0, 0.0, cls.HORIZON - cls.FULL_WEIGHT_SLOTS + 1)[1:]
        return cls.WEIGHT * np.concatenate([np.ones(cls.FULL_WEIGHT_SLOTS), fade])

    # The residual load [GW] as published and forecast (see residualload.py). In backtests of the next day from
    # midnight, Timestral with it was better than without by a fifth over the whole day.
    RESIDUAL_LOAD = "residual_load"

    @staticmethod
    def covariates(predictor: PricePredictor, df: pd.DataFrame, residual_load: bool = True) -> np.ndarray:
        """What is known about the future, one row per covariate"""
        wind = df.filter(regex=r"^wind_\d+$")
        onshore = len(predictor.region.latitudes)
        rows = [
            wind.iloc[:, :onshore].mean(axis=1, skipna=False),
            ((wind.clip(lower=0, upper=RATED_WIND_SPEED) / RATED_WIND_SPEED) ** 3).mean(axis=1, skipna=False),
            df.filter(regex=r"^irradiance_\d+$").mean(axis=1, skipna=False),
            df.filter(regex=r"^temp_\d+$").mean(axis=1, skipna=False),
            (df["holiday"] >= 0.5).astype(float),
        ]
        if wind.shape[1] > onshore:
            rows.append(wind.iloc[:, onshore:].mean(axis=1, skipna=False))
        for column in ("eu_wind", "eu_wind_northsea", "eu_irradiance", "gasprice"):
            if column in df.columns:
                rows.append(df[column])
        if residual_load and TimestralBlend.RESIDUAL_LOAD in df.columns:
            rows.append(df[TimestralBlend.RESIDUAL_LOAD])
        return np.vstack([row.to_numpy(dtype=float) for row in rows])

    @staticmethod
    def known_horizon(covariates: np.ndarray, history: int) -> int:
        """Quarter hours after the history for which every covariate is known"""
        known = ~np.isnan(covariates[:, history:]).any(axis=0)
        return int(np.argmin(known)) if not known.all() else int(known.size)

    def _forecast(self, prices: np.ndarray, covariates: np.ndarray, weekday: np.ndarray) -> np.ndarray:
        import torch
        from timestral import TimestralForecaster

        if self.forecaster is None:
            log.info(f"loading {self.model_name}, revision {self.revision or 'latest'}")
            self.forecaster = TimestralForecaster.from_pretrained(self.model_name, revision=self.revision)
        with torch.no_grad():
            forecast = self.forecaster.forecast(
                prices[None, None, :],
                prediction_length=covariates.shape[1] - len(prices),
                known_covariates=covariates[None],
                categorical_known_covariates=weekday[None, None, :],
            )
        return np.asarray(forecast.median)[0, 0]

    async def forecast(self, predictor: PricePredictor) -> pd.Series | None:
        """
        Forecast for the quarter hours after the last known price, or None if it cannot be made.

        predictor: a model of the region without cross predictors - only its input data is used
        """
        last_known = predictor.pricestore.get_last_known()
        if last_known is None:
            return None

        first = last_known + SLOT
        df = await predictor.prepare_dataframe(first - self.CONTEXT * SLOT, last_known + self.HORIZON * SLOT)
        if df is None:
            return None
        history, future = df.loc[:last_known], df.loc[first:]
        # needs all of the future it is asked for, and most of the history (it copes with gaps in there)
        covariates = self.covariates(predictor, df)
        horizon = self.known_horizon(covariates, len(history))
        if self.RESIDUAL_LOAD in df.columns:
            # an optional input: where it ends earlier than the others, the forecast is made without it
            without = self.covariates(predictor, df, residual_load=False)
            if self.known_horizon(without, len(history)) > horizon:
                log.info(f"{predictor.region.bidding_zone_entsoe}: residual load is not known far enough - Timestral forecast without it")
                covariates = without
                horizon = self.known_horizon(without, len(history))
        if horizon < 96 or len(history) < self.CONTEXT // 2 or history["price"].isna().mean() > 0.1:
            log.warning(f"{predictor.region.bidding_zone_entsoe}: not enough data for a Timestral forecast "
                        f"({len(history)} quarter hours of history, {history['price'].isna().mean():.0%} of them without price, "
                        f"inputs known for {horizon} quarter hours ahead)")
            return None

        rows = len(history) + horizon
        # gaps in the history stay NaN: Timestral masks them (a zero would read as no wind, no sun, ...)
        covariates = covariates[:, :rows]
        values = await asyncio.to_thread(
            self._forecast,
            history["price"].to_numpy(dtype=float),
            covariates,
            df["weekday"].to_numpy(dtype=float)[:rows],
        )
        values = np.asarray(values, dtype=float)
        if values.shape != (horizon,) or not np.isfinite(values).all():
            log.warning(f"{predictor.region.bidding_zone_entsoe}: Timestral returned an unusable forecast "
                        f"(shape {values.shape}, {int(np.size(values) - np.isfinite(values).sum())} non-finite values) - not used")
            return None
        return pd.Series(values, index=future.index[:horizon], name="price")

    async def _forecast_or_none(self, name: str, forecaster, predictor: PricePredictor) -> pd.Series | None:
        try:
            return await forecaster.forecast(predictor)
        except Exception:
            log.exception(f"{predictor.region.bidding_zone_entsoe}: {name} forecast failed - continuing without it")
            return None

    async def blend(self, predictor: PricePredictor, predictions: list[pd.DataFrame], others: dict | None = None) -> bool:
        """
        Mix the Timestral forecast into the price forecasts (in place). Returns whether anything was mixed in.
        A failure only costs the second opinion, never the forecast itself.

        others: further forecasters by name (objects with the same forecast method, e.g. the market model). Each
                one that delivers gets the same weight as Timestral and the tree models.

        What was mixed is kept in last_forecasts, by name, for the forecast log.
        """
        forecasts: dict[str, pd.Series] = {}
        for name, forecaster in {"timestral": self, **(others or {})}.items():
            forecast = await self._forecast_or_none(name, forecaster, predictor)
            if forecast is not None:
                forecasts[name] = forecast
        self.last_forecasts = forecasts
        if len(forecasts) == 0:
            return False

        # a quarter hour with k further forecasts: each of them and the tree models count 1 / (k + 1) while the weight
        # is full, and the tree models take over as it fades out
        full = self.weights() / self.WEIGHT
        for prediction in predictions:
            wanted = prediction.index[prediction["price"].notna()]
            total = pd.Series(0.0, index=wanted)
            count = pd.Series(0.0, index=wanted)
            fade = pd.Series(0.0, index=wanted)
            for forecast in forecasts.values():
                common = forecast.index.intersection(wanted)
                total[common] += forecast[common]
                count[common] += 1.0
                fade[common] = np.maximum(fade[common], pd.Series(full[:len(forecast)], index=forecast.index)[common])
            mixed = count > 0
            share = fade[mixed] * count[mixed] / (count[mixed] + 1.0)
            prediction.loc[mixed[mixed].index, "price"] = (1.0 - share) * prediction.loc[mixed[mixed].index, "price"] + share * total[mixed] / count[mixed]
        log.info(f"{predictor.region.bidding_zone_entsoe}: blended " + ", ".join(f"{name} ({len(forecast)} quarter hours)" for name, forecast in forecasts.items()))
        return True
