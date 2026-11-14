"""
Optional second opinion: the Timestral time series foundation model (https://github.com/mistralai/timestral).

It forecasts the price from its recent course plus the weather as known-future covariates, without any training.
Where the tree models read the price off the weather, it continues the last known prices, so it is best right
after them, and the two make different mistakes. Their average was better than either one in backtests.

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


class TimestralBlend:
    # Quarter hours of price history the model looks at (about 85 days)
    CONTEXT = 8192
    # Quarter hours after the last known price that get the full weight, and where the weight has faded to zero.
    # Backtested up to 38 hours ahead - further out, the tree models are on their own.
    FULL_WEIGHT_SLOTS = 152
    HORIZON = 192
    WEIGHT = 0.5

    model_name: str
    forecaster = None

    def __init__(self, model_name: str):
        self.model_name = model_name

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
        return TimestralBlend(model_name)

    @classmethod
    def weights(cls) -> np.ndarray:
        """Weight of the Timestral forecast for each quarter hour after the last known price"""
        fade = np.linspace(1.0, 0.0, cls.HORIZON - cls.FULL_WEIGHT_SLOTS + 1)[1:]
        return cls.WEIGHT * np.concatenate([np.ones(cls.FULL_WEIGHT_SLOTS), fade])

    @staticmethod
    def covariates(predictor: PricePredictor, df: pd.DataFrame) -> np.ndarray:
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
        return np.vstack([row.to_numpy(dtype=float) for row in rows])

    def _forecast(self, prices: np.ndarray, covariates: np.ndarray, weekday: np.ndarray) -> np.ndarray:
        import torch
        from timestral import TimestralForecaster

        if self.forecaster is None:
            log.info(f"loading {self.model_name}")
            self.forecaster = TimestralForecaster.from_pretrained(self.model_name)
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
        covariates = self.covariates(predictor, df)
        known = ~np.isnan(covariates).any(axis=0)
        # needs all of the future it is asked for, and most of the history (it copes with gaps in there)
        horizon = int(np.argmin(known[len(history):])) if not known[len(history):].all() else len(future)
        if horizon < 96 or len(history) < self.CONTEXT // 2 or history["price"].isna().mean() > 0.1:
            log.warning(f"{predictor.region.bidding_zone_entsoe}: not enough data for a Timestral forecast")
            return None

        rows = len(history) + horizon
        covariates = np.nan_to_num(covariates[:, :rows])
        values = await asyncio.to_thread(
            self._forecast,
            history["price"].to_numpy(dtype=float),
            covariates,
            df["weekday"].to_numpy(dtype=float)[:rows],
        )
        return pd.Series(values, index=future.index[:horizon], name="price")

    async def blend(self, predictor: PricePredictor, predictions: list[pd.DataFrame]) -> bool:
        """
        Mix the Timestral forecast into the price forecasts (in place). Returns whether it did.
        A failure only costs the second opinion, never the forecast itself.
        """
        try:
            forecast = await self.forecast(predictor)
        except Exception:
            log.exception(f"{predictor.region.bidding_zone_entsoe}: Timestral forecast failed - continuing without it")
            return False
        if forecast is None:
            return False

        weight = pd.Series(self.weights()[:len(forecast)], index=forecast.index)
        for prediction in predictions:
            common = forecast.index.intersection(prediction.index[prediction["price"].notna()])
            prediction.loc[common, "price"] = (1 - weight[common]) * prediction.loc[common, "price"] + weight[common] * forecast[common]
        log.info(f"{predictor.region.bidding_zone_entsoe}: blended Timestral forecast for {len(forecast)} quarter hours")
        return True
