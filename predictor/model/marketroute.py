"""
Optional third opinion for Germany: the market model (marketmodel.py) with its supply shift forecast by Timestral.

For every quarter hour with a known price, the supply shift is worked out: how much more or less supply there was
than the model's curves say, in GW. Timestral continues that series into the future, with the residual load as
known-future covariate, and the market is cleared again with the forecast shift and the forecast residual load.

The price forecast it gives is about as good as the one of Timestral on the price itself (timestralblend.py), and
they do not make all the same mistakes: in backtests of the next day from midnight, the average of the two was
better than Timestral on the price alone, and adding this one to the tree models and Timestral lowered their
error over the whole day by 7 to 13 %. That gain needs a good residual load forecast - with the residual load
forecast from weather forecasts two to three days old, there was none.

Only what the application fetches anyway is used: prices, the residual load and the gas price. The carbon price and
the output of the other plant are set to constants, and the neighbours are left out. What that leaves out ends up
in the supply shift, which made no difference in the backtests of this use.

Runs with the Timestral model that is switched on with EPEXPREDICTOR_TIMESTRAL_MODEL, unless
EPEXPREDICTOR_MARKET_MODEL is set to false.
"""

import asyncio
import logging
import os

import numpy as np
import pandas as pd

from . import marketmodel
from .pricepredictor import PricePredictor
from .timestralblend import SLOT, TimestralBlend

log = logging.getLogger(__name__)

# Carbon price [EUR/t] and output of the plant that is no part of the model (biomass, hydro, waste, ...) [GW]
CARBON_PRICE = 75.0
OTHER_PLANT = 8.0

# The gas price is used as it was two days before [quarter hours]: so it is known for all of the forecast, and the
# supply shift of the past is worked out the same way as the forecast is cleared
GAS_PRICE_LAG = 192

# Prices of the application are in ct/kWh, the market model works in EUR/MWh
EUR_PER_MWH = 10.0


class MarketRoute:
    CONTEXT = TimestralBlend.CONTEXT
    HORIZON = TimestralBlend.HORIZON
    RESIDUAL_LOAD = TimestralBlend.RESIDUAL_LOAD
    GAS_PRICE = "gasprice"

    timestral: TimestralBlend

    def __init__(self, timestral: TimestralBlend):
        self.timestral = timestral

    @staticmethod
    def from_env(timestral: TimestralBlend | None) -> "MarketRoute | None":
        if timestral is None:
            return None
        if os.getenv("EPEXPREDICTOR_MARKET_MODEL", "true").strip().lower() in ("no", "false", "f", "0"):
            return None
        return MarketRoute(timestral)

    @staticmethod
    def gas_price(df: pd.DataFrame) -> np.ndarray:
        """The gas price of two days before for every row of df; the first two days get the earliest one there is"""
        gas = df[MarketRoute.GAS_PRICE]
        lagged = gas.copy()
        lagged.index = lagged.index + GAS_PRICE_LAG * SLOT
        return lagged.reindex(df.index).fillna(gas.dropna().iloc[0] if gas.notna().any() else np.nan).to_numpy(dtype=float)

    @staticmethod
    def supply_shift(df: pd.DataFrame) -> np.ndarray:
        """Supply shift [GW] for every row with a known price, residual load and gas price - NaN elsewhere"""
        return marketmodel.supply_shift(
            df[MarketRoute.RESIDUAL_LOAD].to_numpy(dtype=float) - OTHER_PLANT,
            df["price"].to_numpy(dtype=float) * EUR_PER_MWH,
            MarketRoute.gas_price(df),
            CARBON_PRICE,
        )

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
        if df is None or self.RESIDUAL_LOAD not in df.columns or self.GAS_PRICE not in df.columns:
            return None
        history, future = df.loc[:last_known], df.loc[first:]

        demand = df[self.RESIDUAL_LOAD].to_numpy(dtype=float) - OTHER_PLANT
        gas = self.gas_price(df)
        inputs = np.vstack([demand, gas])
        horizon = TimestralBlend.known_horizon(inputs, len(history))
        shift = self.supply_shift(df)[:len(history)]
        if horizon < 96 or len(history) < self.CONTEXT // 2 or np.isnan(shift).mean() > 0.1:
            log.warning(f"{predictor.region.bidding_zone_entsoe}: not enough data for a market model forecast "
                        f"({len(history)} quarter hours of history, {np.isnan(shift).mean():.0%} of them without supply shift, "
                        f"residual load and gas price known for {horizon} quarter hours ahead)")
            return None

        rows = len(history) + horizon
        # gaps in the history stay NaN: Timestral masks them
        values = await asyncio.to_thread(
            self.timestral._forecast,
            shift,
            demand[None, :rows],
            df["weekday"].to_numpy(dtype=float)[:rows],
        )
        values = np.asarray(values, dtype=float)
        if values.shape != (horizon,) or not np.isfinite(values).all():
            log.warning(f"{predictor.region.bidding_zone_entsoe}: Timestral returned an unusable forecast of the supply shift "
                        f"(shape {values.shape}, {int(np.size(values) - np.isfinite(values).sum())} non-finite values) - not used")
            return None

        prices = marketmodel.clearing_price(demand[len(history):rows], gas[len(history):rows], CARBON_PRICE, shift=values) / EUR_PER_MWH
        return pd.Series(prices, index=future.index[:horizon], name="price")
