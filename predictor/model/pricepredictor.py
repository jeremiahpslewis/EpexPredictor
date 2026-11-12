#!/usr/bin/python3

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, cast

import numpy as np
import pandas as pd
import lightgbm as lgb

from .auxdatastore import AuxDataStore
from .europeweather import EuropeWeather
from .priceregion import PriceRegion
from .pricestore import PriceStore
from .weatherstore import WeatherStore
from .entsoedatastore import EntsoeDataStore
from .gaspricestore import GasPriceStore

log = logging.getLogger(__name__)

# Lower bound for the gas price [EUR/MWh] when it is used as divisor
MIN_GAS_PRICE = 1.0


class PricePredictor:
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    entsoestore: EntsoeDataStore
    auxstore: AuxDataStore
    gasstore: GasPriceStore

    traindata: pd.DataFrame | None = None

    predictor: lgb.Booster | None = None
    # second model for regions with a gas price, trained on price / gas price (see train())
    gas_predictor: lgb.Booster | None = None

    # 2-stage forecasting: stage-1 predictors whose price forecasts are added as extra features
    cross_predictors: list["PricePredictor"]

    # large-scale weather features, shared by the models of all regions. Set by whoever creates the models
    europe_weather: EuropeWeather | None = None

    # Inputs whose source does not reach as far as the others, so they may be missing for part of a forecast
    optional_inputs: list[str]

    def __init__(self, region: PriceRegion, storage_dir: str | None = None, cross_predictors: list["PricePredictor"] | None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region)
        self.entsoestore = EntsoeDataStore(region, storage_dir)
        self.gasstore = GasPriceStore(region, storage_dir)

        # When set, the price forecasts of these (stage-1) predictors are added as extra
        # input features. Each cross predictor is responsible for delivering its own
        # forecast (and for any horizon handling of its data).
        self.cross_predictors = cross_predictors or []
        self.optional_inputs = []

    async def load_from_persistence(self):
        await asyncio.gather(
            self.weatherstore.load(),
            self.pricestore.load(),
            self.entsoestore.load(),
            self.gasstore.load()
        )
        return self
    
    def last_data_update(self) -> datetime:
        return max(self.weatherstore.last_updated, self.pricestore.last_updated, self.entsoestore.last_updated, self.gasstore.last_updated)

    def use_datastores_from(self, other: "PricePredictor"):
        assert self.region.bidding_zone_entsoe == other.region.bidding_zone_entsoe
        self.weatherstore = other.weatherstore
        self.pricestore = other.pricestore
        self.auxstore = other.auxstore
        self.entsoestore = other.entsoestore
        self.gasstore = other.gasstore
        self.europe_weather = other.europe_weather

    def is_trained(self) -> bool:
        return self.predictor is not None


    async def train(self, start: datetime, end: datetime):
        self.traindata = await self.prepare_dataframe(start, end)
        if self.traindata is None:
            return
        self.traindata.dropna(inplace=True)

        params = self.traindata.drop(columns=["price"])
        output = self.traindata["price"]

        lgb_dataset = lgb.Dataset(params, label=output)
        params_lgb = {
            "force_col_wise": True,
            "verbosity": -1,
        }

        self.predictor = await asyncio.to_thread(lgb.train, params=params_lgb, train_set=lgb_dataset)

        # Whenever gas power plants set the price, it follows the gas price almost proportionally. But a tree model
        # cannot predict a price level it has not seen in its training data, so it lags behind when gas prices
        # move. A second model therefore predicts price / gas price, which also keeps older training data comparable.
        # Neither view is right all the time (and for every region), so predict() averages both models.
        self.gas_predictor = None
        gas = self.get_gas_scale(params)
        if gas is not None:
            lgb_dataset = lgb.Dataset(
                self.to_gas_relative(params, gas),
                label=output / gas,
                weight=gas.to_numpy() ** 2, # keeps the objective a squared error in price units
            )
            self.gas_predictor = await asyncio.to_thread(lgb.train, params=params_lgb, train_set=lgb_dataset)



    async def predict(self, start: datetime, end: datetime, fill_known=True) -> pd.DataFrame:
        assert self.is_trained() and self.predictor is not None

        df = await self.prepare_dataframe(start, end)
        assert df is not None

        prices_known = df["price"]

        params = df.drop(columns=["price"])

        prediction = np.asarray(self.predictor.predict(params))
        gas = self.get_gas_scale(params)
        if self.gas_predictor is not None and gas is not None:
            gas_relative = np.asarray(self.gas_predictor.predict(self.to_gas_relative(params, gas)))
            prediction = (prediction + gas_relative * gas.to_numpy()) / 2

        # The models read a missing input as zero: no wind and no sun, or a price of zero in another region. A
        # forecast made from that looks plausible and is wrong, so these rows get none.
        required = [c for c in params.columns if c not in self.optional_inputs]
        prediction[params[required].isna().any(axis=1).to_numpy()] = np.nan

        resultdf = pd.DataFrame(index=params.index)
        resultdf["price"] = prediction

        if fill_known:
            resultdf.update(prices_known)

        return resultdf

    def to_price_dict(self, df : pd.DataFrame) -> Dict[datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result



    def get_gas_scale(self, params: pd.DataFrame) -> pd.Series | None:
        """
        The gas price each row is divided by for the gas-relative model, or None without gas price data
        """
        if "gasprice" not in params.columns:
            return None
        return params["gasprice"].clip(lower=MIN_GAS_PRICE)


    def to_gas_relative(self, params: pd.DataFrame, gas: pd.Series) -> pd.DataFrame:
        """
        Input features of the gas-relative model: cross-region prices are converted to the unit of its target
        """
        cross_prices = [c for c in params.columns if c.startswith("cross_price_")]
        if len(cross_prices) == 0:
            return params
        params = params.copy()
        params[cross_prices] = params[cross_prices].div(gas, axis=0)
        return params


    async def get_cross_features(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        """
        Build the extra input features from the stage-1 (cross) predictors.

        For each cross predictor we produce a single column named after its region:
        the predicted price for each timestamp. Each cross predictor is responsible for
        delivering its own forecast; any horizon handling of its data is its concern.
        """
        if len(self.cross_predictors) == 0:
            return None

        frames = []
        for cp in self.cross_predictors:
            col = f"cross_price_{cp.region.bidding_zone_entsoe}"
            pred = await cp.predict(start, end, fill_known=False)
            pred = pred.rename(columns={"price": col})
            frames.append(pred)

        return pd.concat(frames, axis=1, sort=True)

    async def prepare_dataframe(self, actual_start: datetime, end: datetime) -> pd.DataFrame | None:
        # gas prices are not available for the days ahead. If forecast range is in the future, we might have nothing to ffill. Ensure we do
        start = min(datetime.now(timezone.utc) - timedelta(days=14), actual_start)

        weather, prices, auxdata = await asyncio.gather(
            self.weatherstore.get_data(start, end),
            self.pricestore.get_data(start, end),
            self.auxstore.get_data(start, end)
        )

        df = pd.concat([weather, auxdata], axis=1, sort=True)

        
        if self.region.use_entsoe_load_forecast:
            entsoedata = await self.entsoestore.get_data(start, end)
            if len(entsoedata) > 0:
                df = pd.concat([df, entsoedata], axis=1, sort=True)
                self.optional_inputs = list(entsoedata.columns)

        if self.region.use_de_nat_gas_price:
            gasprices = await self.gasstore.get_data(start, end)
            gasprices = gasprices.reindex(df.index).ffill()
            df = pd.concat([df, gasprices], axis=1, sort=True)

        if self.europe_weather is not None:
            europe_weather = await self.europe_weather.get_data(start, end)
            if europe_weather is not None:
                df = pd.concat([df, europe_weather], axis=1, sort=True)

        if len(self.cross_predictors) > 0:
            cross_features = await self.get_cross_features(start, end)
            if cross_features is not None:
                df = pd.concat([df, cross_features], axis=1, sort=True)

        df = pd.concat([df, prices], axis=1, sort=True)
        df = df[actual_start:]
        return df

    async def refresh_forecasts(self, start : datetime, end: datetime):
        """
            Will re-fetch everything starting from yesterday during next training
            Not sure when past data becomes "stable", so better be sure and fetch a bit more
        """
        await self.weatherstore.refresh_range(start, end)
        await self.entsoestore.refresh_range(start, end)


    def cleanup(self):
        """
        Delete data older than 5 years
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=1825)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)
        self.entsoestore.drop_before(cutoff)
        self.gasstore.drop_before(cutoff)




