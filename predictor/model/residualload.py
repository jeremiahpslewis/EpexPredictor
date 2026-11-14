"""
Forecast of the residual load (load minus wind and solar power) as input of the price model.

The day-ahead auction clears on what the market expects the residual load to be, and the transmission system
operators (TSOs) publish that expectation as day-ahead forecasts of wind power, solar power and load. But they
publish it one day ahead only, which is too late for a price forecast. So this module learns those figures from
the weather forecast on a dense grid of locations, and delivers the published figure where there is one and its
own forecast everywhere else.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import override

import aiohttp
import lightgbm as lgb
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout

from .auxdatastore import AuxDataStore
from .datastore import DataStore
from .priceregion import PriceRegion
from .pricestore import PriceStore
from .weatherstore import WeatherStore

log = logging.getLogger(__name__)

# Wind speed [km/h] from which on a wind turbine starts to deliver power, and from which on it delivers its rated power
CUT_IN_WIND_SPEED = 10.8
RATED_WIND_SPEED = 43.2
# Locations north of this latitude make up the "north" of a region, where most of the wind power of Germany is located
NORTH_LATITUDE = 52.0
# Less training data than this is not enough to learn the seasons from
MIN_TRAINING_ROWS = 60 * 96


class GridWeatherStore(WeatherStore):
    """
    Hourly weather on the region's dense generation grid, from OpenMeteo
    """

    STORAGE_PREFIX = "gridweather_v1"
    RESOLUTION = "hourly"
    VARIABLES = {
        "wind_speed_120m": "wind",
        "shortwave_radiation": "radiation",
        "temperature_2m": "temp",
    }
    # A request for this many locations weighs heavily on OpenMeteo's rate limit, so fetch in smaller pieces and slowly
    MAX_RANGE_DAYS = 30
    REQUEST_PAUSE_SECONDS = 15.0

    @override
    def latitudes(self) -> list[float]:
        return self.region.generation_latitudes + self.region.generation_offshore_latitudes

    @override
    def longitudes(self) -> list[float]:
        return self.region.generation_longitudes + self.region.generation_offshore_longitudes


class GenerationForecastStore(DataStore):
    """
    Day-ahead forecasts of wind power, solar power and load [GW] as published by the TSOs, from energy-charts.info
    """

    # production type at energy-charts -> column it is added to
    PRODUCTION_TYPES = {
        "wind_onshore": "tso_wind",
        "wind_offshore": "tso_wind",
        "solar": "tso_solar",
        "load": "tso_load",
    }
    MAX_RANGE_DAYS = 180

    update_lock: asyncio.Lock

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        super().__init__(region, storage_dir, "generation_v1")
        self.update_lock = asyncio.Lock()

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        # The forecasts for the next day appear some time during the day
        return datetime.now(timezone.utc) + timedelta(hours=1)

    @override
    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        if self.region.generation_forecast_country is None:
            return False

        async with self.update_lock:
            updated = False
            for rstart, rend in self.gen_missing_date_ranges(start, end):
                while rstart <= rend:
                    chunk_end = min(rend, rstart + timedelta(days=self.MAX_RANGE_DAYS))
                    df = await self.fetch_range(rstart, chunk_end)
                    if df is not None and len(df) > 0:
                        updated = self._update_data(df) or updated
                    rstart = chunk_end + timedelta(minutes=15)

            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: generation forecast data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()
            return updated

    async def fetch_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        columns: dict[str, pd.Series] = {}
        for production_type, column in self.PRODUCTION_TYPES.items():
            values = await self.fetch_forecast(production_type, rstart, rend)
            if values is None:
                return None
            columns[column] = columns[column].add(values, fill_value=0) if column in columns else values
        df = pd.DataFrame(columns).dropna() / 1000
        df.index.name = "time"
        return df

    @PriceStore._limit_energycharts
    async def fetch_forecast(self, production_type: str, rstart: datetime, rend: datetime) -> pd.Series | None:
        try:
            start_formatted = rstart.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            end_formatted = rend.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            url = (f"https://api.energy-charts.info/public_power_forecast?country={self.region.generation_forecast_country}"
                   f"&production_type={production_type}&forecast_type=day-ahead&start={start_formatted}&end={end_formatted}")
            log.info(f"{self.region.bidding_zone_entsoe}: fetching generation forecast: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"accept": "application/json"}, timeout=ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    data = json.loads(await resp.text())
                    values = pd.Series(
                        data["forecast_values"],
                        index=pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
                        dtype=float,
                    ).dropna()
                    return values[~values.index.duplicated()]
        except Exception as e:
            log.error(f"{self.region.bidding_zone_entsoe}: failed to fetch generation forecast from energy-charts: {e}")
            return None


def generation_features(region: PriceRegion, weather: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """
    Input features of the generation models, on the (15 minute) index of aux.

    weather: hourly grid weather as delivered by GridWeatherStore
    aux: time features as delivered by AuxDataStore
    """
    index = aux.index
    weather = weather.reindex(weather.index.union(index)).interpolate(method="time", limit=3, limit_area="inside").reindex(index)

    n = len(region.generation_latitudes) + len(region.generation_offshore_latitudes)
    onshore = np.arange(n) < len(region.generation_latitudes)
    north = onshore & (np.array(region.generation_latitudes + region.generation_offshore_latitudes) >= NORTH_LATITUDE)
    south = onshore & ~north

    def values(name: str) -> np.ndarray:
        return weather.reindex(columns=[f"{name}_{i}" for i in range(n)]).to_numpy(dtype=float)

    def mean(a: np.ndarray, columns: np.ndarray) -> np.ndarray:
        return a[:, columns].mean(axis=1) if columns.any() else np.zeros(len(a))

    wind = values("wind")
    radiation = values("radiation")
    temp = values("temp")
    # roughly the share of its rated power a turbine delivers at this wind speed
    power = np.clip(wind / RATED_WIND_SPEED, 0, 1) ** 3
    power_linear = np.clip((wind - CUT_IN_WIND_SPEED) / (RATED_WIND_SPEED - CUT_IN_WIND_SPEED), 0, 1)

    features: dict[str, np.ndarray] = {f"wind_power_{i}": power[:, i] for i in range(n)}
    features["wind_power_onshore"] = mean(power, onshore)
    features["wind_power_north"] = mean(power, north)
    features["wind_power_south"] = mean(power, south)
    features["wind_power_offshore"] = mean(power, ~onshore)
    features["wind_power_linear"] = mean(power_linear, onshore)
    features["wind_speed_max"] = wind[:, onshore].max(axis=1)
    features.update({f"radiation_{i}": radiation[:, i] for i in np.flatnonzero(onshore)})
    features["radiation_all"] = mean(radiation, onshore)
    features["radiation_south"] = mean(radiation, south)
    features["temp_all"] = mean(temp, onshore)
    features["temp_south"] = mean(temp, south)
    for name in ("ts", "slot", "weekday", "holiday", "sunelevation", "azimuth"):
        features[name] = aux[name].to_numpy(dtype=float)
    return pd.DataFrame(features, index=index)


class ResidualLoad:
    """
    Delivers the residual load [GW] as input feature of the price model: the figure published by the TSOs where
    there is one, and a forecast from the grid weather everywhere else.
    """

    FEATURE = "residual_load"
    TARGETS = ("tso_wind", "tso_solar", "tso_load")
    LGB_PARAMS = {"force_col_wise": True, "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31}
    LGB_ROUNDS = 400

    region: PriceRegion
    gridweather: GridWeatherStore
    generation: GenerationForecastStore
    auxstore: AuxDataStore
    models: dict[str, lgb.Booster] | None = None

    def __init__(self, region: PriceRegion, storage_dir: str | None, auxstore: AuxDataStore):
        self.region = region
        self.gridweather = GridWeatherStore(region, storage_dir)
        self.generation = GenerationForecastStore(region, storage_dir)
        self.auxstore = auxstore

    @staticmethod
    def is_configured(region: PriceRegion) -> bool:
        return region.generation_forecast_country is not None and len(region.generation_latitudes) > 0

    def with_shared_stores(self) -> "ResidualLoad":
        """
        A forecaster that reads the same data, but trains its own models. So a model trained on a window of the past
        (eval plot) does not replace the live one.
        """
        other = ResidualLoad.__new__(ResidualLoad)
        other.region = self.region
        other.gridweather = self.gridweather
        other.generation = self.generation
        other.auxstore = self.auxstore
        return other

    async def load(self):
        await asyncio.gather(self.gridweather.load(), self.generation.load())

    def last_data_update(self) -> datetime:
        return max(self.gridweather.last_updated, self.generation.last_updated)

    async def refresh_forecasts(self, start: datetime, end: datetime):
        await self.gridweather.refresh_range(start, end)

    def drop_before(self, cutoff: datetime):
        self.gridweather.drop_before(cutoff)
        self.generation.drop_before(cutoff)

    async def get_grid_weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            return await self.gridweather.get_data(start, end)
        except Exception as e:
            # The price forecast must not fail with this input: carry on with what is there. Times without weather
            # get no residual load, and too little of it leaves the price model without this input altogether.
            log.warning(f"{self.region.bidding_zone_entsoe}: failed to fetch grid weather, using what is stored: {e}")
            return self.gridweather.data.loc[start.astimezone(timezone.utc):end.astimezone(timezone.utc)]

    async def get_features(self, start: datetime, end: datetime) -> pd.DataFrame:
        weather, aux = await asyncio.gather(
            self.get_grid_weather(start, end),
            self.auxstore.get_data(start, end),
        )
        return generation_features(self.region, weather, aux)

    async def train(self, start: datetime, end: datetime):
        self.models = None
        features = await self.get_features(start, end)
        published = await self.generation.get_data(start, end)
        published = published.reindex(index=features.index, columns=list(self.TARGETS))

        usable = features.notna().all(axis=1) & published.notna().all(axis=1)
        if usable.sum() < MIN_TRAINING_ROWS:
            log.warning(f"{self.region.bidding_zone_entsoe}: not enough data for the residual load forecast "
                        f"({int(usable.sum())} quarter hours) - the price model is trained without it")
            return

        models = {}
        for target in self.TARGETS:
            dataset = lgb.Dataset(features[usable], label=published.loc[usable, target])
            models[target] = await asyncio.to_thread(lgb.train, params=self.LGB_PARAMS, train_set=dataset, num_boost_round=self.LGB_ROUNDS)
        self.models = models

    async def get_data(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        """
        The residual load for the given range, or None if the forecaster is not trained. Rows without published
        figure and without weather data are NaN.
        """
        if self.models is None:
            return None

        features = await self.get_features(start, end)
        published = await self.generation.get_data(start, end)
        figures = published.reindex(index=features.index, columns=list(self.TARGETS))

        complete = features.notna().all(axis=1)
        missing = figures.isna().any(axis=1) & complete
        if missing.any():
            for target, model in self.models.items():
                figures.loc[missing, target] = np.asarray(model.predict(features[missing]))

        return pd.DataFrame({self.FEATURE: figures["tso_load"] - figures["tso_wind"] - figures["tso_solar"]})
