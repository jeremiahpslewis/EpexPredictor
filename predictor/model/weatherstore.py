import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Self, override

import aiohttp
import pandas as pd
from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class WeatherStore(DataStore):
    """
    Fetches and caches weather data from OpenMeteo
    TODO: add management of allowed API calls, delay queries if needed
    """

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str|None

    update_lock: asyncio.Lock
    fetch_lock: asyncio.Lock

    # Shared across all WeatherStore instances: OpenMeteo dislikes concurrent
    # requests, so we serialize every call to its API globally.
    _openmeteo_lock: asyncio.Lock = asyncio.Lock()

    # What is fetched, and under which column prefix it is stored. Subclasses fetch something else.
    STORAGE_PREFIX = "weather_v2"
    RESOLUTION = "minutely_15"
    VARIABLES = {
        "wind_speed_80m": "wind",
        "temperature_2m": "temp",
        "global_tilted_irradiance": "irradiance",
        "pressure_msl": "pressure",
        "relative_humidity_2m": "humidity",
    }
    # longest range fetched with one request (OpenMeteo allows 90 days)
    MAX_RANGE_DAYS = 90
    # pause after a request, for stores whose requests weigh heavily on OpenMeteo's rate limit
    REQUEST_PAUSE_SECONDS = 0.0

    def __init__(self, region : PriceRegion, storage_dir: str|None =None):
        super().__init__(region, storage_dir, self.STORAGE_PREFIX)
        self.update_lock = asyncio.Lock()
        self.fetch_lock = asyncio.Lock()

    def latitudes(self) -> list[float]:
        return self.region.weather_latitudes

    def longitudes(self) -> list[float]:
        return self.region.weather_longitudes

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return self.last_updated + timedelta(hours=6)

    def expected_columns(self) -> set[str]:
        return {
            f"{name}_{i}"
            for i in range(len(self.latitudes()))
            for name in self.VARIABLES.values()
        }

    @override
    async def load(self) -> Self:
        await super().load()
        if not self.data.empty and set(self.data.columns) != self.expected_columns():
            # The region's sample locations changed since this cache was written. Mixing both layouts would
            # leave rows without values for the new locations, so start over and re-fetch.
            log.warning(f"{self.region.bidding_zone_entsoe}: cached weather data does not match the configured locations, discarding it")
            self.data = pd.DataFrame()
            self.known_source_horizon = None
            self.source_horizon_revalitation_ts = None
        return self

    async def refresh_range(self, rstart: datetime, rend: datetime) -> bool:
        async with self.update_lock:
            lats = ",".join(map(str, self.latitudes()))
            lons = ",".join(map(str, self.longitudes()))

            updated = False
            if self.needs_history_query(rstart):
                host = "historical-forecast-api.open-meteo.com"
            else:
                host = "api.open-meteo.com"

            url = f"https://{host}/v1/forecast?latitude={lats}&longitude={lons}&azimuth=0&tilt=0&start_date={rstart.date().isoformat()}&end_date={rend.date().isoformat()}&{self.RESOLUTION}={",".join(self.VARIABLES)}&timezone=UTC"
            log.info(f"{self.region.bidding_zone_entsoe}: Fetching weather data: {url}")

            try:
                # Serialize all OpenMeteo calls globally; it dislikes concurrency.
                async with WeatherStore._openmeteo_lock:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as resp:
                            data = await resp.text()

                            data = json.loads(data)
                            if isinstance(data, dict):
                                # one location is returned as an object instead of a list, and so is an error
                                if "error" in data:
                                    raise RuntimeError(data.get("reason", "OpenMeteo reported an error"))
                                data = [data]
                            frames = []
                            for i, fc in enumerate(data):
                                df = pd.DataFrame()

                                df["time"] = fc[self.RESOLUTION]["time"]
                                for variable, name in self.VARIABLES.items():
                                    df[f"{name}_{i}"] = fc[self.RESOLUTION][variable]

                                df.set_index("time", inplace=True)
                                df = df.dropna()
                                frames.append(df)

                            df = pd.concat(frames, axis=1).reset_index()
                            df["time"] = pd.to_datetime(df["time"], utc=True)
                            df.set_index("time", inplace=True)

                    if self.REQUEST_PAUSE_SECONDS > 0:
                        await asyncio.sleep(self.REQUEST_PAUSE_SECONDS)

                updated = self._update_data(df) or updated
            except Exception as e:
                log.warning(f"{self.region.bidding_zone_entsoe}: Failed to fetch weather data: error: {str(e)}")
                raise e
            finally:
                if updated:
                    log.info(f"{self.region.bidding_zone_entsoe}: weather data updated")
                    self.data.sort_index(inplace=True)
                    await self.serialize()

            return updated

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        updated = False

        # The models of several regions read this store (see EuropeWeather). Let only one of them at a time look
        # for missing data, or they would all fetch the same range.
        async with self.fetch_lock:
            for rstart, rend in self.gen_missing_date_ranges(start, end):
                updated = await self.refresh_range(rstart, rend) or updated

        return updated

    @override
    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        # a random hour of the day so we can easily check if we already have that day.
        # OpenMeteo only has full day queries anyway.
        start = start.replace(hour=12, minute=0, second=0, microsecond=0)
        end = end.replace(hour=12, minute=0, second=0, microsecond=0)

        curr = start
        result = []

        rangestart = None
        while curr <= end:
            next_day = curr + timedelta(days=1)

            apiswitch = rangestart is not None and self.needs_history_query(rangestart) != self.needs_history_query(next_day)


            if rangestart is not None and (next_day in self.data.index or next_day > end or apiswitch or (curr - rangestart).total_seconds() > 60 * 60 * 24 * self.MAX_RANGE_DAYS):
                # We have the next timeslot already OR its the last timeslot OR the current range exceeds the maximum (90 days for openmeteo) OR we need to change APIs
                result.append((pd.to_datetime(rangestart), pd.to_datetime(curr)))
                rangestart = None

            if rangestart is None and curr not in self.data.index:
                rangestart = curr

            curr = next_day
        return result

    def needs_history_query(self, dt: datetime) -> bool:
        """
        If query is older than a few weeks, we need OpenMeteo's historical data API.
        We switch to historical API at 1 day to be safe.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=1, minutes=1)
        return dt < cutoff
