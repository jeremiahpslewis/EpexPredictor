from datetime import datetime

import pandas as pd

from .weatherstore import WeatherStore

# Zones whose weather is averaged...
EUROPE_WEATHER_ZONES = ("DE_LU", "AT", "BE", "FR", "NL", "SE_4", "DK_1", "DK_2")
# ...and the ones around the North Sea, where most of the wind power of the coupled markets is located
NORTH_SEA_WEATHER_ZONES = ("DE_LU", "NL", "BE", "DK_1", "DK_2")
# Wind speed [km/h] from which on a wind turbine roughly delivers its rated power
RATED_WIND_SPEED = 45.0


class EuropeWeather:
    """
    Large-scale weather context, shared by the models of all regions: wind power potential, irradiance and
    temperature averaged over several zones, plus the wind power potential around the North Sea.
    Prices depend on the weather of the whole coupled market, not only on a region's own.

    Computed from the weather stores of the regions' models, so no additional data has to be fetched.
    """

    weatherstores: list[WeatherStore]

    # Every model of a training cycle asks for the same range, so keep the last few results
    CACHE_SIZE = 8
    cache: dict[tuple, pd.DataFrame | None]

    def __init__(self, weatherstores: list[WeatherStore]):
        self.weatherstores = [w for w in weatherstores if w.region.bidding_zone_entsoe in EUROPE_WEATHER_ZONES]
        self.cache = {}

    async def get_data(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        zones = [(store, await store.get_data(start, end)) for store in self.weatherstores]

        key = (start, end, tuple((store.last_updated, len(weather)) for store, weather in zones))
        if key not in self.cache:
            if len(self.cache) >= self.CACHE_SIZE:
                self.cache.clear()
            self.cache[key] = self._compute(zones)
        return self.cache[key]

    def _compute(self, zones: list[tuple[WeatherStore, pd.DataFrame]]) -> pd.DataFrame | None:
        wind = []
        wind_northsea = []
        irradiance = []
        temperature = []
        for store, weather in zones:
            if weather.empty:
                continue

            # roughly the share of rated power a turbine delivers, averaged over the zone's sample locations
            wind_speed = weather.filter(regex=r"^wind_\d+$").clip(lower=0, upper=RATED_WIND_SPEED)
            zone_wind = ((wind_speed / RATED_WIND_SPEED) ** 3).mean(axis=1)
            wind.append(zone_wind)
            if store.region.bidding_zone_entsoe in NORTH_SEA_WEATHER_ZONES:
                wind_northsea.append(zone_wind)
            irradiance.append(weather.filter(regex=r"^irradiance_\d+$").mean(axis=1))
            temperature.append(weather.filter(regex=r"^temp_\d+$").mean(axis=1))

        if len(wind) == 0:
            return None

        features = {
            "eu_wind": wind,
            "eu_wind_northsea": wind_northsea,
            "eu_irradiance": irradiance,
            "eu_temperature": temperature,
        }
        return pd.DataFrame({
            name: pd.concat(zones, axis=1, sort=True).mean(axis=1)
            for name, zones in features.items() if len(zones) > 0
        })
