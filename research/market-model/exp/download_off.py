"""Weather for extra offshore sample locations of a zone, fetched through the repo's WeatherStore.
usage: download_off.py DATA_DIR START END ZONE LAT,LAT,... LON,LON,..."""
import asyncio
import dataclasses
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/user/EpexPredictor")
import aiohttp  # noqa: E402

_Orig = aiohttp.ClientSession


class _ProxySession(_Orig):  # the sandbox only reaches the internet through HTTPS_PROXY
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _ProxySession

from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING)
DATA_DIR = sys.argv[1]
START = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
ZONE = PriceRegionName(sys.argv[4])
LATS = [float(x) for x in sys.argv[5].split(",")]
LONS = [float(x) for x in sys.argv[6].split(",")]


async def main():
    base = ZONE.to_region()
    region = dataclasses.replace(base, latitudes=LATS, longitudes=LONS, extra_weather_latitudes=[], extra_weather_longitudes=[],
                                 bidding_zone_entsoe=f"{base.bidding_zone_entsoe}_OFF")
    region.holidays = base.holidays
    ws = await WeatherStore(region, DATA_DIR).load()
    ws.known_source_horizon = None
    df = None
    for attempt in range(20):
        try:
            df = await ws.get_data(START, END)
            break
        except Exception as e:
            print(ZONE.value, "retry", attempt, repr(e)[:140], flush=True)
            await asyncio.sleep(30 + 20 * attempt)
    print(ZONE.value, "offshore weather:", None if df is None else (len(df), str(df.index.min()), str(df.index.max()), len(df.columns)), flush=True)


asyncio.run(main())
