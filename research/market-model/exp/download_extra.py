"""Weather for extra DE sample locations, fetched through the repo's WeatherStore."""
import asyncio
import dataclasses
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/user/EpexPredictor")
import aiohttp  # noqa: E402

_Orig = aiohttp.ClientSession


class _ProxySession(_Orig):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _ProxySession

from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
DATA_DIR = sys.argv[1]
START = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
# North Sea offshore, Schleswig-Holstein west coast, Emsland, Saxony-Anhalt, East Brandenburg, Baltic offshore
LATS = [54.4, 54.5, 52.9, 52.0, 52.4, 54.9]
LONS = [6.4, 8.9, 7.6, 11.6, 14.0, 13.8]


async def main():
    region = dataclasses.replace(PriceRegionName.DE.to_region(), latitudes=LATS, longitudes=LONS, bidding_zone_entsoe="DE_X6")
    region.holidays = PriceRegionName.DE.to_region().holidays
    ws = await WeatherStore(region, DATA_DIR).load()
    ws.known_source_horizon = None
    for attempt in range(15):
        try:
            df = await ws.get_data(START, END)
            break
        except Exception as e:
            print("retry", attempt, repr(e)[:160], flush=True)
            await asyncio.sleep(20 + 15 * attempt)
    print("DE_X6 weather", len(df), df.index.min(), df.index.max(), list(df.columns)[:5], flush=True)


asyncio.run(main())
