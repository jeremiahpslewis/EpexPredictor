"""Add the downloaded offshore locations to a zone's weather cache, in the format the repo's WeatherStore writes.
usage: merge_off.py BASE_DATA_DIR RAW_DIR OUT_DATA_DIR ZONE NUMBER_OF_EXTRA_LOCATIONS"""
import asyncio
import dataclasses
import sys

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")

from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402

BASE, RAW, OUT, ZONE, N = sys.argv[1], sys.argv[2], sys.argv[3], PriceRegionName(sys.argv[4]), int(sys.argv[5])


async def main():
    region = ZONE.to_region()
    nbase = len(region.latitudes)
    base = await WeatherStore(region, BASE).load()
    off_region = dataclasses.replace(region, latitudes=[0.0] * N, longitudes=[0.0] * N, bidding_zone_entsoe=f"{region.bidding_zone_entsoe}_OFF")
    off_region.holidays = region.holidays
    off = await WeatherStore(off_region, RAW).load()
    renamed = off.data.rename(columns={c: f"{c.rsplit('_', 1)[0]}_{nbase + int(c.rsplit('_', 1)[1])}" for c in off.data.columns})
    merged = pd.concat([base.data, renamed], axis=1, sort=True).dropna()
    region.extra_weather_latitudes, region.extra_weather_longitudes = [0.0] * N, [0.0] * N   # only the count matters here
    out = WeatherStore(region, OUT)
    assert set(merged.columns) == out.expected_columns()
    out.data = merged
    await out.serialize()
    print(ZONE.value, "base rows", len(base.data), "offshore rows", len(off.data), "merged rows", len(merged), str(merged.index.min()), "..", str(merged.index.max()), "columns", len(merged.columns), flush=True)


asyncio.run(main())
