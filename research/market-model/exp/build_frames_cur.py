"""Build per-zone frames through the repo's current code: prepare_dataframe() of the stage-1 predictors, with the
Europe-wide weather wired in the way the app does it. Optionally with extra weather-only locations for some zones.
usage: build_frames_cur.py DATA_DIR OUT_DIR START END ZONES [EXTRA_JSON]   EXTRA_JSON: {"NL": [[lat, ...], [lon, ...]]}"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")

from predictor.model.europeweather import EuropeWeather  # noqa: E402
from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR, OUT_DIR = sys.argv[1], sys.argv[2]
START = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc)
REGIONS = [PriceRegionName(r) for r in sys.argv[5].split(",")]
EXTRA = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING)


async def main():
    for zone, (lats, lons) in EXTRA.items():
        region = PriceRegionName(zone).to_region()
        region.extra_weather_latitudes, region.extra_weather_longitudes = lats, lons
    grid = pd.date_range(START, END, freq="15min", tz="UTC")
    base = {}
    for name in REGIONS:
        base[name] = await PricePredictor(name.to_region(), DATA_DIR).load_from_persistence()
        for s in (base[name].weatherstore, base[name].pricestore, base[name].gasstore):
            async def _nofetch(start, end, _s=s):
                return False
            s.fetch_missing_data = _nofetch  # everything is cached; a gap should show up instead of being fetched
    europe_weather = EuropeWeather([p.weatherstore for p in base.values()])
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in REGIONS:
        p = base[name]
        p.europe_weather = europe_weather
        df = await p.prepare_dataframe(START, END)
        df = df.reindex(grid)
        df.index.name = "time"
        print(name.value, df.shape, "rows with NaN:", int(df.isna().any(axis=1).sum()), "| wind columns:", sum(c.startswith("wind_") for c in df.columns), flush=True)
        df.to_parquet(f"{OUT_DIR}/frame_{name.value}.parquet")


asyncio.run(main())
