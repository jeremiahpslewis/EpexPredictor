"""Build per-region stage-1 feature frames with the repo's own prepare_dataframe (no cross features)."""
import asyncio
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")

from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
START = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc)
REGIONS = [PriceRegionName(r) for r in sys.argv[5].split(",")]

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING)


async def main():
    grid = pd.date_range(START, END, freq="15min", tz="UTC")
    for name in REGIONS:
        p = await PricePredictor(name.to_region(), DATA_DIR).load_from_persistence()
        # everything is cached: forbid any network access so a gap fails loudly instead of fetching
        for s in (p.weatherstore, p.pricestore, p.gasstore):
            async def _nofetch(start, end, _s=s):
                return False
            s.fetch_missing_data = _nofetch
        df = await p.prepare_dataframe(START, END)
        df = df.reindex(grid)
        df.index.name = "time"
        nan_rows = int(df.isna().any(axis=1).sum())
        print(name.value, df.shape, "rows with NaN:", nan_rows, "| first/last complete:",
              df.dropna().index.min(), df.dropna().index.max(), flush=True)
        if nan_rows:
            print("   NaN per column:", df.isna().sum()[df.isna().sum() > 0].to_dict())
        print("   columns:", list(df.columns))
        df.to_parquet(f"{OUT_DIR}/frame_{name.value}.parquet")


asyncio.run(main())
