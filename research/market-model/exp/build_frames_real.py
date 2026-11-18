"""Build per-zone frames through the real (modified) repo code: stage-1 features + the stage-2 Europe-wide
weather features, and let the real gas store fetch its data."""
import asyncio
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")
import aiohttp  # noqa: E402

_Orig = aiohttp.ClientSession


class _ProxySession(_Orig):  # the sandbox only reaches the internet through HTTPS_PROXY
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _ProxySession

from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR, OUT_DIR = sys.argv[1], sys.argv[2]
START = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc)
REGIONS = [PriceRegionName(r) for r in sys.argv[5].split(",")]
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


async def main():
    grid = pd.date_range(START, END, freq="15min", tz="UTC")
    base = {}
    for name in REGIONS:
        base[name] = await PricePredictor(name.to_region(), DATA_DIR).load_from_persistence()
        for s in (base[name].weatherstore, base[name].pricestore):
            async def _nofetch(start, end, _s=s):
                return False
            s.fetch_missing_data = _nofetch  # cached; a gap should fail loudly instead of fetching
    for name in REGIONS:
        p = base[name]
        df = await p.prepare_dataframe(START, END)
        stage2 = PricePredictor(name.to_region(), DATA_DIR, cross_predictors=[base[n] for n in REGIONS])
        stage2.use_datastores_from(p)
        eu = await stage2.get_europe_weather_features(START, END)
        price = df.pop("price")
        df = pd.concat([df, eu], axis=1, sort=True)
        df["price"] = price
        df = df.reindex(grid)
        df.index.name = "time"
        print(name.value, df.shape, "rows with NaN:", int(df.isna().any(axis=1).sum()), "| last cols:", list(df.columns)[-7:], flush=True)
        df.to_parquet(f"{OUT_DIR}/frame_{name.value}.parquet")


asyncio.run(main())
