"""Download and cache raw inputs through the repo's own data stores."""
import asyncio
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/user/EpexPredictor")

import aiohttp  # noqa: E402

# The sandbox only reaches the internet through HTTPS_PROXY; aiohttp ignores it unless trust_env is set.
_OrigSession = aiohttp.ClientSession


class _ProxySession(_OrigSession):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _ProxySession

from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR = sys.argv[1]
START = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
WHAT = sys.argv[4].split(",")  # weather,price,gas
REGIONS = [PriceRegionName(r) for r in sys.argv[5].split(",")]

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


async def main():
    for name in REGIONS:
        p = await PricePredictor(name.to_region(), DATA_DIR).load_from_persistence()
        # the persisted horizon gate would stop us from extending the cached range
        for s in (p.weatherstore, p.pricestore, p.gasstore):
            s.known_source_horizon = None
        if "price" in WHAT:
            df = await p.pricestore.get_data(START, END)
            print(name.value, "price", len(df), df.index.min(), df.index.max(), flush=True)
        if "gas" in WHAT and p.region.use_de_nat_gas_price:
            df = await p.gasstore.get_data(START, END)
            print(name.value, "gas", len(df), df.index.min(), df.index.max(), flush=True)
        if "weather" in WHAT:
            for attempt in range(12):
                try:
                    df = await p.weatherstore.get_data(START, END)
                    break
                except Exception as e:  # transient proxy resets / rate limits: resume from what is cached
                    print(name.value, "weather retry", attempt, repr(e)[:200], flush=True)
                    await asyncio.sleep(20 + 20 * attempt)
            else:
                raise RuntimeError(f"weather download failed for {name.value}")
            print(name.value, "weather", len(df), df.index.min(), df.index.max(), flush=True)
            await asyncio.sleep(3)


asyncio.run(main())
