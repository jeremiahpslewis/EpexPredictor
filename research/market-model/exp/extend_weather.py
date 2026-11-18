"""Fetch the weather of the last days (and the forecast) into a copy of the caches, through the repo's WeatherStore."""
import asyncio, logging, sys
from datetime import datetime, timezone
sys.path.insert(0, "/home/user/EpexPredictor")
import aiohttp
_Orig = aiohttp.ClientSession
class _ProxySession(_Orig):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True); super().__init__(*a, **k)
aiohttp.ClientSession = _ProxySession
from predictor.model.priceregion import PriceRegionName
from predictor.model.weatherstore import WeatherStore
logging.basicConfig(level=logging.ERROR)
D = sys.argv[1]
S = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc); E = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
async def main():
    for z in sys.argv[4].split(","):
        ws = await WeatherStore(PriceRegionName(z).to_region(), D).load(); ws.known_source_horizon = None
        for attempt in range(12):
            try:
                df = await ws.get_data(S, E); break
            except Exception as e:
                print(z, "retry", attempt, repr(e)[:100], flush=True); await asyncio.sleep(20 + 10 * attempt)
        print(z, "weather until", ws.data.index.max(), flush=True)
asyncio.run(main())
