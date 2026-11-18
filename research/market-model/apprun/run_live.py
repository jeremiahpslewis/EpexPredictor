import sys, aiohttp
_Orig = aiohttp.ClientSession
class _ProxySession(_Orig):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True); super().__init__(*a, **k)
aiohttp.ClientSession = _ProxySession
sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.weatherstore import WeatherStore
async def _no_download(self, rstart, rend): return False      # the weather is in the stores already (Open-Meteo's daily limit is used up)
WeatherStore.refresh_range = _no_download
import uvicorn
uvicorn.run("predictor.api.priceapi:app", host="127.0.0.1", port=8766, log_level="info")
