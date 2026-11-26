"""Fetch what the faithful harness needs for fresh target days (after 2026-11-08) into SP/fresh_data. Resumable.

1. Stores, with the app's own store classes (same sources as the deployed app), into SP/fresh_data/stores, a copy
   of SP/live_data (which stays untouched):
   - prices, 8 zones (energy-charts), gas (THE), TSO day-ahead wind/solar/load (energy-charts): missing ranges only;
   - zone weather (weather_v2) and DE grid weather (gridweather_v1): the live_data copies were written on 2026-11-18,
     so their rows from 2026-11-17 on are forecasts, not archive. They are dropped and re-fetched from the
     historical-forecast API (the host the app uses for past ranges).
2. Open-Meteo previous-runs vintages d1..d3 from 2026-11-01 (overlap with the local files, for a consistency check):
   - every zone's weather points, the 5 weather_v2 variables: fresh_data/prev_<zone>.parquet (columns <var>_<i>_d<k>);
   - the 58-point grid, wind_speed_120m / shortwave_radiation / temperature_2m: fresh_data/grid_prev.parquet (columns
     w120_<i>_d<k>, sw_<i>_d<k>, t2_<i>_d<k>, as dense/prev.parquet). Its d2 also extends the matched models' history.
Stops at the first HTTP error (rate limit). No API key is used (ENTSO-E is off, as in the deployed store set).

usage: venv/bin/python fetch_fresh.py [END]       (END = last date of vintages, default 2026-11-28)
"""
import asyncio
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

import aiohttp
import pandas as pd

# the app's aiohttp sessions do not read HTTPS_PROXY; here every request has to go through the session's proxy
_Session = aiohttp.ClientSession


class _ProxySession(_Session):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _ProxySession

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
FRESH = f"{SP}/fresh_data"
STORES = f"{FRESH}/stores"
os.makedirs(STORES, exist_ok=True)
for fn in os.listdir(f"{SP}/live_data"):
    if fn.endswith(".json.gz") and not os.path.exists(f"{STORES}/{fn}"):
        shutil.copy2(f"{SP}/live_data/{fn}", f"{STORES}/{fn}")
os.environ["EPEXPREDICTOR_DATADIR"] = STORES
os.environ.pop("EPEXPREDICTOR_ENTSOE_API_KEY", None)
sys.path.insert(0, "/home/user/EpexPredictor")

from predictor.model import gaspricestore  # noqa: E402
from predictor.model.gaspricestore import GasPriceStore  # noqa: E402
from predictor.model.pricestore import PriceStore  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.residualload import GenerationForecastStore, GridWeatherStore  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402

gaspricestore.THE_USER_AGENT = "research-backtest"   # generic client name
ZONES = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]
VSTART = date(2026, 11, 1)
VEND = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 11, 28)
ARCHIVE_FROM = datetime(2026, 11, 8, tzinfo=timezone.utc)   # live_data departs from the archive from ~11-11 (grid) / ~11-15 (zones)
STORE_FROM = datetime(2026, 11, 8, tzinfo=timezone.utc)
STORE_TO = datetime(2026, 11, 28, tzinfo=timezone.utc)
ZVARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance",
         "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
GVARS = {"wind_speed_120m": "w120", "shortwave_radiation": "sw", "temperature_2m": "t2"}


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r} - stopping")
        except (urllib.error.URLError, ConnectionError, ValueError, OSError) as e:   # resets/cut transfers via the proxy: retry twice
            print("retry", attempt, repr(e)[:120], flush=True)
            if attempt == 2:
                raise SystemExit("connection failed 3 times - stopping")
            time.sleep(20)
    if isinstance(data, dict) and "hourly" in data:
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"unexpected answer, stopping: {str(data)[:300]}")
    return data


def prev_runs(lats, lons, variables: dict, name_of, extra=""):
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, lats)) + "&longitude="
           + ",".join(map(str, lons)) + f"{extra}&start_date={VSTART}&end_date={VEND}&hourly="
           + ",".join(f"{v}_previous_day{k}" for v in variables for k in (1, 2, 3)) + "&timezone=UTC")
    data = get(url)
    assert len(data) == len(lats)
    return pd.concat([pd.DataFrame({name_of(v, i, k): loc["hourly"][f"{v}_previous_day{k}"] for v in variables for k in (1, 2, 3)},
                                   index=pd.to_datetime(loc["hourly"]["time"], utc=True)) for i, loc in enumerate(data)], axis=1)


def covered(fn):
    if not os.path.exists(fn):
        return False
    d = pd.read_parquet(fn)
    return d.index.min() <= pd.Timestamp(VSTART, tz="UTC") and d.index.max() >= pd.Timestamp(VEND, tz="UTC") + pd.Timedelta(hours=23)


def save(df, fn):
    df.astype("float32").to_parquet(fn + ".tmp")
    os.replace(fn + ".tmp", fn)
    print(os.path.basename(fn), df.shape, df.index.min(), df.index.max(), "nulls", int(df.isna().sum().sum()), flush=True)


def marker(name):
    return f"{FRESH}/done_{name}"


async def step(name, fn):
    """one store step, skipped when its marker exists; a weather refresh that fails is retried twice"""
    if os.path.exists(marker(name)):
        print(name, "done earlier, skipped", flush=True)
        return
    for attempt in range(3):
        try:
            print(name, await fn(), flush=True)
            open(marker(name), "w").write(datetime.now(timezone.utc).isoformat())
            return
        except Exception as e:  # noqa: BLE001 - proxy resets come back as empty bodies / JSON errors
            print(name, "retry", attempt, repr(e)[:120], flush=True)
            await asyncio.sleep(20)
    raise SystemExit(f"{name}: failed 3 times - stopping")


async def stores():
    de = PriceRegionName.DE.to_region()

    async def prices(z):
        p = PriceStore(PriceRegionName(z).to_region(), STORES)
        await p.load()
        await p.fetch_missing_data(STORE_FROM, STORE_TO)
        return f"prices to {p.get_last_known()}"

    async def gas(z="DE"):     # the DE gas price, stored per region that uses it (DE, AT, BE)
        g = GasPriceStore(PriceRegionName(z).to_region(), STORES)
        await g.load()
        await g.fetch_missing_data(STORE_FROM, STORE_TO)
        return f"gas to {g.data.dropna().index.max()}"

    async def generation():
        gen = GenerationForecastStore(de, STORES)
        await gen.load()
        await gen.fetch_missing_data(STORE_FROM, STORE_TO)
        return f"generation to {gen.data.dropna().index.max()}"

    end = datetime(2026, 11, 26, 23, tzinfo=timezone.utc)

    async def weather(cls, region, slot):
        w = cls(region, STORES)
        await w.load()      # from disk every attempt: a failed attempt has not been written
        w.drop_after(ARCHIVE_FROM - slot)
        await w.refresh_range(ARCHIVE_FROM, end)
        await w.serialize()
        return f"archive to {w.data.dropna().index.max()}"

    for z in ZONES:
        await step(f"prices_{z}", lambda z=z: prices(z))
    await step("gas", gas)
    for z in ("AT", "BE"):
        await step(f"gas_{z}", lambda z=z: gas(z))
    await step("generation", generation)
    for z in ZONES:
        await step(f"weather_{z}", lambda z=z: weather(WeatherStore, PriceRegionName(z).to_region(), pd.Timedelta(minutes=15)))
    await step("gridweather", lambda: weather(GridWeatherStore, de, pd.Timedelta(hours=1)))


def vintages():
    for z in ZONES:
        fn = f"{FRESH}/prev_{z}.parquet"
        if covered(fn):
            print(fn, "complete, skipped")
            continue
        r = PriceRegionName(z).to_region()
        df = prev_runs(r.weather_latitudes, r.weather_longitudes, ZVARS, lambda v, i, k: f"{ZVARS[v]}_{i}_d{k}", "&azimuth=0&tilt=0")
        save(df, fn)
        time.sleep(5)
    fn = f"{FRESH}/grid_prev.parquet"
    if covered(fn):
        print(fn, "complete, skipped")
        return
    pts = json.load(open(f"{SP}/dense/points.json"))["points"]
    parts = []
    for s in range(0, len(pts), 15):
        chunk = pts[s:s + 15]
        parts.append(prev_runs([p[0] for p in chunk], [p[1] for p in chunk], GVARS, lambda v, i, k, s=s: f"{GVARS[v]}_{s + i}_d{k}"))
        time.sleep(10)
    save(pd.concat(parts, axis=1), fn)


if __name__ == "__main__":
    t = time.time()
    vintages()
    asyncio.run(stores())
    print(f"done in {time.time() - t:.0f}s")
