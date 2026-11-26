"""Live data fetch for one delivery day D, the app's way (team E). Run by live_forecast.py; usable alone.

Brings the persistent live store directory SP/live/stores up to date with the APP'S OWN store classes and calls (no
re-implementation of any source), then snapshots it to SP/live/<D>/stores for the forecast step:
  - prices, 8 zones (energy-charts, the app's 31 s politeness limit), from the store's last price to now + 2 d;
  - THE gas (DE, AT, BE stores);
  - TSO day-ahead wind / solar / load (energy-charts), as GenerationForecastStore keeps it (a row only when all three
    figures exist, so D's load alone is not stored: the app's behaviour);
  - weather: WeatherStore.refresh_range(now - 1 d, now + 8 d) for the 8 zones and GridWeatherStore for the DE grid,
    exactly the app's refresh_forecasts window. `now` is recomputed right before each store, so the app's own host
    choice (needs_history_query) picks the normal forecast API, api.open-meteo.com. (In the running app `currts` is
    computed once per cycle and the regions are refreshed with 5-20 s pauses, so after about a minute the app itself
    switches to the historical-forecast host for later regions - an app quirk, not reproduced here.)
  - FALLBACK: if a weather request fails (HTTP 429 daily limit), the identical URL is sent to
    previous-runs-api.open-meteo.com, whose unsuffixed variables are the latest forecast (separate quota; minutely_15
    supported). The host used per store is logged.
Also fetched for the matched variant and the log only (not app inputs):
  - Open-Meteo previous-runs d1/d2 for the 58-point grid, D-8 .. D+1 -> SP/live/<D>/grid_prev_live.parquet;
  - energy-charts day-ahead load / wind / solar for D as served now (published-load evidence);
  - Open-Meteo model metadata (last run init / availability) for the models behind the default forecast.
Writes SP/live/<D>/fetch_log.json. Network goes through the session proxy (aiohttp trust_env).

usage: tsvenv/bin/python live_fetch.py --day 2026-11-28 [--outdir SP/live/2026-11-28]
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
REPO = "/home/user/EpexPredictor"
LIVE_STORES = f"{SP}/live/stores"           # persistent, like the app's data directory
WFT = f"{LIVE_STORES}/weather_fetch_times.json"   # per weather store: when its current forecast was fetched
SEED = f"{SP}/fresh_data/stores"            # initial content (re-fetched app stores up to 2026-11-26)
TZ = ZoneInfo("Europe/Berlin")
ZONES = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]
OM_FALLBACK = "previous-runs-api.open-meteo.com"
OM_HOSTS = ("api.open-meteo.com", "historical-forecast-api.open-meteo.com")
META_MODELS = ("dwd_icon_d2", "dwd_icon_eu", "dwd_icon", "ecmwf_ifs025", "ncep_gfs013", "meteofrance_arome_france_hd",
               "knmi_harmonie_arome_europe", "dmi_harmonie_arome_europe")

REQUESTS: list[dict] = []        # every HTTP request: url (no secrets are used), host, status, time
REWRITE = {"on": False}
_Session = aiohttp.ClientSession


class _LoggedSession(_Session):
    """the app's aiohttp sessions, through the session proxy, with a request log and the Open-Meteo host fallback"""

    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)

    async def _request(self, method, url, *a, **k):
        url = str(url)
        if REWRITE["on"]:
            for h in OM_HOSTS:
                url = url.replace(f"https://{h}/", f"https://{OM_FALLBACK}/")
        t = datetime.now(timezone.utc).isoformat(timespec="seconds")
        resp = await super()._request(method, url, *a, **k)
        REQUESTS.append({"t_utc": t, "host": url.split("/")[2], "status": resp.status, "url": url[:300]})
        return resp


aiohttp.ClientSession = _LoggedSession


def init_env(outdir):
    os.makedirs(LIVE_STORES, exist_ok=True)
    if not os.listdir(LIVE_STORES):
        for fn in os.listdir(SEED):
            if fn.endswith(".json.gz"):
                shutil.copy2(f"{SEED}/{fn}", f"{LIVE_STORES}/{fn}")
    os.environ["EPEXPREDICTOR_DATADIR"] = LIVE_STORES
    os.environ.pop("EPEXPREDICTOR_ENTSOE_API_KEY", None)   # the deployed store set has no ENTSO-E data
    sys.path.insert(0, REPO)
    os.makedirs(outdir, exist_ok=True)


def get_json(url):
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = json.loads(r.read())
            REQUESTS.append({"t_utc": t, "host": url.split("/")[2], "status": 200, "url": url[:300]})
            return data
        except urllib.error.HTTPError as e:
            REQUESTS.append({"t_utc": t, "host": url.split("/")[2], "status": e.code, "url": url[:300]})
            raise
        except (urllib.error.URLError, ConnectionError, OSError):
            if attempt == 2:
                raise
            time.sleep(10)


def grid_prev_runs(d: date, path: str) -> dict:
    """previous-runs d1/d2 (and d3) for the 58-point grid, D-8 .. D+1, columns as dense/prev.parquet"""
    pts = json.load(open(f"{SP}/dense/points.json"))["points"]
    gv = {"wind_speed_120m": "w120", "shortwave_radiation": "sw", "temperature_2m": "t2"}
    s, e = d - timedelta(days=8), d + timedelta(days=1)
    parts = []
    for i0 in range(0, len(pts), 15):
        chunk = pts[i0:i0 + 15]
        url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(p[0]) for p in chunk)
               + "&longitude=" + ",".join(str(p[1]) for p in chunk) + f"&start_date={s}&end_date={e}&hourly="
               + ",".join(f"{v}_previous_day{k}" for v in gv for k in (1, 2, 3)) + "&timezone=UTC")
        data = get_json(url)
        data = [data] if isinstance(data, dict) else data
        assert len(data) == len(chunk), data if isinstance(data, dict) else len(data)
        for i, loc in enumerate(data):
            parts.append(pd.DataFrame({f"{gv[v]}_{i0 + i}_d{k}": loc["hourly"][f"{v}_previous_day{k}"] for v in gv for k in (1, 2, 3)},
                                      index=pd.to_datetime(loc["hourly"]["time"], utc=True)))
        time.sleep(3)
    df = pd.concat(parts, axis=1).astype("float32")
    df.to_parquet(path)
    fetched = pd.Timestamp.now(tz="UTC")
    out = {}
    for k in (1, 2, 3):
        cols = [c for c in df.columns if c.endswith(f"_d{k}")]
        last = df[cols].dropna(how="all").index.max()
        # a value is a genuine k-day-old forecast only if its valid time - 24h*k is before the fetch
        out[f"d{k}_last_valid"] = str(last)
        out[f"d{k}_last_genuine_valid"] = str(min(last, (fetched + pd.Timedelta(hours=24 * k)).floor("1h")))
    out["rows"], out["fetched_utc"] = len(df), fetched.isoformat(timespec="seconds")
    return out


def model_meta() -> dict:
    out = {}
    for m in META_MODELS:
        try:
            j = get_json(f"https://api.open-meteo.com/data/{m}/static/meta.json")
            out[m] = {k: datetime.fromtimestamp(j[k], timezone.utc).isoformat() for k in
                      ("last_run_initialisation_time", "last_run_availability_time", "last_run_modification_time") if k in j}
        except Exception as e:  # noqa: BLE001 - log only
            out[m] = {"error": repr(e)[:200]}
    return out


async def published_today(region, d: date) -> dict:
    """energy-charts day-ahead load / wind / solar for D as served now (one request each, the app's fetch_forecast)"""
    from predictor.model.residualload import GenerationForecastStore
    g = GenerationForecastStore(region, None)
    D0 = pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=TZ)).tz_convert("UTC")
    D1 = pd.Timestamp(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TZ)).tz_convert("UTC") - pd.Timedelta(minutes=15)
    out = {}
    for pt in ("load", "wind_onshore", "solar"):
        s = await g.fetch_forecast(pt, D0.to_pydatetime(), D1.to_pydatetime())
        s = s.loc[D0:D1] if s is not None else None
        out[pt] = {"slots_on_D": 0 if s is None else int(s.notna().sum()), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if pt == "load" and s is not None and len(s):
            out[pt]["values_GW"] = [round(float(x) / 1000, 4) for x in s.to_numpy()]
            out[pt]["index_first"], out[pt]["index_last"] = str(s.index.min()), str(s.index.max())
    return out


async def fetch_stores(d: date, only=None) -> dict:
    from predictor.model import gaspricestore
    from predictor.model.gaspricestore import GasPriceStore
    from predictor.model.pricestore import PriceStore
    from predictor.model.priceregion import PriceRegionName
    from predictor.model.residualload import GenerationForecastStore, GridWeatherStore
    from predictor.model.weatherstore import WeatherStore
    gaspricestore.THE_USER_AGENT = "research-forecast"   # generic client name
    log: dict = {"weather": {}, "prices": {}, "gas": {}}
    de = PriceRegionName.DE.to_region()
    now = datetime.now(timezone.utc)

    async def prices():
        for z in ZONES:
            p = await PriceStore(PriceRegionName(z).to_region(), LIVE_STORES).load()
            before = p.get_last_known()
            await p.fetch_missing_data(before - timedelta(hours=1), now + timedelta(days=2))
            log["prices"][z] = {"before": str(before), "last": str(p.get_last_known()), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    async def generation():
        g = await GenerationForecastStore(de, LIVE_STORES).load()
        before = g.data.index.max()
        await g.fetch_missing_data(before - timedelta(days=1), now + timedelta(days=2))
        log["generation"] = {"before": str(before), "last": str(g.data.dropna().index.max()), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log["published_on_D"] = await published_today(de, d)

    async def gas():
        for z in ("DE", "AT", "BE"):
            g = await GasPriceStore(PriceRegionName(z).to_region(), LIVE_STORES).load()
            before = g.data.dropna().index.max()
            await g.fetch_missing_data(before, now + timedelta(days=2))
            log["gas"][z] = {"before": str(before), "last": str(g.data.dropna().index.max()), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    async def weather():
        stores = [(z, WeatherStore, PriceRegionName(z).to_region()) for z in ZONES] + [("GRID", GridWeatherStore, de)]
        w_t0 = time.time()
        for name, cls, region in stores:
            w = await cls(region, LIVE_STORES).load()
            before = w.data.index.max()
            n0 = len(REQUESTS)
            entry = {"before": str(before)}
            # primary, fallback; then twice more after a pause (the 429s seen on 2026-11-26 came and went within seconds)
            for i, attempt in enumerate(("primary", "fallback", "primary", "fallback", "primary", "fallback")):
                if i in (2, 4):
                    if time.time() - w_t0 > 240:   # weather budget: no more pauses after 4 min, the stored forecast stays
                        break
                    await asyncio.sleep(45 * i // 2)
                REWRITE["on"] = attempt == "fallback"
                t = datetime.now(timezone.utc)      # recomputed per store: the app's host choice gives the forecast API
                try:
                    if before < t - timedelta(days=1):   # a gap before the refresh window (history host, the app's fill)
                        await w.fetch_missing_data(before, t - timedelta(days=1))
                    await w.refresh_range(t - timedelta(days=1), t + timedelta(days=8))
                    entry["source"] = attempt
                    break
                except Exception as e:  # noqa: BLE001 - 429 daily limit comes back as RuntimeError from the app
                    entry.setdefault("errors", []).append(f"{attempt}: {repr(e)[:150]}")
                finally:
                    REWRITE["on"] = False
            entry["hosts"] = sorted({r["host"] for r in REQUESTS[n0:] if "open-meteo" in r["host"]})
            entry["last"] = str(w.data.dropna().index.max())
            entry["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # when the weather the store now holds was fetched: this run, or (both hosts failed) the last good fetch,
            # which stays in the persistent store - a forecast on it is still issued, labelled
            wft = json.load(open(WFT)) if os.path.exists(WFT) else {}
            if "source" in entry:
                wft[name] = {"fetched_utc": entry["fetched_utc"], "source": entry["source"], "hosts": entry["hosts"]}
                json.dump(wft, open(WFT, "w"), indent=1)
            else:
                entry["source"] = "stored"
            entry["weather_fetched_utc"] = wft.get(name, {}).get("fetched_utc", "seed (fresh_data, 2026-11-26 ~10 UTC)")
            entry["weather_source"] = wft.get(name, {}).get("source", "seed")
            log["weather"][name] = entry
            await asyncio.sleep(2)

    parts = {"prices": prices, "generation": generation, "gas": gas, "weather": weather}
    await asyncio.gather(*(f() for k, f in parts.items() if only is None or k == only))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--only", choices=("weather",), help="refresh the weather only and update the existing fetch_log.json")
    a = ap.parse_args()
    d = date.fromisoformat(a.day)
    outdir = a.outdir or f"{SP}/live/{d}"
    init_env(outdir)
    t0 = time.time()
    if a.only:
        log = json.load(open(f"{outdir}/fetch_log.json"))
        REQUESTS[:0] = log.get("requests", [])
        log["weather_refetch_start_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        log = {"day": str(d), "fetch_start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    log["openmeteo_model_runs_at_start"] = model_meta()
    new = asyncio.run(fetch_stores(d, a.only))
    if a.only:
        log["weather"] = new["weather"]
    else:
        log |= new
    if not a.only or "error" in log.get("grid_prev_live", {"error": 1}):
        try:
            log["grid_prev_live"] = grid_prev_runs(d, f"{outdir}/grid_prev_live.parquet")
        except Exception as e:  # noqa: BLE001 - only the matched variant needs it
            log["grid_prev_live"] = {"error": repr(e)[:300]}
    log["openmeteo_model_runs_at_end"] = model_meta()
    # snapshot of the stores the forecast reads (the persistent directory moves on the next day)
    snap = f"{outdir}/stores"
    if os.path.exists(snap):
        shutil.rmtree(snap)
    shutil.copytree(LIVE_STORES, snap, ignore=shutil.ignore_patterns("*.tmp", "*log*"))
    log["fetch_end_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["fetch_secs"] = round(time.time() - t0, 1)
    log["requests"] = REQUESTS
    json.dump(log, open(f"{outdir}/fetch_log.json", "w"), indent=1)
    w = {k: (v.get("source"), v.get("hosts"), v.get("last")) for k, v in log["weather"].items()}
    print(json.dumps({"prices_DE": log.get("prices", {}).get("DE"), "generation": log.get("generation"), "gas_DE": log["gas"].get("DE"),
                      "weather": w, "grid_prev_live": log["grid_prev_live"], "secs": log["fetch_secs"]}, indent=1))


if __name__ == "__main__":
    main()
