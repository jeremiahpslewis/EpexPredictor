"""The deployed app's own forecast path on a fetched store snapshot (team E, live file (a)).

Runs predictor/ unchanged, the way the running app produces its forecast: BaseModelCoordinator.ensure_loaded() and
_do_update() (train window now - 540 d .. now + 7 d), then RegionPriceManager(DE).update_data_if_needed() (stage 2,
Timestral, market route, TimestralBlend.blend). The only interventions:
  - network and persistence off (the stores were fetched just before by live_fetch.py with the app's classes);
  - prices and TSO figures cut after D-1 23:45 local (a no-op on the morning of D-1; needed for a dry run after the
    auction). Gas is not cut (app behaviour; THE serves gas day D before the gate);
  - ForecastLog.append is captured instead of written: it receives exactly what the app logs, {trees, timestral,
    market, price}, where `price` is the blend the app serves.
Writes <out>/app_<D>.npz with idx (UTC ns) and trees / timestral / market / issued on D's slots, and meta (JSON).

usage: tsvenv/bin/python app_run.py --day D --stores DIR --out DIR [--torch-threads N]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--day", required=True)
ap.add_argument("--stores", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--torch-threads", type=int, default=0)
A = ap.parse_args()

os.environ["EPEXPREDICTOR_DATADIR"] = A.stores
os.environ["EPEXPREDICTOR_TIMESTRAL_MODEL"] = "mistralai/Timestral-123M"   # as deployed (revision pinned by the app)
os.environ.pop("EPEXPREDICTOR_ENTSOE_API_KEY", None)
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/user/EpexPredictor")
TZ = ZoneInfo("Europe/Berlin")

from predictor.model.datastore import DataStore  # noqa: E402
from predictor.model.entsoedatastore import EntsoeDataStore  # noqa: E402
from predictor.model.forecastlog import ForecastLog  # noqa: E402
from predictor.model.gaspricestore import GasPriceStore  # noqa: E402
from predictor.model.pricestore import PriceStore  # noqa: E402
from predictor.model.residualload import GenerationForecastStore  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402


async def _nothing(self, *a, **k):
    return False


async def _no_serialize(self):
    return None


for cls in (PriceStore, GasPriceStore, EntsoeDataStore, WeatherStore, GenerationForecastStore):
    cls.fetch_missing_data = _nothing
WeatherStore.refresh_range = _nothing
EntsoeDataStore.refresh_range = _nothing
DataStore.serialize = _no_serialize
CAPTURED: dict = {}


def _capture(self, last_known, columns, issued=None):
    CAPTURED[os.path.basename(self.path or "none").split("_")[0]] = (last_known, {k: v.copy() for k, v in columns.items() if v is not None})
    return True


ForecastLog.append = _capture

from predictor.api import priceapi as api  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402


async def main():
    if A.torch_threads > 0:
        import torch
        torch.set_num_threads(A.torch_threads)
    d = date.fromisoformat(A.day)
    D0 = pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=TZ)).tz_convert("UTC")
    D = pd.date_range(D0, pd.Timestamp(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TZ)).tz_convert("UTC"),
                      freq="15min", inclusive="left")
    cutoff = D[0] - pd.Timedelta(minutes=15)
    t0 = time.time()
    await api.base_models.ensure_loaded()
    beyond = {}
    for name, p in api.base_models.base.items():
        # prices cut at the cutoff (no-op on the morning of D-1). Gas is NOT cut: the app keeps whatever THE serves,
        # and THE serves gas day D (preliminary) before the gate (seen 2026-11-26 09:55 UTC for gas day 11-27)
        s = p.pricestore
        beyond[f"{s.storage_fn_prefix}_{name.value}"] = int((s.data.index > cutoff).sum())
        s.drop_after(cutoff.to_pydatetime())
        if len(p.gasstore.data):
            beyond[f"gas_kept_after_cutoff_{name.value}"] = int((p.gasstore.data.index > cutoff).sum())
        if p.residual_load is not None:
            g = p.residual_load.generation
            beyond[f"generation_{name.value}"] = int((g.data.index > cutoff).sum())
            g.drop_after(cutoff.to_pydatetime())
    de = api.base_models.base[PriceRegionName.DE]
    assert de.pricestore.get_last_known() == cutoff, (de.pricestore.get_last_known(), cutoff)
    await api.base_models._do_update()
    t1 = time.time()
    mgr = api.RegionPriceManager(PriceRegionName.DE, PriceRegionName.DE.to_region())
    await mgr.update_data_if_needed()
    t2 = time.time()
    lk, cols = CAPTURED["forecastlog"] if "forecastlog" in CAPTURED else next(iter(CAPTURED.values()))
    res = {"idx": D.asi8}
    for k_out, k_in in (("trees", "trees"), ("timestral", "timestral"), ("market", "market"), ("issued", "price")):
        res[k_out] = cols[k_in].reindex(D).to_numpy(float) if k_in in cols else np.full(len(D), np.nan)
    served = mgr.cachedprices["price"].reindex(D).to_numpy(float)
    assert np.allclose(served, res["issued"], equal_nan=True), "captured blend != served prices"
    assert np.isfinite(res["issued"]).all(), "app blend has gaps on D"
    meta = {"last_known_price": str(lk), "cutoff": str(cutoff), "rows_beyond_cutoff_dropped": beyond,
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "secs": {"base": round(t1 - t0, 1), "stage2_blend": round(t2 - t1, 1)},
            "blend_is_mean_of_thirds": bool(np.allclose(res["issued"], (res["trees"] + res["timestral"] + res["market"]) / 3))}
    os.makedirs(A.out, exist_ok=True)
    np.savez(f"{A.out}/app_{d}.npz", meta=json.dumps(meta), **res)
    print(json.dumps(meta), flush=True)


asyncio.run(main())
