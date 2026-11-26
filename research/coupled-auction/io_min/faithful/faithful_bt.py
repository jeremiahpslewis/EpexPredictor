"""Faithful as-of-gate backtest of the DEPLOYED EpexPredictor ensemble for Germany (DE_LU).

Drives the app code in predictor/ (nothing re-implemented): the 8 stage-1 base models and the DE stage-2 model as
priceapi.BaseModelCoordinator / RegionPriceManager build them, the residual load forecast (retrained per cutoff),
Timestral on the price, the slim market route and TimestralBlend.blend, exactly as RegionPriceManager calls it.

Per target local delivery day D:
  gate   = D-1 12:00 Europe/Berlin (day-ahead auction gate)
  cutoff = D-1 23:45 local, the last quarter hour with a known price (prices of D-1 are known at the gate)
  * prices (8 zones), gas and the TSO generation forecasts are physically cut after the cutoff in the working copy
    AND hidden by set_horizon_cutoff;
  * the weather (8 zones' weather_v2 + DE gridweather_v1) from the gate onward is DELETED and replaced by Open-Meteo
    previous-runs vintages; nothing after D+2 is kept (see README.md for the vintage policy);
  * residual load, stage-1 and stage-2 trees are retrained on [cutoff - 540 d, cutoff];
  * Timestral + market route are run and blended as the app does.
One .npz per day in OUT (resumable), plus a log line with the runtime.

usage: tsvenv/bin/python faithful_bt.py --days 2026-09-08,2026-10-26 [--vintage strict|d1] [--latency 6] [--out DIR]
       tsvenv/bin/python faithful_bt.py --start 2026-09-08 --end 2026-11-08 ...
"""
import argparse
import asyncio
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = "/home/user/EpexPredictor"
SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
# FAITHFUL_FRESH=1: fresh target days (after 2026-11-08) from the data fetch_fresh.py put in SP/fresh_data (stores
# re-fetched up to 2026-11-26, vintages d1..d3 2026-11-01..11-28 appended to the local ones); own working copy
FRESH = os.environ.get("FAITHFUL_FRESH", "") == "1"
FRESH_DIR = f"{SP}/fresh_data"
SOURCE_STORES = f"{FRESH_DIR}/stores" if FRESH else f"{SP}/live_data"   # read-only source; copied once into WORK
WORK = f"{SP}/faithful/stores_fresh" if FRESH else f"{SP}/faithful/stores"
TZ = ZoneInfo("Europe/Berlin")
SLOT = pd.Timedelta(minutes=15)
STORE_FILES = ("prices_v3_", "gasprices_the_v1_", "weather_v2_", "gridweather_v1_", "generation_v1_")
# hourly radiation is the mean of the preceding hour, the 15-minute one of the preceding quarter hour
IRRADIANCE_SHIFT = pd.Timedelta(minutes=-22.5)
ZONE_FILE = {"DE_LU": "DE", "AT": "AT", "BE": "BE", "FR": "FR", "NL": "NL", "SE_4": "SE4", "DK_1": "DK1", "DK_2": "DK2"}
MAX_K = 3  # deepest vintage held locally

# ---------------------------------------------------------------------------------------------------- environment
os.makedirs(WORK, exist_ok=True)
for fn in os.listdir(SOURCE_STORES):
    if fn.startswith(STORE_FILES) and not os.path.exists(f"{WORK}/{fn}"):
        shutil.copy2(f"{SOURCE_STORES}/{fn}", f"{WORK}/{fn}")
os.environ["EPEXPREDICTOR_DATADIR"] = WORK
os.environ["EPEXPREDICTOR_TIMESTRAL_MODEL"] = "mistralai/Timestral-123M"   # as deployed (revision pinned by the app)
os.environ.pop("EPEXPREDICTOR_ENTSOE_API_KEY", None)                       # deployed store set has no ENTSO-E data
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, REPO)

from predictor.model.datastore import DataStore  # noqa: E402
from predictor.model.pricestore import PriceStore  # noqa: E402
from predictor.model.gaspricestore import GasPriceStore  # noqa: E402
from predictor.model.entsoedatastore import EntsoeDataStore  # noqa: E402
from predictor.model.weatherstore import WeatherStore  # noqa: E402
from predictor.model.residualload import GenerationForecastStore  # noqa: E402


async def _nothing(self, *a, **k):
    return False


async def _no_serialize(self):
    return None


def _cutoff_only(self, start, end):
    # downloads are off, so the source-horizon gate (wall-clock based) is meaningless here: only the cutoff applies
    if self.horizon_cutoff:
        start, end = min(start, self.horizon_cutoff), min(end, self.horizon_cutoff)
    return start, end


for cls in (PriceStore, GasPriceStore, EntsoeDataStore, WeatherStore, GenerationForecastStore):
    cls.fetch_missing_data = _nothing
WeatherStore.refresh_range = _nothing
EntsoeDataStore.refresh_range = _nothing
DataStore.serialize = _no_serialize
DataStore.apply_horizon = _cutoff_only

from predictor.api import priceapi as api  # noqa: E402
from predictor.model import pricepredictor as pp  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.residualload import ResidualLoad, generation_features  # noqa: E402
from predictor.model.timestralload import TimestralLoad  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger("faithful")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------- RL variants
# The two recommendations of ../rl/RESULTS.md, patched onto ResidualLoad at class level (with_shared_stores() builds
# its copy with ResidualLoad.__new__, so a subclass would not reach stage 2). Every consumer of the residual load reads
# it through ResidualLoad.get_data (PricePredictor.prepare_dataframe -> tree features of stage 1 and 2, and
# TimestralBlend.inputs -> Timestral's covariate and the market route, after TimestralLoad's mix).
#   loadcorr: tso_load + mean(published - held-out forecast) over the last 24 qh up to the cutoff * exp(-(j+1)/96),
#             j = 0 at the first slot after the cutoff (../rl/combo.py, daily_load.py "decay"). The held-out load
#             booster is trained on rows up to noon D-2 (the study's d0 - 144), so D-1 is out of sample as in the study.
#   matched:  slots whose grid weather is d2 or older use boosters trained on the _previous_day2 grid history
#             (SP/dense/hist.parquet, *_d2 columns, from 2025-10-01); d1 slots keep the app's boosters.
#   both:     matched, each model set corrected with its own D-1 load residual ("VM+decay(load)" in combo.py).
# Only rows after the cutoff without a published figure change; everything up to the cutoff is the app's.
RL_DECAY_QH = 96
RL_CORR_QH = 24
RL_HOLDOUT = pd.Timedelta(hours=36)     # held-out load models: rows < first slot of D - 36 h (= up to D-2 11:45 local)
D2_HIST = f"{SP}/dense/hist.parquet"


class RLState:
    variant = "base"
    cutoff: pd.Timestamp | None = None
    gate: pd.Timestamp | None = None
    latency: pd.Timedelta | None = None
    policy = "strict"
    day_m1: date | None = None
    grid_new: pd.DataFrame | None = None        # staged grid vintages (hourly), for the leak checks
    d2: pd.DataFrame | None = None              # hourly _previous_day2 grid history, store column names
    cache: dict = {}                            # extra boosters, shared by the DE base and stage-2's copy (same data)
    info: dict = {}

    @classmethod
    def uses(cls, part: str) -> bool:
        return cls.variant == part or cls.variant == "both"


RLV = RLState


def load_d2_history() -> pd.DataFrame:
    h = pd.read_parquet(D2_HIST)
    if FRESH:
        h = h.combine_first(pd.read_parquet(f"{FRESH_DIR}/grid_prev.parquet"))
    names = {"w120": "wind", "sw": "radiation", "t2": "temp"}
    cols = [c for c in h.columns if c.endswith("_d2") and c.split("_")[0] in names]
    return h[cols].rename(columns=lambda c: f"{names[c.split('_')[0]]}_{c.split('_')[1]}").astype(float).sort_index()


def slot_k(index: pd.DatetimeIndex) -> np.ndarray:
    """grid weather vintage per 15-min slot (same rule as the staging, applied per quarter hour); 0 before the gate"""
    k = wanted_k(index, RLV.gate, RLV.policy, RLV.latency, RLV.day_m1)
    return np.where(index >= RLV.gate, k, 0)


def _fit(features: pd.DataFrame, label: pd.Series) -> "lgb.Booster":
    import lightgbm as lgb
    return lgb.train(params=ResidualLoad.LGB_PARAMS, train_set=lgb.Dataset(features, label=label), num_boost_round=ResidualLoad.LGB_ROUNDS)


async def _training_set(rl: ResidualLoad, start, end, weather: str):
    """features (app's generation_features) and published targets on [start, end]; weather 'app' = the store, 'd2'"""
    if weather == "app":
        features = await rl.get_features(start, end)
    else:
        aux = await rl.auxstore.get_data(start, end)
        s, e = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
        w = RLV.d2.loc[s.floor("1h"):e.ceil("1h")]
        # leak check: valid time no later than the first hour after the training end, issued 48 h earlier <= gate
        assert len(w) == 0 or (w.index.max() <= e.ceil("1h") and w.index.max() - pd.Timedelta(hours=48) <= RLV.gate - RLV.latency)
        features = generation_features(rl.region, w, aux)
    published = (await rl.generation.get_data(start, end)).reindex(index=features.index, columns=list(ResidualLoad.TARGETS))
    usable = features.notna().all(axis=1) & published.notna().all(axis=1)
    assert published.loc[usable].index.max() <= RLV.cutoff
    return features[usable], published[usable]


async def _extra_models(rl: ResidualLoad, start, end) -> dict:
    key = (pd.Timestamp(start), pd.Timestamp(end), RLV.cutoff)
    if key in RLV.cache:
        return RLV.cache[key]
    first_d = RLV.cutoff + SLOT
    hold_end = (first_d - RL_HOLDOUT - SLOT).to_pydatetime()
    out, info = {}, {}
    if RLV.uses("loadcorr"):
        X, Y = await _training_set(rl, start, hold_end, "app")
        assert X.index.max() <= first_d - RL_HOLDOUT - SLOT
        out["hold_app_load"] = await asyncio.to_thread(_fit, X, Y["tso_load"])
        info["hold_app"] = [str(X.index.min()), str(X.index.max()), int(len(X))]
    if RLV.uses("matched"):
        X, Y = await _training_set(rl, start, end, "d2")
        out["matched"] = {t: await asyncio.to_thread(_fit, X, Y[t]) for t in ResidualLoad.TARGETS}
        info["matched"] = [str(X.index.min()), str(X.index.max()), int(len(X))]
        if RLV.variant == "both":
            X, Y = await _training_set(rl, start, hold_end, "d2")
            assert X.index.max() <= first_d - RL_HOLDOUT - SLOT
            out["hold_matched_load"] = await asyncio.to_thread(_fit, X, Y["tso_load"])
            info["hold_matched"] = [str(X.index.min()), str(X.index.max()), int(len(X))]
    # D-1 load residuals: the last 24 qh up to the cutoff, published TSO load against the held-out forecast made from
    # the staged grid weather (d1 there under strict: forecasts made 24 h before valid time, all <= gate - latency)
    win = pd.date_range(RLV.cutoff - (RL_CORR_QH - 1) * SLOT, RLV.cutoff, freq="15min")
    feats = (await rl.get_features((win[0] - pd.Timedelta(hours=2)).to_pydatetime(), (win[-1] + pd.Timedelta(hours=3)).to_pydatetime())).reindex(win)
    pub = (await rl.generation.get_data(win[0].to_pydatetime(), win[-1].to_pydatetime()))["tso_load"].reindex(win)
    assert pub.notna().all() and feats.notna().all().all(), "D-1 correction window incomplete"
    assert win[-1] == RLV.cutoff and rl.generation.data.index.max() <= RLV.cutoff
    kw = slot_k(win)
    if RLV.policy == "strict":
        assert ((kw >= 1) & (win - pd.Timedelta(hours=24) * kw <= RLV.gate - RLV.latency)).all(), "D-1 weather newer than the gate"
    # the window's weather is the staged vintage, not the archive
    hrs = RLV.grid_new.index.intersection(pd.date_range(win[0].floor("1h"), win[-1].ceil("1h"), freq="1h"))
    assert len(hrs) and np.allclose(rl.gridweather.data.loc[hrs, RLV.grid_new.columns].to_numpy(float), RLV.grid_new.loc[hrs].to_numpy(float), equal_nan=True)
    res = {"insample_app": float((pub - rl.models["tso_load"].predict(feats)).mean())}
    if "hold_app_load" in out:
        res["app"] = float((pub - out["hold_app_load"].predict(feats)).mean())
    if "hold_matched_load" in out:
        res["matched"] = float((pub - out["hold_matched_load"].predict(feats)).mean())
    if "matched" in out:
        res["insample_matched"] = float((pub - out["matched"]["tso_load"].predict(feats)).mean())
    out["resid"] = res
    info["resid"] = {k: round(v, 4) for k, v in res.items()}
    info["corr_window"] = [str(win[0]), str(win[-1])]
    RLV.info = info
    RLV.cache[key] = out
    return out


_orig_rl_train = ResidualLoad.train
_orig_rl_get_data = ResidualLoad.get_data


async def _rl_train(self, start, end):
    await _orig_rl_train(self, start, end)
    self.extra = None
    if self.models is not None and RLV.variant != "base":
        self.extra = await _extra_models(self, start, end)


async def rl_components(rl: ResidualLoad, start, end) -> pd.DataFrame:
    """Per slot after the cutoff: both model sets' wind/solar/load, the decayed corrections, the grid vintage, and the
    selected (variant) wind/solar/load. Rows without weather are NaN."""
    features = await rl.get_features(start, end)
    features = features.loc[features.index > RLV.cutoff]
    features = features[features.notna().all(axis=1)]
    out = pd.DataFrame(index=features.index)
    if len(features) == 0:
        return out
    for t, m in rl.models.items():
        out[f"base_{t}"] = np.asarray(m.predict(features))
    ex = getattr(rl, "extra", None) or {}
    if "matched" in ex:
        for t, m in ex["matched"].items():
            out[f"matched_{t}"] = np.asarray(m.predict(features))
    # the held-out load boosters' own forecast (stored only): the study corrected the model that made the residual,
    # here the correction goes onto the app model trained through D-1 (see README)
    for name in ("hold_app_load", "hold_matched_load"):
        if name in ex:
            out[f"{name[:-5]}_tso_load"] = np.asarray(ex[name].predict(features))
    decay = np.exp(-np.asarray((features.index - RLV.cutoff) / SLOT, dtype=float) / RL_DECAY_QH)
    out["decay"] = decay
    out["corr_app"] = ex["resid"]["app"] * decay if "app" in ex.get("resid", {}) else 0.0
    out["corr_matched"] = ex["resid"]["matched"] * decay if "matched" in ex.get("resid", {}) else 0.0
    out["k"] = slot_k(features.index)
    use_m = (out["k"].to_numpy() >= 2) if "matched" in ex else np.zeros(len(out), bool)
    for t in ResidualLoad.TARGETS:
        b = out[f"base_{t}"].to_numpy()
        out[f"sel_{t}"] = np.where(use_m, out[f"matched_{t}"].to_numpy(), b) if "matched" in ex else b
    if RLV.uses("loadcorr") or RLV.variant == "both":
        out["sel_tso_load"] += np.where(use_m, out["corr_matched"], out["corr_app"])
    out["use_matched"] = use_m.astype(float)
    return out


async def _rl_get_data(self, start, end):
    df = await _orig_rl_get_data(self, start, end)
    if df is None or RLV.variant == "base" or RLV.cutoff is None or pd.Timestamp(end) <= RLV.cutoff:
        return df
    # recompute only the rows after the cutoff (from 3 h before it, so the hourly weather interpolates as in df)
    sub_start = max(pd.Timestamp(start), RLV.cutoff - pd.Timedelta(hours=3))
    comp = await rl_components(self, sub_start.to_pydatetime(), end)
    published = await self.generation.get_data(sub_start.to_pydatetime(), end)
    rows = comp.index.difference(published.dropna().index).intersection(df.index)
    if len(rows) == 0:
        return df
    c = comp.loc[rows]
    base_rl = c["base_tso_load"] - c["base_tso_wind"] - c["base_tso_solar"]
    assert np.allclose(base_rl.to_numpy(), df.loc[rows, ResidualLoad.FEATURE].to_numpy(), atol=1e-9), "recomputed base RL != app RL"
    df = df.copy()
    df.loc[rows, ResidualLoad.FEATURE] = c["sel_tso_load"] - c["sel_tso_wind"] - c["sel_tso_solar"]
    return df


ResidualLoad.train = _rl_train
ResidualLoad.get_data = _rl_get_data


# ---------------------------------------------------------------------------------------------------- vintages
def load_vintages() -> dict:
    """zone -> {k: hourly DataFrame with the store's column names}; 'GRID' for the dense grid"""
    out = {}
    for zone, f in ZONE_FILE.items():
        frames = {}
        p23 = pd.read_parquet(f"{SP}/prev_all/prev_{f}.parquet")
        for k in (2, 3):
            frames[k] = p23[[c for c in p23.columns if c.endswith(f"_d{k}")]].rename(columns=lambda c: c.rsplit("_", 1)[0])
        d1 = []
        if zone == "DE_LU":
            p = pd.read_parquet(f"{SP}/prevruns_DE.parquet")
            d1.append(p[[c for c in p.columns if c.endswith("_d1")]])
        if os.path.exists(f"{SP}/prev_d1/prev_d1_{f}.parquet"):
            d1.append(pd.read_parquet(f"{SP}/prev_d1/prev_d1_{f}.parquet"))
        frames[1] = (pd.concat(d1, axis=1).rename(columns=lambda c: c.rsplit("_", 1)[0]) if d1
                     else pd.DataFrame(index=frames[2].index))
        if FRESH:   # local values first, the fresh fetch fills the rest (they agree on the overlap, see fetch_fresh.py)
            fr = pd.read_parquet(f"{FRESH_DIR}/prev_{f}.parquet")
            for k in (1, 2, 3):
                frames[k] = frames[k].combine_first(fr[[c for c in fr.columns if c.endswith(f"_d{k}")]].rename(columns=lambda c: c.rsplit("_", 1)[0]))
        out[zone] = frames
    dense = pd.read_parquet(f"{SP}/dense/prev.parquet")
    if FRESH:
        dense = dense.combine_first(pd.read_parquet(f"{FRESH_DIR}/grid_prev.parquet"))
    names = {"w120": "wind", "sw": "radiation", "t2": "temp"}
    out["GRID"] = {k: dense[[c for c in dense.columns if c.endswith(f"_d{k}") and c.split("_")[0] in names]]
                   .rename(columns=lambda c: f"{names[c.split('_')[0]]}_{c.split('_')[1]}").astype(float) for k in (1, 2, 3)}
    return out


def check_grid_points(region):
    pts = json.load(open(f"{SP}/dense/points.json"))["points"]
    mine = list(zip(region.generation_latitudes + region.generation_offshore_latitudes,
                    region.generation_longitudes + region.generation_offshore_longitudes))
    assert [tuple(map(float, p)) for p in pts] == [tuple(map(float, p)) for p in mine], "dense grid order != region"


def wanted_k(index: pd.DatetimeIndex, gate: pd.Timestamp, policy: str, latency: pd.Timedelta, day_m1: date) -> np.ndarray:
    """Vintage (previous_day k) per timestamp. previous_dayk = forecast made 24h*k before valid time."""
    if policy == "strict":
        # the run made 24h*k before t must have been out at the gate: t - 24h*k <= gate - latency
        lead = np.asarray((index - gate + latency) / pd.Timedelta(hours=24), dtype=float)
        return np.maximum(1, np.ceil(lead).astype(int))
    local = index.tz_convert(TZ).date
    return np.maximum(1, np.array([(d - day_m1).days for d in local]))


def to_15min(h: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for c in h.columns:
        s = h[c].astype(float).dropna()
        if c.startswith("irradiance_"):
            s.index = s.index + IRRADIANCE_SHIFT
        cols[c] = s.reindex(s.index.union(index)).interpolate("time", limit_area="inside").reindex(index)
    return pd.DataFrame(cols, index=index).ffill(limit=4).bfill(limit=4)


def vintage_frame(frames: dict, columns, index, ks, quarter: bool) -> tuple[pd.DataFrame, dict]:
    """Values per (slot, column) from vintage ks[slot]; a missing value falls back to the next older vintage."""
    lo, hi = index[0] - pd.Timedelta(hours=3), index[-1] + pd.Timedelta(hours=3)
    per_k = {}
    for k in range(1, MAX_K + 1):
        h = frames[k].loc[lo:hi].reindex(columns=columns)
        per_k[k] = to_15min(h, index) if quarter else h.reindex(index)
    arr = np.full((len(index), len(columns)), np.nan)
    used = np.zeros((len(index), len(columns)), dtype=int)
    for k in range(1, MAX_K + 1):
        sel = (np.minimum(ks, MAX_K) <= k)[:, None] & np.isnan(arr)
        vals = per_k[k].to_numpy(dtype=float)
        take = sel & ~np.isnan(vals)
        arr[take] = vals[take]
        used[take] = k
    out = pd.DataFrame(arr, index=index, columns=columns)
    stats = {"wanted_k_gt_max": int((ks > MAX_K).sum()),
             "fallback_cells": int((used > np.minimum(ks, MAX_K)[:, None]).sum()),
             "missing_cells": int(out.isna().to_numpy().sum()), "cells": int(out.size),
             "k_hist": {int(k): int((used == k).sum()) for k in range(1, MAX_K + 1)}}
    return out, stats


# ---------------------------------------------------------------------------------------------------- one day
def day_index(d: date) -> pd.DatetimeIndex:
    s = pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=TZ))
    e = pd.Timestamp(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TZ))
    return pd.date_range(s, e, freq="15min", inclusive="left").tz_convert("UTC")


class Harness:
    def __init__(self, args):
        self.args = args
        self.vint = load_vintages()

    async def setup(self):
        await api.base_models.ensure_loaded()
        self.bases = api.base_models.base
        assert sorted(n.value for n in self.bases) == sorted(["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]), self.bases.keys()
        self.de = self.bases[PriceRegionName.DE]
        check_grid_points(self.de.region)
        # pristine copies (read-only)
        self.orig = {}
        for name, p in self.bases.items():
            for s in (p.pricestore, p.gasstore, p.weatherstore):
                self.orig[id(s)] = s.data
        self.rl = self.de.residual_load
        for s in (self.rl.gridweather, self.rl.generation):
            self.orig[id(s)] = s.data
        self.actual = self.de.pricestore.data["price"].copy()
        g = self.rl.generation.data
        self.rl_true = (g.tso_load - g.tso_wind - g.tso_solar)

    def stage_day(self, d: date):
        a = self.args
        gate = pd.Timestamp(datetime.combine(d - timedelta(days=1), datetime.min.time(), tzinfo=TZ).replace(hour=12)).tz_convert("UTC")
        cutoff = day_index(d)[0] - SLOT
        last_needed = day_index(d + timedelta(days=2))[-1]        # end of D+2
        info = {"gate": str(gate), "cutoff": str(cutoff), "weather": {}}
        for name, p in self.bases.items():
            for s in (p.pricestore, p.gasstore):
                s.data = self.orig[id(s)].loc[:cutoff] if len(self.orig[id(s)]) else self.orig[id(s)]
                s.known_source_horizon = s.source_horizon_revalitation_ts = None
            p.set_horizon_cutoff(cutoff.to_pydatetime())
            # weather: archive up to just before the gate, vintages from the gate to the end of D+2, nothing after
            w = p.weatherstore
            idx = pd.date_range(gate, last_needed, freq="15min")
            ks = wanted_k(idx, gate, a.vintage, pd.Timedelta(hours=a.latency), d - timedelta(days=1))
            new, st = vintage_frame(self.vint[p.region.bidding_zone_entsoe], list(self.orig[id(w)].columns), idx, ks, True)
            w.data = pd.concat([self.orig[id(w)].loc[:gate - SLOT], new])
            w.known_source_horizon = w.source_horizon_revalitation_ts = None
            w.last_updated = datetime.now(timezone.utc)
            info["weather"][p.region.bidding_zone_entsoe] = st
            if name == PriceRegionName.DE:
                self.de_weather_new = new
                self.de_ks = ks
        gw, gen = self.rl.gridweather, self.rl.generation
        gen.data = self.orig[id(gen)].loc[:cutoff]
        gen.known_source_horizon = gen.source_horizon_revalitation_ts = None
        hidx = pd.date_range(gate.ceil("1h"), last_needed.floor("1h"), freq="1h")
        hks = wanted_k(hidx, gate, a.vintage, pd.Timedelta(hours=a.latency), d - timedelta(days=1))
        new, st = vintage_frame(self.vint["GRID"], list(self.orig[id(gw)].columns), hidx, hks, False)
        gw.data = pd.concat([self.orig[id(gw)].loc[:gate.ceil("1h") - pd.Timedelta(hours=1)], new])
        gw.known_source_horizon = gw.source_horizon_revalitation_ts = None
        gw.last_updated = datetime.now(timezone.utc)
        info["weather"]["GRID"] = st
        self.grid_new = new
        # residual-load variant state for this day
        RLV.cutoff, RLV.gate, RLV.latency, RLV.policy = cutoff, gate, pd.Timedelta(hours=a.latency), a.vintage
        RLV.day_m1, RLV.grid_new, RLV.cache, RLV.info = d - timedelta(days=1), new, {}, {}
        # caches keyed on store state
        self.de.europe_weather.cache.clear()
        if api.timestral.load is not None:
            api.timestral.load.cached = None
        return gate, cutoff, info

    async def assert_no_leak(self, d, cutoff, stage2):
        D = day_index(d)
        for name, p in self.bases.items():
            assert p.pricestore.get_last_known() == cutoff, (name, p.pricestore.get_last_known(), cutoff)
            assert p.pricestore.data.index.max() <= cutoff
            if len(p.gasstore.data):
                assert p.gasstore.data.index.max() <= cutoff, (name, p.gasstore.data.index.max())
        assert self.rl.generation.data.index.max() <= cutoff
        assert (await self.rl.generation.get_data(D[0], D[-1])).index.max() <= cutoff  # get_data clamps start to the cutoff
        for p in list(self.bases.values()) + [stage2]:
            df = await p.prepare_dataframe(D[0], D[-1])
            assert df.loc[D[0]:D[-1], "price"].isna().all(), p.region.bidding_zone_entsoe
        # the overwritten weather is what the model reads
        df = await self.de.prepare_dataframe(D[0], D[-1])
        got = df.loc[D, [c for c in self.de_weather_new.columns if c.startswith("wind_")]]
        assert np.allclose(got.to_numpy(), self.de_weather_new.loc[D, got.columns].to_numpy(), equal_nan=True)

    async def run_day(self, d: date):
        a = self.args
        fn = f"{a.out}/{d}.npz"
        if os.path.exists(fn):
            log.info(f"{d}: exists, skipped")
            return
        t0 = time.time()
        gate, cutoff, info = self.stage_day(d)
        start, end = cutoff - pd.Timedelta(days=api.TRAINING_DAYS), cutoff
        # 1) stage 1 (all base models, incl. DE's residual load), as BaseModelCoordinator._do_update
        await asyncio.gather(*(p.train(start.to_pydatetime(), end.to_pydatetime()) for p in self.bases.values()))
        t1 = time.time()
        # 2) stage 2 for DE, built as RegionPriceManager builds it
        stage2 = pp.PricePredictor(self.de.region, storage_dir=api.EPEXPREDICTOR_DATADIR, cross_predictors=api.base_models.get_cross_predictors())
        stage2.use_datastores_from(self.de)
        await stage2.train(start.to_pydatetime(), end.to_pydatetime())
        t2 = time.time()
        await self.assert_no_leak(d, cutoff, stage2)
        # 3) predict from an earlier start (gas ffill) and slice; then the blend as RegionPriceManager does it
        D = day_index(d)
        pstart, pend = cutoff - pd.Timedelta(days=14) + SLOT, day_index(d + timedelta(days=1))[-1]
        newprices = await stage2.predict(pstart.to_pydatetime(), pend.to_pydatetime())
        neweval = await stage2.predict(pstart.to_pydatetime(), pend.to_pydatetime(), fill_known=False)
        trees = neweval["price"].copy()
        others = {"market": api.market_route} if api.market_route is not None else None
        # spy: the frames Timestral and the market route read during the blend
        seen, orig_inputs = [], api.timestral.inputs

        async def spy_inputs(predictor, start, end):
            df = await orig_inputs(predictor, start, end)
            if predictor is self.de and df is not None and ResidualLoad.FEATURE in df.columns:
                seen.append(df[ResidualLoad.FEATURE].copy())
            return df
        api.timestral.inputs = spy_inputs
        try:
            blended = await api.timestral.blend(self.de, [newprices, neweval], others)
        finally:
            del api.timestral.inputs   # back to the class method
        fc = dict(api.timestral.last_forecasts)
        core = api.market_route.last_core if api.market_route is not None else None
        t3 = time.time()

        def onD(s):
            return np.full(len(D), np.nan) if s is None else s.reindex(D).to_numpy(dtype=float)

        res = {"idx": D.asi8, "trees": onD(trees), "timestral": onD(fc.get("timestral")), "market": onD(fc.get("market")),
               "core": onD(core), "issued": onD(newprices["price"]), "issued_eval": onD(neweval["price"]),
               "actual": onD(self.actual)}
        # the residual load: published-or-trees (what the tree models read) and the Timestral-mixed one (what Timestral
        # and the market route read), plus Timestral's own figure and the published TSO figure (truth)
        # asked a few hours wider: the hourly grid weather is interpolated only between known hours
        rl_trees = (await self.de.residual_load.get_data(D[0] - pd.Timedelta(hours=2), D[-1] + pd.Timedelta(hours=3)))[ResidualLoad.FEATURE]
        # wide as well: on a range ending at D's last slot the tree RL of the last ~3 slots is NaN and the mix falls back
        # to Timestral alone there, which is not what Timestral/market read (their range runs well past D; spy check)
        mixed = await api.timestral.inputs(self.de, D[0] - SLOT * 4, D[-1] + pd.Timedelta(hours=3))
        rl_ts = await api.timestral.load.residual_load(self.de.residual_load) if api.timestral.load is not None else None
        assert np.isfinite(onD(rl_trees)).all(), "residual load (trees) has gaps on D"
        res |= {"rl_trees": onD(rl_trees), "rl_mixed": onD(mixed[ResidualLoad.FEATURE]), "rl_timestral": onD(rl_ts),
                "rl_true": onD(self.rl_true), "weather_k_DE": self.de_ks[np.searchsorted(self.de_weather_new.index, D)]}
        # residual-load components on D (both model sets, corrections, vintage, selection) and the published ones
        comp = await rl_components(self.de.residual_load, (D[0] - pd.Timedelta(hours=2)).to_pydatetime(), (D[-1] + pd.Timedelta(hours=3)).to_pydatetime())
        comp = comp.reindex(D)
        for c in comp.columns:
            res[f"rlc_{c}"] = comp[c].to_numpy(dtype=float)
        g = self.orig[id(self.rl.generation)]
        for t in ResidualLoad.TARGETS:
            res[f"true_{t}"] = onD(g[t])
        sel_rl = res["rlc_sel_tso_load"] - res["rlc_sel_tso_wind"] - res["rlc_sel_tso_solar"]
        base_rl = res["rlc_base_tso_load"] - res["rlc_base_tso_wind"] - res["rlc_base_tso_solar"]
        res["rl_base_model"] = base_rl
        # S5: every consumer reads the variant series
        chk = {"variant": a.rl_variant}
        assert np.allclose(res["rl_trees"], sel_rl, atol=1e-9), "get_data != selected components"
        # asked a few hours wider, as rl_trees: the hourly weather is interpolated only between known hours
        wide = ((D[0] - pd.Timedelta(hours=2)).to_pydatetime(), (D[-1] + pd.Timedelta(hours=3)).to_pydatetime())
        df_de = await self.de.prepare_dataframe(*wide)
        df_s2 = await stage2.prepare_dataframe(*wide)
        assert np.allclose(df_de.loc[D, ResidualLoad.FEATURE].to_numpy(float), sel_rl, atol=1e-9), "DE base features != variant RL"
        chk["stage2_minus_base_maxabs"] = float(np.max(np.abs(df_s2.loc[D, ResidualLoad.FEATURE].to_numpy(float) - sel_rl)))
        assert chk["stage2_minus_base_maxabs"] < 1e-6, f"stage-2 features != variant RL ({chk['stage2_minus_base_maxabs']})"
        if rl_ts is not None:
            w = pd.Series(TimestralLoad.weights(len(rl_ts)), index=rl_ts.index).reindex(D).to_numpy()
            expect = w * res["rl_timestral"] + (1 - w) * sel_rl
            chk["mixed_maxabs"] = float(np.max(np.abs(res["rl_mixed"] - expect)))
            assert chk["mixed_maxabs"] < 1e-9, "Timestral mix != w*Timestral + (1-w)*variant"
        assert len(seen) >= 1, "spy saw no Timestral/market inputs"
        chk["spy_calls"] = len(seen)
        chk["spy_maxabs"] = float(max(np.nanmax(np.abs(sv.reindex(D).to_numpy(float) - res["rl_mixed"])) for sv in seen))
        assert chk["spy_maxabs"] < 1e-9, "Timestral/market inputs != mixed variant RL"
        chk["variant_minus_base_maxabs"] = float(np.max(np.abs(sel_rl - base_rl)))
        if a.rl_variant != "base":
            assert chk["variant_minus_base_maxabs"] > 0, "variant RL identical to base"
        info["rl_check"] = chk
        info["rl_variant"] = a.rl_variant
        info["rl_extra"] = RLV.info
        # checks
        assert len(D) in (92, 96, 100)
        assert np.isfinite(res["trees"]).all(), "tree forecast has gaps on D"
        if blended and np.isfinite(res["timestral"]).all() and np.isfinite(res["market"]).all():
            assert np.allclose(res["issued"], (res["trees"] + res["timestral"] + res["market"]) / 3), "blend != mean of thirds"
        info |= {"blended": bool(blended), "forecasters": sorted(fc), "train_rows_stage1_DE": int(len(self.de.traindata)),
                 "train_first_DE": str(self.de.traindata.index.min()), "train_last_DE": str(self.de.traindata.index.max()),
                 "train_rows_stage2": int(len(stage2.traindata)), "vintage": a.vintage, "latency_h": a.latency,
                 "secs": {"stage1": round(t1 - t0, 1), "stage2": round(t2 - t1, 1), "predict_blend": round(t3 - t2, 1),
                          "total": round(time.time() - t0, 1)}}
        os.makedirs(a.out, exist_ok=True)
        np.savez(fn + ".tmp.npz", meta=json.dumps(info), **res)
        os.replace(fn + ".tmp.npz", fn)
        mae = {k: round(float(np.nanmean(np.abs(res[k] - res["actual"]))), 3) for k in ("trees", "timestral", "market", "issued")}
        log.info(f"{d}: {len(D)} slots | MAE ct/kWh {mae} | RL MAE GW trees {np.nanmean(np.abs(res['rl_trees'] - res['rl_true'])):.2f} "
                 f"mixed {np.nanmean(np.abs(res['rl_mixed'] - res['rl_true'])):.2f} | secs {info['secs']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days")
    ap.add_argument("--start", default="2026-09-08")
    ap.add_argument("--end", default="2026-11-08")
    ap.add_argument("--vintage", choices=("strict", "d1"), default="strict")
    ap.add_argument("--latency", type=float, default=6.0, help="hours between a model run's nominal time and its availability")
    ap.add_argument("--out")
    ap.add_argument("--rl-variant", choices=("base", "loadcorr", "matched", "both"), default="base",
                    help="residual-load variant (see RLState); non-base variants write to out_<vintage>[_lat]_rl<variant>")
    ap.add_argument("--torch-threads", type=int, default=0, help="0 = torch default; lower it on a contended machine")
    a = ap.parse_args()
    if a.torch_threads > 0:
        import torch
        torch.set_num_threads(a.torch_threads)  # applies to the asyncio worker threads that run Timestral as well
    a.out = a.out or (f"{SP}/faithful/out_{a.vintage}" + ("" if a.latency == 6.0 else f"_lat{a.latency:g}")
                      + ("" if a.rl_variant == "base" else f"_rl{a.rl_variant}"))
    RLV.variant = a.rl_variant
    if a.rl_variant in ("matched", "both"):
        RLV.d2 = load_d2_history()
    days = ([date.fromisoformat(x) for x in a.days.split(",")] if a.days else
            [date.fromisoformat(a.start) + timedelta(days=i) for i in range((date.fromisoformat(a.end) - date.fromisoformat(a.start)).days + 1)])
    assert len(day_index(date(2026, 10, 25))) == 100 and len(day_index(date(2026, 3, 29))) == 92
    h = Harness(a)
    t = time.time()
    await h.setup()
    log.info(f"setup {time.time() - t:.0f}s; {len(days)} days -> {a.out}")
    failed = []
    for d in days:
        try:
            await h.run_day(d)
        except Exception as e:  # a failed day (including a failed leak check) is logged and skipped, not fatal
            log.exception(f"{d}: FAILED ({type(e).__name__}: {e})")
            failed.append(str(d))
    if failed:
        log.warning(f"failed days ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
