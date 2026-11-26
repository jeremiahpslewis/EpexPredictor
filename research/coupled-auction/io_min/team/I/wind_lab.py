"""Wind booster screen (pre-registered in WIND_PREREG.md). Offline, 1 process, 1 thread.

usage: tsvenv/bin/python wind_lab.py SP build
       tsvenv/bin/python wind_lab.py SP screen V0,V1,... [--cadence weekly|daily] [--days 61|fresh18|check|all]
Results: SP/team/I/windlab/pred_<variant>_<cadence>.parquet (per eval slot) and a printed summary.
"""
import glob, json, os, sys, time
from datetime import timedelta
import numpy as np, pandas as pd

SP = sys.argv[1]
OUT = f"{SP}/team/I/windlab"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/home/user/EpexPredictor")
os.environ.setdefault("EPEXPREDICTOR_DATADIR", f"{OUT}/nostore")
from predictor.model.priceregion import PriceRegionName  # noqa: E402
from predictor.model.residualload import generation_features, RATED_WIND_SPEED  # noqa: E402
from predictor.model.auxdatastore import AuxDataStore  # noqa: E402
import lightgbm as lgb  # noqa: E402

REGION = PriceRegionName.DE.to_region()
NON = len(REGION.generation_latitudes); NPT = NON + len(REGION.generation_offshore_latitudes)
START = pd.Timestamp("2025-10-01", tz="UTC"); END = pd.Timestamp("2026-11-28", tz="UTC")
TZ = "Europe/Berlin"
PARAMS = {"force_col_wise": True, "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31, "num_threads": 1}
ROUNDS = 400
CHECK_DAYS = ["2026-09-10", "2026-09-24", "2026-10-08", "2026-10-22", "2026-11-05", "2026-11-12", "2026-11-19", "2026-11-26"]


def build():
    names = {"w120": "wind", "sw": "radiation", "t2": "temp"}
    h = pd.read_parquet(f"{SP}/dense/hist.parquet").combine_first(pd.read_parquet(f"{SP}/fresh_data/grid_prev.parquet"))
    cols = [c for c in h.columns if c.endswith("_d2") and c.split("_")[0] in names]
    d2 = h[cols].rename(columns=lambda c: f"{names[c.split('_')[0]]}_{c.split('_')[1]}").astype(float).sort_index()
    app = pd.read_json(f"{SP}/fresh_data/stores/gridweather_v1_DE_LU.json.gz")
    app.index = pd.DatetimeIndex(app.index).tz_localize("UTC")
    aux = AuxDataStore(REGION)._compute_data(START.to_pydatetime(), END.to_pydatetime())
    aux = aux.loc[START:END]
    gen = pd.read_json(f"{SP}/fresh_data/stores/generation_v1_DE_LU.json.gz")
    gen.index = pd.DatetimeIndex(gen.index).tz_localize("UTC")
    F2 = generation_features(REGION, d2.loc[START - pd.Timedelta(hours=2):], aux)
    FA = generation_features(REGION, app.loc[START - pd.Timedelta(hours=2):], aux)
    # raw speeds for the physical variants (km/h), interpolated as generation_features does
    w = d2[[f"wind_{i}" for i in range(NPT)]]
    w = w.reindex(w.index.union(aux.index)).interpolate(method="time", limit=3, limit_area="inside").reindex(aux.index)
    F2.astype("float32").to_parquet(f"{OUT}/F_d2.parquet"); FA.astype("float32").to_parquet(f"{OUT}/F_app.parquet")
    w.astype("float32").to_parquet(f"{OUT}/speed_d2.parquet")
    gen.loc[START:END, ["tso_wind"]].to_parquet(f"{OUT}/target.parquet")
    # eval slots: k >= 2 in the stored matched runs, with the stored matched wind for the sanity check
    rows = []
    for d, st in (("out_strict_rlmatched", "61"), ("out_fresh_strict_rlmatched", "fresh18")):
        for f in sorted(glob.glob(f"{SP}/faithful/{d}/*.npz")):
            day = os.path.basename(f)[:-4]
            if st == "61" and not ("2026-09-08" <= day <= "2026-11-07"):
                continue
            z = np.load(f)
            rows.append(pd.DataFrame({"t": pd.to_datetime(z["idx"], unit="us", utc=True), "day": day, "set": st, "k": z["rlc_k"],
                                      "stored_matched": z["rlc_matched_tso_wind"], "stored_true": z["true_tso_wind"]}))
    E = pd.concat(rows, ignore_index=True)
    E = E[E.k >= 2].set_index("t")
    E.to_parquet(f"{OUT}/eval.parquet")
    print("built", F2.shape, FA.shape, len(E), E.groupby("set").day.nunique().to_dict())


def capacity(index):
    """installed wind (onshore + offshore) GW, energy-charts yearly values taken as end-of-year anchors (2026 = at
    last update, ~now), linear in between and extrapolated with the last slope"""
    j = json.load(open(f"{SP}/ext/ecz/inst_de.json"))
    yrs = j["time"]; tot = None
    for p in j["production_types"]:
        if p["name"] in ("Wind onshore", "Wind offshore"):
            v = np.array([np.nan if x is None else x for x in p["data"]])
            tot = v if tot is None else tot + v
    anchors = {pd.Timestamp(f"{y}-12-31", tz="UTC"): tot[i] for i, y in enumerate(yrs) if np.isfinite(tot[i]) and int(y) >= 2023}
    anchors[pd.Timestamp("2026-11-26", tz="UTC")] = anchors.pop(pd.Timestamp("2026-12-31", tz="UTC"))
    x = np.array([t.value for t in anchors]); y = np.array(list(anchors.values()))
    xi = index.asi8 * (1 if index.asi8[0] > 1e17 else 1000)
    out = np.interp(xi, x, y)
    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.where(xi > x[-1], y[-1] + slope * (xi - x[-1]), out)


def physical(F, w):
    """V4: smooth power curve per point (km/h: cut-in 10.8, rated 43.2, cut-out 90), replacing the cubed/capped speed"""
    a = w.to_numpy(float)
    x = np.clip((a - 10.8) / (43.2 - 10.8), 0, 1)
    pc = np.where(a > 90, 0.0, x * x * (3 - 2 * x))
    G = F.drop(columns=[c for c in F.columns if c.startswith("wind_power_")]).copy()
    for i in range(NPT):
        G[f"pc_{i}"] = pc[:, i]
    on = np.arange(NPT) < NON
    G["pc_onshore"] = pc[:, on].mean(1); G["pc_offshore"] = pc[:, ~on].mean(1)
    G["speed_onshore"] = a[:, on].mean(1); G["speed_offshore"] = a[:, ~on].mean(1); G["speed_std"] = a[:, on].std(1)
    return G


def aggregates(F, w):
    """V5: V0 features + capped-cube mean, onshore p90 speed, 3 h change of mean onshore speed"""
    a = w.to_numpy(float); on = np.arange(NPT) < NON
    G = F.copy()
    G["cube_capped"] = (np.clip(a[:, on], 0, RATED_WIND_SPEED) ** 3).mean(1) / RATED_WIND_SPEED ** 3
    G["p90_onshore"] = np.nanpercentile(a[:, on], 90, axis=1)
    m = pd.Series(a[:, on].mean(1), index=F.index)
    G["dspeed_3h"] = (m - m.shift(12)).to_numpy()
    return G


def load():
    F2 = pd.read_parquet(f"{OUT}/F_d2.parquet").astype(float); FA = pd.read_parquet(f"{OUT}/F_app.parquet").astype(float)
    w = pd.read_parquet(f"{OUT}/speed_d2.parquet").astype(float)
    y = pd.read_parquet(f"{OUT}/target.parquet")["tso_wind"].reindex(F2.index)
    E = pd.read_parquet(f"{OUT}/eval.parquet")
    return F2, FA, w, y, E


def fit(X, y, obj="regression", weight=None):
    p = dict(PARAMS, objective=obj)
    return lgb.train(p, lgb.Dataset(X, label=y, weight=weight), num_boost_round=ROUNDS)


def cutoff_of(day: str, cadence: str) -> pd.Timestamp:
    d = pd.Timestamp(day)
    if cadence == "weekly":
        d = d - pd.Timedelta(days=d.weekday())         # the Monday on or before D
    return (pd.Timestamp(d.date(), tz=TZ) - pd.Timedelta(minutes=15)).tz_convert("UTC")   # D-1 23:45 local


def run(variants, cadence, which):
    F2, FA, w, y, E = load()
    if which == "check":
        E = E[E.day.isin(CHECK_DAYS)]
    elif which in ("61", "fresh18"):
        E = E[E.set == which]
    cap = pd.Series(capacity(F2.index), index=F2.index)
    feats = {"V0": F2, "V1": F2, "V2": F2, "V3": F2, "V4": physical(F2, w), "V5": aggregates(F2, w), "V6": F2}
    feats["V7"] = feats["V5"]   # post hoc: V1 (capacity-factor target) + V5 features, the two that beat V0 on both windows
    groups = E.groupby(E.day.map(lambda d: cutoff_of(d, cadence)))
    for v in variants:
        t0 = time.time(); preds = []
        X = feats[v]
        for cut, ev in groups:
            ok = X.index <= cut
            ok &= X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
            Xt, yt = X[ok], y[ok]
            if v in ("V1", "V7"):
                m = fit(Xt, yt / cap[ok]); p = m.predict(X.loc[ev.index]) * cap.loc[ev.index].to_numpy()
            elif v == "V2":
                p = fit(Xt, yt, "regression_l1").predict(X.loc[ev.index])
            elif v == "V3":
                age = (cut - Xt.index) / pd.Timedelta(days=1)
                p = fit(Xt, yt, weight=0.5 ** (np.asarray(age, float) / 120)).predict(X.loc[ev.index])
            elif v == "V6":
                okA = (FA.index <= cut) & FA.notna().all(axis=1).to_numpy() & y.reindex(FA.index).notna().to_numpy()
                mA = fit(FA[okA], y.reindex(FA.index)[okA])
                p = 0.5 * fit(Xt, yt).predict(X.loc[ev.index]) + 0.5 * mA.predict(X.loc[ev.index])
            else:
                p = fit(Xt, yt).predict(X.loc[ev.index])
            preds.append(pd.DataFrame({"day": ev.day, "set": ev.set, "pred": p, "true": y.loc[ev.index].to_numpy(),
                                       "stored_matched": ev.stored_matched, "stored_true": ev.stored_true}, index=ev.index))
        P = pd.concat(preds)
        P.to_parquet(f"{OUT}/pred_{v}_{cadence}_{which}.parquet")
        s = P.assign(ae=(P.pred - P.true).abs(), aes=(P.stored_matched - P.stored_true).abs()).groupby("set")[["ae", "aes"]].mean()
        print(f"{v} {cadence} {which}: fits {len(groups)} secs {time.time() - t0:.0f} | " +
              " ".join(f"{k}: MAE {r.ae:.3f} (stored matched {r.aes:.3f})" for k, r in s.iterrows()), flush=True)


if __name__ == "__main__":
    if sys.argv[2] == "build":
        build()
    else:
        cad = sys.argv[sys.argv.index("--cadence") + 1] if "--cadence" in sys.argv else "weekly"
        which = sys.argv[sys.argv.index("--days") + 1] if "--days" in sys.argv else "all"
        run(sys.argv[3].split(","), cad, which)
