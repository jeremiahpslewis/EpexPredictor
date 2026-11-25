"""2. What the trees learn (EXPECTATIONS.md S1, S2, L1). Single-stage LightGBM for Germany in the style of
research/market-model/exp/live_test3_app.py (frame_DE features + TSO residual load 'rl', 540-day window, default
LightGBM, prediction = mean of a raw-target and a gas-ratio-target model).
Part S: models trained on data before 2026-09-01.
  - finite-difference dp/dRL (central, +-2 GW) of the full model and of an rl+calendar+gas model on training hours;
  - the cf.py / slope11 net-slope estimator applied to out-of-fold tree-fitted prices (8 contiguous folds);
  - the structural model's per-MTU slope (slope_train_old.pkl) on the same weighting and bands.
Part L: ablations retrained every holdout day (61 days), rl = deployed pre-gate forecast (gen_lwA_autumn rl_pes),
  weather = archive (an information advantage shared by all variants), gas = value one day before D.
Writes trees_slope.md and SP/distill/trees_ablate.npz."""
import sys, os
import numpy as np, pandas as pd
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import *
PANEL = "/tmp/claude-0/-home-user-EpexPredictor/1711225d-3f79-42cf-ba0d-e4080065c4b2/scratchpad/io/panel.parquet"
L = 540 * 96; CUT0 = pd.Timestamp("2026-09-01", tz="Europe/Berlin").tz_convert("UTC")
G = {}
def frame():
    F = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
    fc = pd.concat([pd.read_parquet(f"{SP}/ecfc/{p}.parquet") for p in ("wind_onshore", "wind_offshore", "solar", "load")], axis=1).reindex(F.index) / 1000
    F["rl"] = fc["load"] - fc["wind_onshore"] - fc["wind_offshore"] - fc["solar"]; return F
BASE = None
def cols(variant, F):
    allc = [c for c in F.columns if c != "price"]
    cal = ["holiday", "weekday", "sunelevation", "azimuth", "sr_influence", "ss_influence", "slot"]
    return {"full": allc, "no_ts": [c for c in allc if c != "ts"], "rlcalgas": cal + ["gasprice", "rl", "ts"],
            "merit": cal + ["gasprice", "rl"]}[variant.split("|")[0]]
def target_kind(variant):
    k = variant.split("|")[1] if "|" in variant else "blend"; return "blend" if k == "frozen" else k
def fit1(X, y, w=None):
    import lightgbm as lgb
    return lgb.train({"force_col_wise": True, "verbosity": -1, "num_threads": 1}, lgb.Dataset(X, label=y, weight=w))
def fit_models(F, c, i0, i1, kind, rows=None):
    """train on rows [i0, i1) (or the given row index array); returns predict(X) closure"""
    Xtr = F[c].to_numpy(float)[i0:i1] if rows is None else F[c].to_numpy(float)[rows]
    ytr = F["price"].to_numpy(float)[i0:i1] if rows is None else F["price"].to_numpy(float)[rows]
    ok = ~(np.isnan(Xtr).any(1) | np.isnan(ytr)); g = c.index("gasprice"); ms = []
    if kind in ("blend", "raw"): ms.append(("raw", fit1(Xtr[ok], ytr[ok])))
    if kind in ("blend", "ratio"): ms.append(("ratio", fit1(Xtr[ok], ytr[ok] / Xtr[ok][:, g], w=Xtr[ok][:, g] ** 2)))
    def pred(X):
        return np.mean([m.predict(X) * (X[:, g] if k == "ratio" else 1.0) for k, m in ms], axis=0)
    return pred
def init():
    os.environ["OMP_NUM_THREADS"] = "1"; G["F"] = frame()
def ablate_day(args):
    variant, j = args; F = G["F"]; c = cols(variant, F); kind = target_kind(variant)
    I = np.load(f"{O}/inputs_autumn.npz", allow_pickle=True); cut = pd.DatetimeIndex(I["cutoffs"]).tz_localize("UTC")
    rl = np.load(f"{O}/gen_lwA_autumn.npz", allow_pickle=True)["rl_pes"].astype(float)
    d0 = F.index.get_loc(cut[j]); i1 = d0 - 34 * 4 if "frozen" not in variant else F.index.get_loc(CUT0) - 34 * 4
    pred = fit_models(F, c, i1 - L, i1, kind)
    X = F[c].iloc[d0:d0 + 96].copy()
    if "rl" in c: X["rl"] = rl[j, :96]
    X["gasprice"] = F["gasprice"].iloc[d0 - 96]
    return variant, j, pred(X.to_numpy(float))

def oof_fold(args):
    variant, k, nf = args; F = G["F"]; c = cols(variant, F); kind = target_kind(variant)
    i1 = F.index.get_loc(CUT0); i0 = i1 - L; idx = np.arange(i0, i1); folds = np.array_split(idx, nf)
    te = folds[k]; tr = np.setdiff1d(idx, te)
    pred = fit_models(F, c, 0, 0, kind, rows=tr)
    X = F[c].to_numpy(float)
    out = {"te": te, "p": pred(X[te])}
    if k == 0:   # full-window model for finite differences on the last 365 training days
        pf = fit_models(F, c, i0, i1, kind); rows = np.arange(i1 - 365 * 96, i1); Xr = X[rows].copy(); ri = c.index("rl")
        Xp, Xm = Xr.copy(), Xr.copy(); Xp[:, ri] += 2; Xm[:, ri] -= 2
        out["fd_rows"] = rows; out["fd"] = (pf(Xp) - pf(Xm)) / 4 * 10; out["fit"] = pf(Xr)   # EUR/MWh per GW
    return variant, k, out

def cf_slope(p_hourly, Pn):
    """cf.py / slope11 TRAIN_END recipe: price and demands residualised on day and hour-month-weekend cells; coefficient
    of German demand in a regression with neighbour demand as control. Returns (slope, q05, q95) by level tercile."""
    dfe = pd.DataFrame({"p": p_hourly, "D": Pn.DDE.to_numpy(), "DN": Pn.DN.to_numpy(), "day": Pn.day.to_numpy(), "hour": Pn.hour.to_numpy(),
                        "month": Pn.index.month, "wk": (Pn.day.dt.dayofweek >= 5).astype(int)}, index=Pn.index).dropna()
    cell = dfe.hour.astype(str) + "_" + dfe.month.astype(str) + "_" + dfe.wk.astype(str)
    res = lambda s: (lambda r_: r_ - r_.groupby(cell).transform("mean"))(s - s.groupby(dfe.day).transform("mean"))
    Dr, pr, DNr = res(dfe.D), res(dfe.p), res(dfe.DN)
    lev = pd.qcut(dfe.p.groupby(dfe.day).transform("mean"), 3, labels=["low", "middle", "high"])
    rng_ = np.random.default_rng(0); EMP = {}
    for name, mk in [("all", np.ones(len(dfe), bool))] + [(l, (lev == l).to_numpy()) for l in ("low", "middle", "high")]:
        X_ = np.c_[Dr[mk], DNr[mk]]; y_ = pr[mk].to_numpy(); b_ = np.linalg.lstsq(X_, y_, rcond=None)[0][0]
        days_ = dfe.day[mk].to_numpy(); ud = np.unique(days_); ix = pd.Series(np.arange(mk.sum())).groupby(days_).apply(np.array).to_dict()
        bs = [np.linalg.lstsq(X_[ii], y_[ii], rcond=None)[0][0] for ii in (np.concatenate([ix[d] for d in rng_.choice(ud, len(ud))]) for _ in range(200))]
        EMP[name] = (float(b_), float(np.quantile(bs, .05)), float(np.quantile(bs, .95)))
    return EMP, Dr

if __name__ == "__main__":
    out = ["# What the trees learn: implied slopes and level ablations", ""]
    VS = ["full", "rlcalgas"]
    with get_context("spawn").Pool(4, initializer=init) as pool:
        res = pool.map(oof_fold, [(v, k, 8) for v in VS for k in range(8)], chunksize=1)
        AB = ["full", "full|raw", "full|ratio", "no_ts", "no_ts|raw", "rlcalgas", "merit|raw", "merit", "full|frozen"]
        ab = pool.map(ablate_day, [(v, j) for v in AB for j in range(5, 66)], chunksize=2)
    init(); F = G["F"]
    Pn = pd.read_parquet(PANEL); Pn = Pn[Pn.index < CUT0]
    oof = {}; fd = {}
    for v, k, o in res:
        s = oof.setdefault(v, pd.Series(np.nan, index=F.index)); s.iloc[o["te"]] = o["p"]
        if "fd" in o: fd[v] = (pd.Series(o["fd"], index=F.index[o["fd_rows"]]), pd.Series(o["fit"], index=F.index[o["fd_rows"]]))
    EMPa, Dr = cf_slope(Pn.p.to_numpy(), Pn)
    rows = {"observed price (empirical)": {k: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in EMPa.items()}}
    for v in VS:
        ph = (oof[v] * 10).groupby(oof[v].index.floor("h")).mean().reindex(Pn.index).to_numpy()
        e_, _ = cf_slope(ph, Pn); rows[f"OOF tree-fitted price, {v}"] = {k: f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" for k, x in e_.items()}
        rows[f"OOF tree MAE ct/kWh, {v}"] = {"all": f"{np.nanmean(np.abs(oof[v] - F.price)[oof[v].notna()]):.2f}"}
    out += ["## S1: cf.py net slope (EUR/MWh per GW of German demand; day and hour-month-weekend residualised; neighbour demand control; training hours 2025-10-01..2026-08-31)", "",
            pd.DataFrame(rows).T.to_markdown(), ""]
    # finite differences and structural slope by band, Dr^2 weighted and unweighted
    S = pd.read_pickle(f"{E11}/slope_train_old.pkl"); S = S[S.ok == True]
    ST = S.groupby("t").agg(dp=("dp", "mean"), p=("p", "first")); ST.index = pd.DatetimeIndex(ST.index)
    bands = [(-1e9, 0), (0, 40), (40, 120), (120, 200), (200, 1e9)]
    tab = {}
    w_h = (Dr ** 2)
    obs = F.price * 10
    for v in VS:
        s, fit = fd[v]; ob = obs.reindex(s.index); w = w_h.reindex(s.index.floor("h")).to_numpy()
        for lo, hi in bands:
            m = ((ob > lo) & (ob <= hi)).to_numpy(); mm = m & np.isfinite(w)
            tab.setdefault(f"tree finite diff, {v}", {})[f"{lo:g}..{hi:g}"] = f"{s[m].mean():.2f} (w {np.average(s[mm], weights=w[mm]):.2f})"
        mm = np.isfinite(w); tab[f"tree finite diff, {v}"]["all"] = f"{s.mean():.2f} (w {np.average(s[mm], weights=w[mm]):.2f})"
    ob = ST.p; w = w_h.reindex(ST.index.floor("h")).to_numpy()
    for lo, hi in bands:
        m = ((ob > lo) & (ob <= hi)).to_numpy(); mm = m & np.isfinite(w)
        tab.setdefault("structural dp/dD (slope_train_old)", {})[f"{lo:g}..{hi:g}"] = f"{ST.dp[m].mean():.2f} (w {np.average(ST.dp[mm], weights=w[mm]):.2f})"
    mm = np.isfinite(w); tab["structural dp/dD (slope_train_old)"]["all"] = f"{ST.dp.mean():.2f} (w {np.average(ST.dp[mm], weights=w[mm]):.2f})"
    tab["n tree QH"] = {f"{lo:g}..{hi:g}": int(((obs.reindex(fd['full'][0].index) > lo) & (obs.reindex(fd['full'][0].index) <= hi)).sum()) for lo, hi in bands}
    tab["n structural MTU"] = {f"{lo:g}..{hi:g}": int(((ST.p > lo) & (ST.p <= hi)).sum()) for lo, hi in bands}
    out += ["## S2: dp/dRL by observed price band (EUR/MWh per GW; mean, and Dr^2-weighted as in slope11 in brackets)", "",
            "Tree: central difference +-2 GW in the rl input of the model trained to 2026-09-01, on the last 365 training days, other inputs held.", "",
            pd.DataFrame(tab).T.to_markdown(), ""]
    # slope by fitted-price band (pre-gate-like conditioning)
    tab = {}
    for v in VS:
        s, fit = fd[v]; fp = fit * 10
        for lo, hi in bands:
            m = ((fp > lo) & (fp <= hi)).to_numpy(); tab.setdefault(v, {})[f"{lo:g}..{hi:g}"] = f"{s[m].mean():.2f} (n {m.sum()})"
    out += ["Tree finite difference by the tree's own fitted-price band:", "", pd.DataFrame(tab).T.to_markdown(), ""]
    np.savez(f"{SP}/distill/tree_slopes.npz", **{f"fd_{v}": fd[v][0].to_numpy() for v in VS}, **{f"fit_{v}": fd[v][1].to_numpy() for v in VS},
             idx=fd["full"][0].index.tz_convert(None).to_numpy().astype("datetime64[ns]"))
    # L: ablations on the holdout
    M = load_members(); a = M["a"]; P = {}
    for v, j, p in ab: P.setdefault(v, np.zeros_like(a))[j - 5] = p
    np.savez(f"{SP}/distill/trees_ablate.npz", **{k.replace("|", "_"): v for k, v in P.items()})
    rows = {v: {"MAE": day_mae(p, a).mean(), "bias": (p - a).mean(), "vs full": paired(p, P["full"], a)} for v, p in P.items()}
    rows["T (deployed two-stage, stored)"] = {"MAE": day_mae(M["T"], a).mean(), "bias": (M["T"] - a).mean(), "vs full": paired(M["T"], P["full"], a)}
    out += ["## L: ablations, single-stage tree retrained per holdout day (61 days), deployed pre-gate rl, archive weather", "",
            "variants: full = all frame features + rl; no_ts = without the time index; rlcalgas = calendar + gas + rl + ts; merit = calendar + gas + rl;",
            "|raw = price target only, |ratio = price/gas target only, default = mean of both (as deployed); |frozen = trained once to 2026-09-01.", "",
            pd.DataFrame(rows).T.to_markdown(floatfmt=".3f"), ""]
    open(os.path.dirname(os.path.abspath(__file__)) + "/trees_slope.md", "w").write("\n".join(out)); print("\n".join(out))
