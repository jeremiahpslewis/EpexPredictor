"""4. Stability around regime changes (EXPECTATIONS.md B1-B3).
Daily-retrained single-stage trees (trees_slope.py setup: 540-day window; blend target 'full' and raw target 'full|raw'),
local days 2026-04-01..2026-11-07, rl = TSO day-ahead residual-load forecast (same input type for every day), archive
weather, gas one day before D. Compared with: stored market route A and Timestral P (summer set every 3rd day from
2026-04-01, autumn set daily), structural (b) forecasts (this study's training run 2026-01..08, hourly, and the holdout E
run; both with published TSO inputs, so E-type, not strict). Breaks: advanced hybrid coupling first MTU 2026-06-11
(from slope_train_old 'ahc'); fuel-cost index L = 1.85 Pg + 0.37 tau leaving its earlier range; French unavailability
('pregate' GW from fr_avail.parquet). Writes stability.md and SP/distill/stability_daily.parquet."""
import sys, os
import numpy as np, pandas as pd
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import trees_slope as TS
PANEL = TS.PANEL

def run_day(args):
    variant, day = args; F = TS.G["F"]; c = TS.cols(variant, F)
    d0 = F.index.get_loc(pd.Timestamp(day, tz="Europe/Berlin").tz_convert("UTC")); i1 = d0 - 34 * 4
    pred = TS.fit_models(F, c, i1 - TS.L, i1, TS.target_kind(variant))
    X = F[c].iloc[d0:d0 + 96].copy(); X["gasprice"] = F["gasprice"].iloc[d0 - 96]
    return variant, day, pd.Series(pred(X.to_numpy(float)), index=F.index[d0:d0 + 96])

if __name__ == "__main__":
    days = pd.date_range("2026-04-01", "2026-11-07", freq="D")
    with get_context("spawn").Pool(4, initializer=TS.init) as pool:
        res = pool.map(run_day, [(v, str(d.date())) for v in ("full", "full|raw") for d in days], chunksize=2)
    TS.init(); F = TS.G["F"]; price = F.price
    fc = {}
    for v, d, s in res: fc.setdefault(v, []).append(s)
    fc = {k: pd.concat(v).sort_index() for k, v in fc.items()}
    # stored market route and Timestral (summer + autumn), first 96 slots of each cutoff
    import market1 as M
    Pm = np.load(f"{O}/market1n_post.npz"); th = {k: float(np.median(Pm[k])) for k in M.PN}
    A_, P_ = [], []
    for ssn in ("summer", "autumn"):
        I = np.load(f"{O}/inputs_{ssn}.npz", allow_pickle=True); cut = pd.DatetimeIndex(I["cutoffs"]).tz_localize("UTC")
        S = np.load(f"{O}/tsshift_lwA_lite_{ssn}.npz"); Tm = np.load(f"{O}/tsm_rllwA_{ssn}.npz")["median"]
        for j, c in enumerate(cut):
            ix = c + pd.to_timedelta(np.arange(96) * 15, "min")
            A_.append(pd.Series(np.asarray(M.clear(th, S["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(I["gas"][j]) + 0.37 * 75.0), np.full(96, 75.0), eps=S["median"][j].astype(float)[:96])) / 10, index=ix))
            P_.append(pd.Series(Tm[j, :96].astype(float), index=ix))
    fc["A market route (stored)"] = pd.concat(A_).sort_index(); fc["P Timestral (stored)"] = pd.concat(P_).sort_index()
    st = []
    for f in (f"{SP}/distill/train11.pkl", f"{E11}/oos11.pkl"):
        R = pd.read_pickle(f); R = R[R.variant == "b"]
        st.append(pd.Series(R.p_draws.apply(lambda d: np.nanmedian(d) if np.isfinite(d).any() else np.nan).to_numpy() / 10, index=pd.DatetimeIndex(R.t)))
    fc["E structural (b), published inputs"] = pd.concat(st).sort_index()
    fc = {k: v[~v.index.duplicated()] for k, v in fc.items()}
    # daily (local) MAE on each forecaster's own slots; relative to the day's mean |price|
    rows = []
    for k, s in fc.items():
        e = (s - price.reindex(s.index)).dropna(); dl = e.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)
        g = e.abs().groupby(dl); lvl = price.reindex(e.index).abs().groupby(dl).mean()
        rows.append(pd.DataFrame({"model": k, "mae": g.mean(), "bias": e.groupby(dl).mean(), "lvl": lvl, "n": g.size()}))
    D = pd.concat(rows).reset_index(names="day"); D = D[D.day >= "2026-01-01"]
    D.to_parquet(f"{SP}/distill/stability_daily.parquet")
    Pn = pd.read_parquet(PANEL); Lc = (1.85 * Pn.Pg + 0.37 * Pn.tau); Ld = Lc.groupby(Lc.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)).mean()
    fr = pd.read_parquet(f"{E11}/fr_avail.parquet")["pregate"]; frd = fr.groupby(fr.index.normalize().tz_localize(None)).mean()
    train_hi = float(Lc[Lc.index < "2026-06-01"].quantile(0.95))
    first_above = Ld[(Ld > train_hi) & (Ld.index >= "2026-06-01")].index.min()
    out = ["# Stability around regime changes", "",
           f"Fuel-cost index: 95th percentile before 2026-06 = {train_hi:.0f}; first day after June with a daily mean above it: {first_above.date()}.",
           f"Monthly mean L: " + ", ".join(f"{d:%Y-%m} {v:.0f}" for d, v in Ld.resample('MS').mean().items() if d >= pd.Timestamp('2026-01-01')), "",
           "French nuclear unavailability (pregate GW), monthly: " + ", ".join(f"{d:%Y-%m} {v:.1f}" for d, v in frd.resample('MS').mean().items() if d >= pd.Timestamp('2026-01-01')), ""]
    def window(m, a, b):
        x = D[(D.model == m) & (D.day >= a) & (D.day < b)]; return x.mae.mean(), x.mae.sum() / x.lvl.sum(), x.bias.mean(), len(x)
    brk = {"hybrid coupling (2026-06-11)": pd.Timestamp("2026-06-11"), f"fuel index above earlier range ({first_above.date()})": first_above}
    for name, b in brk.items():
        tab = {}
        for m in fc:
            pre = window(m, b - pd.Timedelta(days=28), b); post = window(m, b, b + pd.Timedelta(days=14)); post2 = window(m, b + pd.Timedelta(days=14), b + pd.Timedelta(days=42))
            tab[m] = {"28d before: MAE / rel / bias / days": f"{pre[0]:.2f} / {pre[1]:.2f} / {pre[2]:+.2f} / {pre[3]}",
                      "14d after": f"{post[0]:.2f} / {post[1]:.2f} / {post[2]:+.2f} / {post[3]}",
                      "days 15-42 after": f"{post2[0]:.2f} / {post2[1]:.2f} / {post2[2]:+.2f} / {post2[3]}",
                      "rel. change, 14d after vs before": f"{post[1] / pre[1] - 1:+.0%}" if pre[3] and post[3] else "n/a"}
        out += [f"## Break: {name} (MAE ct/kWh / MAE relative to mean |price| / bias / days scored)", "", pd.DataFrame(tab).T.to_markdown(), ""]
    # monthly table
    D["month"] = D.day.dt.to_period("M").astype(str)
    mt = D.groupby(["model", "month"]).apply(lambda x: f"{x.mae.mean():.2f} ({x.mae.sum() / x.lvl.sum():.2f})").unstack()
    out += ["## Monthly: MAE ct/kWh (relative MAE)", "", mt.to_markdown(), ""]
    # French unavailability terciles over days scored by all of trees and E
    fdd = frd.reindex(D.day.unique())
    q = fdd.quantile([1 / 3, 2 / 3]).to_numpy(); tab = {}
    for m in fc:
        x = D[D.model == m].set_index("day"); x = x[x.index >= "2026-04-01"]; f = frd.reindex(x.index)
        for nm, mk in (("low unavail.", f <= q[0]), ("mid", (f > q[0]) & (f <= q[1])), ("high unavail.", f > q[1])):
            tab.setdefault(m, {})[nm] = f"{x.mae[mk].mean():.2f} ({x.mae[mk].sum() / x.lvl[mk].sum():.2f}) n{int(mk.sum())}"
    out += [f"## By French unavailability tercile (days from 2026-04-01; cut points {q[0]:.1f}, {q[1]:.1f} GW): MAE (relative MAE)", "", pd.DataFrame(tab).T.to_markdown(), ""]
    # trees' daily relative error: regression on unavailability with a month control is too thin; report partial correlation
    piv = D[D.day >= "2026-04-01"].pivot(index="day", columns="model", values="mae")
    piv["fr"] = frd.reindex(piv.index); piv["L"] = Ld.reindex(piv.index)
    out += ["## Correlation of daily MAE with French unavailability and the fuel index (days from 2026-04-01)", "",
            piv.corr().loc[[c for c in piv.columns if c not in ("fr", "L")], ["fr", "L"]].round(2).to_markdown(), ""]
    open(os.path.dirname(os.path.abspath(__file__)) + "/stability.md", "w").write("\n".join(out)); print("\n".join(out))
