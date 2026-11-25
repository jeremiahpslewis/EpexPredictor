"""
Ensemble backtest (prereg/ENSEMBLE_PROTOCOL.md). Usage: ensemble_bt.py SP E11_PRED.pkl OUT.md
Members from the stored research arrays (SP/out, autumn set; cutoff index 5..65 = local days 2026-09-08..11-07) and the
eleven-zone forecasts; equal-weight ensembles; MAE in ct/kWh by window; paired block bootstrap by day.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SP, E11, OUT = sys.argv[1:4]; E11S = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, "/home/user/EpexPredictor/research/market-model/exp"); import market1 as M
O = f"{SP}/out"; DAYS = slice(5, 66)
I = np.load(f"{O}/inputs_autumn.npz", allow_pickle=True); cut = pd.DatetimeIndex(I["cutoffs"]).tz_localize("UTC")[DAYS]
a = I["actual"].astype(float)[DAYS, :96]
T = np.load(f"{O}/rl_E_opt_T.npz")["vint"].astype(float)[DAYS, :96]; Tp = np.load(f"{O}/rl_E_pes_T.npz")["vint"].astype(float)[DAYS, :96]
P = np.load(f"{O}/tsm_rllwA_autumn.npz")["median"].astype(float)[DAYS, :96]
Pm = np.load(f"{O}/market1n_post.npz"); th = {k: float(np.median(Pm[k])) for k in M.PN}; S = np.load(f"{O}/tsshift_lwA_lite_autumn.npz"); gas = I["gas"]
A = np.stack([np.asarray(M.clear(th, S["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(gas[j]) + 0.37 * 75.0), np.full(96, 75.0), eps=S["median"][j].astype(float)[:96])) / 10
              for j in range(5, 66)])
R = pd.read_pickle(E11); R = R[R.variant == "b"]
e11 = pd.Series(R.p_draws.apply(lambda d: np.nanmedian(d) if np.isfinite(d).any() else np.nan).to_numpy() / 10, index=pd.DatetimeIndex(R.t))
E = np.stack([e11.reindex(c + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy() for c in cut])
def member(path):
    R = pd.read_pickle(path); R = R[R.variant == "b"]
    s_ = pd.Series(R.p_draws.apply(lambda d: np.nanmedian(d) if np.isfinite(d).any() else np.nan).to_numpy() / 10, index=pd.DatetimeIndex(R.t))
    return np.stack([s_.reindex(c + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy() for c in cut])
Eraw = E.copy(); N_MISS = int(np.isnan(E).sum()); E = np.where(np.isnan(E), 100.7 / 10, E)   # missed solves score the training-mean fallback (OOS protocol addendum 1)
Esraw = member(E11S) if E11S else None; Es = np.where(np.isnan(Esraw), 100.7 / 10, Esraw) if E11S else None
loc_h = np.stack([(c + pd.to_timedelta(np.arange(96) * 15, "min")).tz_convert("Europe/Berlin").hour for c in cut])
F = {"T trees (opt)": T, "T trees (pes)": Tp, "P Timestral": P, "A market route": A, "E eleven-zone (b)": E,
     "M3 (T+P+A)/3": (T + P + A) / 3, "M4 (T+P+A+E)/4": (T + P + A + E) / 4, "M3E (T+P+E)/3": (T + P + E) / 3}
renorm = lambda X: np.nanmean(np.stack(X), 0)               # post hoc: a missed member is left out, as the deployed blend does
F["M4 renormalised (post hoc)"] = renorm([T, P, A, Eraw])
if E11S:
    F["E_s strict eleven-zone"] = Es; F["M4s (T+P+A+E_s)/4"] = (T + P + A + Es) / 4; F["M3Es (T+P+E_s)/3"] = (T + P + Es) / 3
    F["M4s renormalised (post hoc)"] = renorm([T, P, A, Esraw])
ok = np.isfinite(a).all(1) & np.all([np.isfinite(v).all(1) for v in F.values()], axis=0)
print("days scored:", int(ok.sum()), "of", len(a), "; E slots scored with the fallback:", N_MISS)
win = {"whole day": np.ones_like(loc_h, bool), "night 0-6": loc_h < 6, "evening 16-21": (loc_h >= 16) & (loc_h < 21)}
def mae(v, m): return np.array([np.abs(v[d] - a[d])[m[d]].mean() for d in np.flatnonzero(ok)])
rows = {k: {w: mae(v, m).mean() for w, m in win.items()} for k, v in F.items()}
rng = np.random.default_rng(0); nd = int(ok.sum()); blk = 7
def boot(x):
    nb = nd // blk; xs = x[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([xs[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
pairs = ([("E_s strict eleven-zone", "E eleven-zone (b)"), ("M4s (T+P+A+E_s)/4", "M3 (T+P+A)/3"), ("M3Es (T+P+E_s)/3", "M3 (T+P+A)/3"), ("M4s renormalised (post hoc)", "M3 (T+P+A)/3")] if E11S else []) + [("M4 renormalised (post hoc)", "M3 (T+P+A)/3"), ("M4 (T+P+A+E)/4", "M3 (T+P+A)/3"), ("M3E (T+P+E)/3", "M3 (T+P+A)/3"), ("E eleven-zone (b)", "M3 (T+P+A)/3"), ("E eleven-zone (b)", "A market route"),
         ("E eleven-zone (b)", "T trees (opt)"), ("E eleven-zone (b)", "P Timestral"), ("T trees (opt)", "T trees (pes)")]
out = ["# Ensemble backtest: deployed techs and the eleven-zone model (descriptive; window already used)", "",
       f"Local delivery days 2026-09-08 to 2026-11-07; {nd} days scored, 96 quarter-hours each; MAE in ct/kWh. Eleven-zone quarter-hours with a missed solve ({N_MISS}) score the training-mean fallback.", "",
       pd.DataFrame(rows).T.round(3).to_markdown(), "", "## Paired differences in MAE (first minus second; negative = first better), 90% block bootstrap (7-day blocks)", ""]
for x, y in pairs:
    for w, m in win.items():
        d = mae(F[x], m) - mae(F[y], m); b = boot(d); out.append(f"- {x} − {y}, {w}: {d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
if E11S:
    # post hoc: per local time-of-day window, non-negative weights on (T, P, A, E_s) summing to 1, chosen on a 0.1 simplex
    # grid by MAE on all weeks except the one being forecast (leave-one-week-out); the weights are applied to that week
    import itertools
    mem = [T, P, A, Es]; wins = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24)]
    grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=4) if sum(w) == 10]
    week = np.arange(len(a)) // 7; W = np.zeros_like(a)
    for wk in np.unique(week):
        tr_ = (week != wk) & ok
        for lo, hi in wins:
            m = (loc_h >= lo) & (loc_h < hi)
            best = min(grid, key=lambda g: np.abs(sum(gi * mi for gi, mi in zip(g, mem)) - a)[tr_[:, None] & m].mean())
            sel = (week == wk)[:, None] & m; W[sel] = sum(gi * mi for gi, mi in zip(best, mem))[sel]
    F["ToD weights, leave-week-out (post hoc)"] = W
    full = {w_: min(grid, key=lambda g: np.abs(sum(gi * mi for gi, mi in zip(g, mem)) - a)[ok[:, None] & (loc_h >= w_[0]) & (loc_h < w_[1])].mean()) for w_ in wins}
    out += ["", "## Post hoc: time-of-day weights", "", "Weights on (T, P, A, E_s) fitted on all days, for reference (each week's forecast used weights fitted without that week):", ""]
    out += [f"- {lo:02d}-{hi:02d} local: T {g[0]:.1f}, P {g[1]:.1f}, A {g[2]:.1f}, E_s {g[3]:.1f}" for (lo, hi), g in full.items()]
    rows = {k: {w: mae(v, m).mean() for w, m in win.items()} for k, v in F.items()}
    out[4] = pd.DataFrame(rows).T.round(3).to_markdown()
    for w, m in win.items():
        d = mae(F["ToD weights, leave-week-out (post hoc)"], m) - mae(F["M3 (T+P+A)/3"], m); b = boot(d); out.append(f"- ToD weights − M3, {w}: {d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
open(OUT, "w").write("\n".join(out)); print("\n".join(out))
