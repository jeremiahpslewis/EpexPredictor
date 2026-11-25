"""
Ensemble backtest (prereg/ENSEMBLE_PROTOCOL.md). Usage: ensemble_bt.py SP E11_PRED.pkl OUT.md
Members from the stored research arrays (SP/out, autumn set; cutoff index 5..65 = local days 2026-09-08..11-07) and the
eleven-zone forecasts; equal-weight ensembles; MAE in ct/kWh by window; paired block bootstrap by day.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SP, E11, OUT = sys.argv[1:4]
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
N_MISS = int(np.isnan(E).sum()); E = np.where(np.isnan(E), 100.7 / 10, E)   # missed solves score the training-mean fallback (OOS protocol addendum 1)
loc_h = np.stack([(c + pd.to_timedelta(np.arange(96) * 15, "min")).tz_convert("Europe/Berlin").hour for c in cut])
F = {"T trees (opt)": T, "T trees (pes)": Tp, "P Timestral": P, "A market route": A, "E eleven-zone (b)": E,
     "M3 (T+P+A)/3": (T + P + A) / 3, "M4 (T+P+A+E)/4": (T + P + A + E) / 4, "M3E (T+P+E)/3": (T + P + E) / 3}
ok = np.isfinite(a).all(1) & np.all([np.isfinite(v).all(1) for v in F.values()], axis=0)
print("days scored:", int(ok.sum()), "of", len(a), "; E slots scored with the fallback:", N_MISS)
win = {"whole day": np.ones_like(loc_h, bool), "night 0-6": loc_h < 6, "evening 16-21": (loc_h >= 16) & (loc_h < 21)}
def mae(v, m): return np.array([np.abs(v[d] - a[d])[m[d]].mean() for d in np.flatnonzero(ok)])
rows = {k: {w: mae(v, m).mean() for w, m in win.items()} for k, v in F.items()}
rng = np.random.default_rng(0); nd = int(ok.sum()); blk = 7
def boot(x):
    nb = nd // blk; xs = x[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([xs[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
pairs = [("M4 (T+P+A+E)/4", "M3 (T+P+A)/3"), ("M3E (T+P+E)/3", "M3 (T+P+A)/3"), ("E eleven-zone (b)", "M3 (T+P+A)/3"), ("E eleven-zone (b)", "A market route"),
         ("E eleven-zone (b)", "T trees (opt)"), ("E eleven-zone (b)", "P Timestral"), ("T trees (opt)", "T trees (pes)")]
out = ["# Ensemble backtest: deployed techs and the eleven-zone model (descriptive; window already used)", "",
       f"Local delivery days 2026-09-08 to 2026-11-07; {nd} days scored, 96 quarter-hours each; MAE in ct/kWh. Eleven-zone quarter-hours with a missed solve ({N_MISS}) score the training-mean fallback.", "",
       pd.DataFrame(rows).T.round(3).to_markdown(), "", "## Paired differences in MAE (first minus second; negative = first better), 90% block bootstrap (7-day blocks)", ""]
for x, y in pairs:
    for w, m in win.items():
        d = mae(F[x], m) - mae(F[y], m); b = boot(d); out.append(f"- {x} − {y}, {w}: {d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
open(OUT, "w").write("\n".join(out)); print("\n".join(out))
