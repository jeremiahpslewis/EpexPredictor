"""Predictive intervals of the structural model (posterior mode plugged in; error scale from the model, level from the
filter, own residual-load forecasts with resampled error paths) against Timestral's. usage: interval_eval.py PERIOD RUN"""
import sys, os, numpy as np
from scipy import stats
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; period, run = sys.argv[1], sys.argv[2]
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; a = I["actual"].astype(float); W = slice(0, 56); y = a[:, W]
Z = [np.load(f"{SP}/out/struct_{run}/{t}.npz") for t in tags]; L = np.load(f"{SP}/out/struct_{run}/level_R1.npz")["lvlh"]; rng = np.random.default_rng(0); TAUS = np.array([.05, .1, .5, .9, .95])
def stats_of(q):   # q: (5, n, T)
    q05, q10, m, q90, q95 = (x[:, W] for x in q); pin = lambda qq, tau: np.mean(np.maximum(tau * (y - qq), (tau - 1) * (y - qq)))
    qs = np.stack([q05, q10, m, q90, q95], -1).reshape(-1, 5); p30 = np.array([1 - np.interp(30, r, TAUS, left=0, right=1) for r in qs]).reshape(y.shape)
    return f"{100*np.mean((y >= q10) & (y <= q90)):5.0f}% {100*np.mean((y >= q05) & (y <= q95)):5.0f}%   {(q90 - q10).mean():5.2f}   {pin(q10, .1):.3f} / {pin(q90, .9):.3f}" + (f"   {np.mean((p30 - (y > 30)) ** 2):.4f}" if (y > 30).sum() >= 20 else "")
print(f"{period}, first 14 h: share of prices inside the 80% and 90% intervals, mean width of the 80% interval (ct), pinball loss q10 / q90" + (f", Brier P(>30 ct) (always 0: {np.mean(y > 30):.4f})" if (y > 30).sum() >= 20 else ""))
tsf = {"autumn": "tso_base", "summer": "ssn_ts", "winter": "win_ts"}[period]
if os.path.exists(f"{SP}/out/{tsf}.npz"):
    t = np.load(f"{SP}/out/{tsf}.npz"); print(f"  {'Timestral price model':<46}" + stats_of([t[k].astype(float) for k in ("q05", "q10", "median", "q90", "q95")]))
nu = np.array([z["misc"][1] for z in Z])
qk = np.stack([np.stack([z["known_shape"] + L[i] + z["scale_known"] * stats.t.ppf(tau, nu[i]) for i, z in enumerate(Z)]) for tau in TAUS]); print(f"  {'structural, published residual load known':<46}" + stats_of(qk))
for arm, lab in (("avg", "structural, average residual-load forecast"), ("own", "structural, tree residual-load forecast")):
    if f"mix_{arm}" not in Z[0].files: continue
    for prop in (True, False):
        Q = []
        for i, z in enumerate(Z):
            base = z[f"{arm}_shape"] - z[f"mixc_{arm}"]                         # within-hour shape
            if prop: d = (z[f"mix_{arm}"][:, None, :] + z[f"mixs_{arm}"][:, None, :] * rng.standard_t(nu[i], size=(40, 50, a.shape[1]))).reshape(-1, a.shape[1]) + base + L[i]
            else: d = z[f"{arm}_shape"] + L[i] + z[f"mixs_{arm}"].mean(0) * rng.standard_t(nu[i], size=(2000, a.shape[1]))
            Q.append(np.percentile(d, 100 * TAUS, axis=0))
        print(f"  {lab + (', its error propagated' if prop else ', taken as exact'):<46}"[:70].ljust(70) + stats_of(np.stack(Q, 1)))
