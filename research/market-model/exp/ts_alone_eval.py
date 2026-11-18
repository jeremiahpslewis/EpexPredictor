"""Timestral alone vs. trees vs. blend vs. seasonal naive, live setup, 66 forecasts, weather 2-3 days old."""
import os, numpy as np
import live_common as lc
SP = lc.SP; z = np.load(f"{SP}/out/live_ho.npz"); a = z["actual"].astype(float); trees = z["vint"].astype(float); cut = z["cutoffs"]
rlE = np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"].astype(float)
F = lc.load(); idx = F["DE"].index; y = F["DE"]["price"].to_numpy(float); cuts = lc.cutoffs(idx, "2026-09-03", "2026-11-07", 1)
naive = np.stack([y[d0 - 7 * 96:d0 - 7 * 96 + lc.HZ] for d0 in cuts])            # same time a week earlier (known at issue)
rng = np.random.default_rng(0)
def ci(e1, e0):
    r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(7)) % 66 for s in rng.integers(0, 66, 10)])[:66] for _ in range(3000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
C = {"seasonal naive (a week earlier)": naive, "trees, PR #1": trees, "trees + residual load (PR #2)": rlE}
TS = {"cal": "Timestral, price + weekday only", "base": "Timestral, weather covariates", "rl": "Timestral, weather + residual load", "rlc": "Timestral, residual load + parts + gas"}
Q = {}
for k, lab in TS.items():
    if os.path.exists(f"{SP}/out/tso_{k}.npz"):
        t = np.load(f"{SP}/out/tso_{k}.npz"); assert (t["cutoffs"] == cut).all(); C[lab] = t["median"].astype(float); Q[lab] = t
if "Timestral, weather + residual load" in C:
    C["blend: trees + resid. load, 50/50 with Timestral + resid. load"] = (rlE + C["Timestral, weather + residual load"]) / 2
C["blend: trees + resid. load, 50/50 with Timestral (PR #2 now)"] = (rlE + np.load(f"{SP}/out/tsl_123_8k.npz")["vint"].astype(float)) / 2
print(f"{'MAE ct/kWh':<62}{'night':>6}{'34-48h':>8}{'48-58h':>8}{'58-72h':>8}{'RMSE':>6}  34-48h vs PR #1")
for lab, p in C.items():
    e = np.abs(p - a)
    print(f"{lab:<62}{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}{e[:, 96:152].mean():8.2f}{np.sqrt(((p - a)[:, :96] ** 2).mean()):6.2f}  {ci(e[:, :56].mean(1), np.abs(trees - a)[:, :56].mean(1)) if lab != 'trees, PR #1' else ''}")
hi, neg = a > 30, a < 0.5
print(f"\nfirst forecast day only: above 30 ct ({hi[:, :96].sum()} quarter hours) MAE / bias; below 0.5 ct ({neg[:, :96].sum()}) MAE")
for lab, p in C.items():
    h, n = hi[:, :96], neg[:, :96]; d = (p - a)[:, :96]
    print(f"  {lab:<60}{np.abs(d[h]).mean():5.1f} / {d[h].mean():+5.1f}   {np.abs(d[n]).mean():5.2f}")
print("\nTimestral quantiles, first forecast day: share of actual prices inside the interval (should be 80% / 90%), mean pinball loss q10/q90, P(>30ct) Brier")
for lab, t in Q.items():
    q10, q90, q05, q95, m = (t[k].astype(float)[:, :96] for k in ("q10", "q90", "q05", "q95", "median")); y1 = a[:, :96]
    pin = lambda q, tau: np.mean(np.maximum(tau * (y1 - q), (tau - 1) * (y1 - q)))
    # probability of exceeding 30 ct read off the quantiles by linear interpolation
    qs = np.stack([q05, q10, m, q90, q95], -1); taus = np.array([.05, .1, .5, .9, .95])
    p30 = np.array([1 - np.interp(30, qs.reshape(-1, 5)[i], taus) for i in range(qs.reshape(-1, 5).shape[0])]).reshape(y1.shape)
    print(f"  {lab:<44}{100*np.mean((y1 >= q10) & (y1 <= q90)):5.0f}% {100*np.mean((y1 >= q05) & (y1 <= q95)):5.0f}%   {pin(q10, .1):.3f} / {pin(q90, .9):.3f}   Brier {np.mean((p30 - (y1 > 30)) ** 2):.4f} (always 0: {np.mean((y1 > 30)):.4f})")
