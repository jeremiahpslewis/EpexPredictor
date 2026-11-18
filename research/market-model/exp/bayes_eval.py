"""Bayesian benchmark vs the other candidates on the same origins. usage: bayes_eval.py PERIOD RUN [RUN2]"""
import sys, glob, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
period, runs = sys.argv[1], sys.argv[2:]
if period == "autumn":
    L = np.load(f"{SP}/out/live_ho.npz"); cut = L["cutoffs"]; a = L["actual"].astype(float)
    C = {"trees, PR #1": L["vint"], "trees + residual load": np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"]}
    C["blend (PR #2)"] = (C["trees + residual load"].astype(float) + np.load(f"{SP}/out/tsl_123_8k.npz")["vint"].astype(float)) / 2
    tsq = np.load(f"{SP}/out/tso_base.npz"); W = slice(0, 56); blk = 7
else:
    L = np.load(f"{SP}/out/ssn_trees.npz"); cut = L["cutoffs"]; a = L["actual"].astype(float)
    C = {"trees, PR #1": L["vint"], "trees + residual load": np.load(f"{SP}/out/ssn_rl.npz")["vint"]}
    C["blend (PR #2)"] = (C["trees + residual load"].astype(float) + np.load(f"{SP}/out/ssn_ts.npz")["median"].astype(float)) / 2
    tsq = np.load(f"{SP}/out/ssn_ts.npz"); W = slice(0, 56); blk = 3
C = {k: v.astype(float) for k, v in C.items()}
tags = [str(c)[:10] for c in cut]; B = {}
for r in runs:
    files = {f.split("/")[-1][:-4]: f for f in glob.glob(f"{SP}/out/bayes_{r}/*.npz")}
    have = [t for t in tags if t in files]
    B[r] = {t: np.load(files[t]) for t in have}
common = [i for i, t in enumerate(tags) if all(t in B[r] for r in runs)]
print(f"{period}: {len(common)} of {len(tags)} forecasts available for all runs")
sel = np.array(common); a = a[sel]; C = {k: v[sel] for k, v in C.items()}
for r in runs:
    C[f"Bayes ({r})"] = np.stack([B[r][tags[i]]["q_prop"][2] for i in common])
    C[f"Bayes ({r}), RL taken as exact"] = np.stack([B[r][tags[i]]["q_exact"][2] for i in common])
ref = C["trees + residual load"]; rng = np.random.default_rng(0); n = len(common)
def ci(e1, e0):
    r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(3000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
print(f"{'MAE ct/kWh':<44}{'night':>7}{'34-48h':>8}   night vs trees+RL      34-48h vs trees+RL")
for lab, p in C.items():
    e = np.abs(p - a)
    extra = "" if lab == "trees + residual load" else f"   {ci(e[:, :32].mean(1), np.abs(ref - a)[:, :32].mean(1)):<22} {ci(e[:, W].mean(1), np.abs(ref - a)[:, W].mean(1))}"
    print(f"{lab:<44}{e[:, :32].mean():7.2f}{e[:, W].mean():8.2f}{extra}")
y1 = a[:, W]
def qstats(q):
    q05, q10, m, q90, q95 = (x[:, W] for x in q)
    pin = lambda qq, tau: np.mean(np.maximum(tau * (y1 - qq), (tau - 1) * (y1 - qq)))
    qs = np.stack([q05, q10, m, q90, q95], -1).reshape(-1, 5); taus = np.array([.05, .1, .5, .9, .95])
    p30 = np.array([1 - np.interp(30, row, taus, left=0, right=1) for row in qs]).reshape(y1.shape)
    return f"{100*np.mean((y1 >= q10) & (y1 <= q90)):4.0f}% {100*np.mean((y1 >= q05) & (y1 <= q95)):5.0f}%   {pin(q10, .1):.3f} / {pin(q90, .9):.3f}   Brier {np.mean((p30 - (y1 > 30)) ** 2):.4f}"
print(f"\nintervals, first 14 h (should be 80% / 90%), pinball q10/q90, P(>30 ct) Brier (always 0: {np.mean(y1 > 30):.4f})")
print(f"  {'Timestral alone':<42}" + qstats([tsq[k].astype(float)[sel] for k in ("q05", "q10", "median", "q90", "q95")]))
for r in runs:
    for kind, lab in (("q_prop", "RL error propagated"), ("q_exact", "RL taken as exact")):
        print(f"  {'Bayes (' + r + '), ' + lab:<42}" + qstats(list(np.stack([B[r][tags[i]][kind] for i in common]).transpose(1, 0, 2))))
for r in runs:
    rl = np.stack([B[r][tags[i]]["rl_hat"] for i in common])[:, :96]; sq = np.stack([B[r][tags[i]]["slope_q"] for i in common])[:, :, :96]; gq = np.stack([B[r][tags[i]]["gas10_q"] for i in common])[:, :, :96]
    lo, hi = np.percentile(rl, [33, 67])
    print(f"\nBayes ({r}): EUR/MWh per +1 GW residual load and per +10% gas, first forecast day; mean of the posterior medians [mean 5%..95% posterior bounds]")
    for lab, m in ((f"residual load < {lo:.0f} GW", rl < lo), (f"{lo:.0f}-{hi:.0f} GW", (rl >= lo) & (rl < hi)), (f">= {hi:.0f} GW", rl >= hi)):
        print(f"  {lab:<24} +1 GW {sq[:, 1][m].mean():+5.1f} [{sq[:, 0][m].mean():+.1f}..{sq[:, 2][m].mean():+.1f}]   gas +10% {gq[:, 1][m].mean():+5.1f} [{gq[:, 0][m].mean():+.1f}..{gq[:, 2][m].mean():+.1f}]")
    f = np.stack([B[r][tags[i]]["fit"] for i in common]); f = f[f[:, 5] == 1]
    print(f"  {len(f)} fits: {f[:, 0].mean():.0f} s each, max r_hat {f[:, 1].max():.3f}, min ess {f[:, 2].min():.0f}, divergences {int(f[:, 3].sum())}")
