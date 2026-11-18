"""Frozen structural model against the trees, per period and arm, with paired intervals. usage: final_eval.py PERIOD RUN"""
import sys, os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; period, run = sys.argv[1], sys.argv[2]
I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); n = len(a); blk = 7 if period == "autumn" else 3; rng = np.random.default_rng(0)
ld = lambda f, k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float)
two = {"autumn": {"known": "two_autumn_known_d1", "own": "two_autumn_own_d1", "ts": "two_autumn_ts_d1", "avg": "two_autumn_avg_d1"}}.get(period, {k: f"two_{period}_{k}" for k in ("known", "own", "ts", "avg")})
T = {k: ld(f) for k, f in two.items() if os.path.exists(f"{SP}/out/{f}.npz")}
tsp = {"autumn": ("tsl_123_8k", "vint"), "summer": ("ssn_ts", "median"), "winter": ("win_ts", "median")}[period]; TS = ld(*tsp) if os.path.exists(f"{SP}/out/{tsp[0]}.npz") else None
pr1 = {"autumn": "live_ho", "summer": "ssn_trees", "winter": "win_trees"}[period]; P1 = ld(pr1) if os.path.exists(f"{SP}/out/{pr1}.npz") else None
S = {k: ld(f"structfc_{run}_{k}_R1") for k in ("known", "own", "ts", "avg") if os.path.exists(f"{SP}/out/structfc_{run}_{k}_R1.npz")}
for X in (T, S):
    for k, v in X.items(): assert v.shape == a.shape, (k, v.shape, a.shape)
W = [("night", slice(0, 32)), ("34-48h", slice(0, 56))] + ([("48-58h", slice(56, 96))] if period == "autumn" else [])
def ci(p1, p0, sl):
    e1, e0 = np.abs(p1 - a)[:, sl].mean(1), np.abs(p0 - a)[:, sl].mean(1)
    r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(4000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
def sc(p):
    hi = a[:, :56] > 30; d = (p - a)[:, :56]
    return "".join(f"{np.abs(p - a)[:, sl].mean():8.2f}" for _, sl in W) + (f"   {np.abs(d[hi]).mean():4.1f} /{d[hi].mean():+5.1f}" if hi.sum() >= 20 else "       -     ")
print(f"{period}: {n} forecasts, {str(I['cutoffs'][0])[:10]} to {str(I['cutoffs'][-1])[:10]}; MAE ct/kWh; mean price {a[:, :56].mean():.1f}, gas {I['gas'].min():.0f}-{I['gas'].max():.0f}, quarter hours above 30 ct in the first 14 h: {int((a[:, :56] > 30).sum())}")
print(f"{'':<58}" + "".join(f"{w:>8}" for w, _ in W) + "   >30ct MAE/bias   vs reference: " + ", ".join(w for w, _ in W[:2]))
def row(lab, p, ref=None):
    print(f"{lab:<58}{sc(p)}" + ("   " + "   ".join(ci(p, ref, sl) for _, sl in W[:2]) if ref is not None else ""))
print("-- published residual load known"); row("two-stage trees", T["known"]); row("structural", S["known"], T["known"]); row("50/50 structural + trees", (S["known"] + T["known"]) / 2, T["known"])
if "own" in T:
    print("-- operational: tree residual-load forecast (as in PR #2)")
    if P1 is not None: row("PR #1 trees (no residual load)", P1)
    row("two-stage trees", T["own"]); 
    if TS is not None: row("PR #2 blend: trees + Timestral price model", (T["own"] + TS) / 2); ref = (T["own"] + TS) / 2
    else: ref = T["own"]
    row("structural", S["own"], T["own"])
for k, lab in (("avg", "average of tree and Timestral residual-load forecasts"), ("ts", "Timestral residual-load forecast")):
    if k not in T or k not in S: continue
    print(f"-- operational: {lab}   (intervals against the PR #2 blend above)")
    row("two-stage trees", T[k], ref); row("structural", S[k], ref); row("50/50 structural + trees", (S[k] + T[k]) / 2, ref)
    if TS is not None: row("trees + Timestral price model", (T[k] + TS) / 2, ref); row("a third each: structural, trees, Timestral price model", (S[k] + T[k] + TS) / 3, ref)
