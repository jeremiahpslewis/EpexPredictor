"""Candidate without trees: structural model (Timestral residual load, online correction) averaged with the Timestral
price model. Trees and the PR #2 blend as yardsticks. usage: cand_eval.py [SUFFIX]"""
import sys, os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; SUF = sys.argv[1] if len(sys.argv) > 1 else "dyn2xxh"
ld = lambda f, k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float); rng = np.random.default_rng(0)
CFG = {"autumn": dict(run="ha", ts=("tsl_123_8k", "vint"), two_ts="two_autumn_ts_d1", two_own="two_autumn_own_d1", two_known="two_autumn_known_d1", blk=7),
       "winter": dict(run="hw2", ts=("win_ts", "median"), two_ts="two_winter_ts", two_own="two_winter_own", two_known="two_winter_known", blk=3),
       "summer": dict(run="hs2", ts=("ssn_ts", "median"), two_ts="two_summer_ts", two_own="two_summer_own", two_known="two_summer_known", blk=3)}
for kv in os.environ.get("RUNS", "").split(","):
    if kv: CFG[kv.split("=")[0]]["run"] = kv.split("=")[1]
PREV = {k.split("=")[0]: k.split("=")[1] for k in os.environ.get("PREV", "").split(",") if k}
for period, c in CFG.items():
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); n = len(a); blk = c["blk"]
    TS = ld(*c["ts"]); S = ld(f"structfc_{c['run']}_ts_{SUF}"); SK = ld(f"structfc_{c['run']}_known_{SUF}"); C0 = ld(f"structfc_{c['run']}_ts_{SUF}", "curve")
    def ci(p1, p0, sl):
        e1, e0 = np.abs(p1 - a)[:, sl].mean(1), np.abs(p0 - a)[:, sl].mean(1)
        r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(4000))]
        return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
    def sc(p): e = np.abs(p - a); hi = a[:, :56] > 30; d = (p - a)[:, :56]; return f"{e[:, :32].mean():7.2f}{e[:, :56].mean():8.2f}" + (f"{d[hi].mean():+8.1f}" if hi.sum() >= 20 else "       -")
    ref = (ld(c["two_own"]) + TS) / 2; cand = (S + TS) / 2
    print(f"\n{period}: {n} forecasts, MAE ct/kWh{'':<26}{'night':>7}{'34-48h':>8}{'>30 bias':>8}   vs PR #2 blend: night, 34-48 h")
    print(f"  {'candidate: structural + Timestral price model':<52}{sc(cand)}   {ci(cand, ref, slice(0, 32))}   {ci(cand, ref, slice(0, 56))}")
    print(f"  {'  structural alone (Timestral residual load)':<52}{sc(S)}   {ci(S, ref, slice(0, 32))}   {ci(S, ref, slice(0, 56))}")
    print(f"  {'  structural, curve only, no online correction':<52}{sc(C0)}")
    print(f"  {'  Timestral price model alone':<52}{sc(TS)}")
    if period in PREV:
        S0 = ld(f"structfc_{PREV[period]}_ts_{SUF}"); c0 = (S0 + TS) / 2
        print(f"  {'candidate before this change':<52}{sc(c0)}   this change: {ci(cand, c0, slice(0, 32))}   {ci(cand, c0, slice(0, 56))}")
        print(f"  {'  structural alone before this change':<52}{sc(S0)}   this change: {ci(S, S0, slice(0, 32))}   {ci(S, S0, slice(0, 56))}")
    print(f"  {'yardstick: PR #2 blend (trees + Timestral price)':<52}{sc(ref)}")
    print(f"  {'yardstick: two-stage trees, Timestral resid. load':<52}{sc(ld(c['two_ts']))}")
    print(f"  {'residual load known: structural / two-stage trees':<52}{sc(SK)} / {sc(ld(c['two_known']))}   {ci(SK, ld(c['two_known']), slice(0, 32))}   {ci(SK, ld(c['two_known']), slice(0, 56))}")
