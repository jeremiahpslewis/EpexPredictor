import os, sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
lead = np.arange(153) / 4.0
for suf, lab in (("ho", "last ten weeks"), ("yr", "previous year")):
    if not os.path.exists(f"{SP}/out/live_{suf}.npz"): continue
    z = np.load(f"{SP}/out/live_{suf}.npz"); a = z["actual"].astype(float); P = {"trees, archived weather": z["arch"].astype(float), "trees, weather as forecast": z["vint"].astype(float)}
    if os.path.exists(f"{SP}/out/tslive_{suf}.npz"):
        t = np.load(f"{SP}/out/tslive_{suf}.npz"); assert (t["cutoffs"] == z["cutoffs"]).all()
        P["Timestral, weather as forecast"] = t["vint"].astype(float)
        for w0, tau in ((0.8, 6), (0.5, 12)):
            w = (w0 * np.exp(-lead / tau))[None, :]; P[f"blend, Timestral weight {w0} fading over {tau} h"] = (1 - w) * P["trees, weather as forecast"] + w * t["vint"]
    n = np.isfinite(a).all(axis=1) & np.all([np.isfinite(p).all(axis=1) for p in P.values()], axis=0)
    print(f"=== {lab}: {n.sum()} forecasts issued at 14:00. MAE ct/kWh by hours ahead of the issue time (first unknown hour = 34)")
    print(f"{'':<44}" + "  ".join(f"{s:>9}" for s in ("34-42 h", "42-48 h", "34-48 h", "48-58 h", "58-72 h")) + "   | night 00-08")
    for k, p in P.items():
        e = np.abs(p[n] - a[n]); seg = lambda lo, hi: e[:, int((lo - 34) * 4):int((hi - 34) * 4)].mean()
        print(f"{k:<44}" + "  ".join(f"{seg(lo, hi):9.2f}" for lo, hi in ((34, 42), (42, 48), (34, 48), (48, 58), (58, 72))) + f"   | {e[:, :32].mean():.2f}")
