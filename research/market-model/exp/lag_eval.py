import os, numpy as np, evaluate as ev
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for suf, lab in (("ho", "last ten weeks"), ("yr", "previous year")):
    fn = f"{SP}/out/lag_{suf}.npz"
    if not os.path.exists(fn): continue
    z = np.load(fn); t = np.load(f"{SP}/out/ts28_{suf}.npz"); cut = z["cutoffs"]; a = z["actual"].astype(float)
    P = {"current design": z["blend"].astype(float), "trees + recent prices": z["lag_blend"].astype(float), "Timestral-28M": t[f"ts28_{suf}"].astype(float)}
    P["avg current + Timestral"] = (P["current design"] + P["Timestral-28M"]) / 2
    P["avg (trees + recent prices) + Timestral"] = (P["trees + recent prices"] + P["Timestral-28M"]) / 2
    for hours, hl in (((0, 8), "00-08"), (None, "all hours")):
        ev.HOURS = hours; b = ev.summarize(*ev.per_iter(a, P["current design"], cut))[1]; print(f"--- {lab}, {hl}: MAE 1/2/3 days, change vs current")
        for k, p in P.items():
            m = ev.summarize(*ev.per_iter(a, p, cut))[1]; print(f"  {k:<42} {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  " + "/".join(f"{100*(x/y-1):+.0f}%" for x, y in zip(m, b)))
    print(f"--- {lab}: MAE by hours after the last known price (current / trees + recent prices / Timestral)")
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72)):
        sl = slice(lo * 4, hi * 4); f = lambda p: np.nanmean(np.abs(p[:, sl] - a[:, sl])); print(f"   {lo:>2}-{hi:<2}h  {f(P['current design']):.2f} / {f(P['trees + recent prices']):.2f} / {f(P['Timestral-28M']):.2f}")
    hi35 = a > 35; print("    mean error where actual > 35 ct:", " | ".join(f"{k}: {np.nanmean((p - a)[hi35]):+.1f}" for k, p in list(P.items())[:3]))
