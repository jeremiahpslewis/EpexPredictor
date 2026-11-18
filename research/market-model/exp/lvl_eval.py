import sys, numpy as np, evaluate as ev
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for per, lab in ((sys.argv[1], "last ten weeks"), (sys.argv[2], "previous year")):
    zs = [np.load(f"{SP}/out/{n}.npz") for n in per.split(",")]
    z = zs[0]; cut, act = z["cutoffs"], z["actual"].astype(float)
    preds = {k: zz[k].astype(float) for zz in zs for k in zz.keys() if k not in ("cutoffs", "actual")}
    if "nn_blend" in preds and "blend" in preds: preds["trees + nn (avg)"] = (preds["blend"] + preds["nn_blend"]) / 2
    ext_lo, ext_hi = act < 5, act > 35
    for hours, hl in (((0, 8), "00-08"), (None, "all hours")):
        ev.HOURS = hours; print(f"--- {lab}, {hl}: MAE 1/2/3 days, change vs current design" + ("   | mean error where actual <5 / >35 ct" if hours is None else ""))
        base = ev.summarize(*ev.per_iter(act, preds["blend"], cut))[1]
        for k, p in preds.items():
            m = ev.summarize(*ev.per_iter(act, p, cut))[1]
            extra = f"   | {np.nanmean((p - act)[ext_lo]):+.1f} / {np.nanmean((p - act)[ext_hi]):+.1f}" if hours is None else ""
            print(f"  {k:<26} {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  " + "/".join(f"{100*(a/b-1):+.0f}%" for a, b in zip(m, base)) + extra)
