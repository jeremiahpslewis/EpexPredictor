import sys, os, numpy as np, evaluate as ev
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for base, lab, suf in (("lvl_ho", "last ten weeks", "ho"), ("lvl_yr", "previous year", "yr")):
    z = np.load(f"{SP}/out/{base}.npz"); cut, act = z["cutoffs"], z["actual"].astype(float); preds = {"current design (stage 1)": z["blend"].astype(float)}
    for n in ("ts28", "ts28nc", "ts209"):
        fn = f"{SP}/out/{n}_{suf}.npz"
        if os.path.exists(fn):
            t = np.load(fn); assert (t["cutoffs"] == cut).all(); preds[n] = t[f"{n}_{suf}"].astype(float)
            preds[f"avg current + {n}"] = (preds[n] + preds["current design (stage 1)"]) / 2
            cov = np.nanmean((act >= t[f"{n}_{suf}_q10"]) & (act <= t[f"{n}_{suf}_q90"])); preds[n + f"  [10-90% band covers {100*cov:.0f}%]"] = preds.pop(n)
    if len(preds) == 1: continue
    for hours, hl in (((0, 8), "00-08"), (None, "all hours")):
        ev.HOURS = hours; print(f"--- {lab}, {hl}: MAE 1/2/3 days, change vs current" + ("   | mean error where actual <5 / >35" if hours is None else ""))
        b = ev.summarize(*ev.per_iter(act, preds["current design (stage 1)"], cut))[1]
        for k, p in preds.items():
            m = ev.summarize(*ev.per_iter(act, p, cut))[1]
            ex = f"   | {np.nanmean((p-act)[act<5]):+.1f} / {np.nanmean((p-act)[act>35]):+.1f}" if hours is None else ""
            print(f"  {k:<44} {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  " + "/".join(f"{100*(x/y-1):+.0f}%" for x, y in zip(m, b)) + ex)
