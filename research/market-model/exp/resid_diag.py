"""Where the in-sample errors of the structural curve sit: sequential decomposition and persistence."""
import sys, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for tag in sys.argv[1:]:
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); r = z["r"] / 10; lo, hi = np.percentile(r, [1, 99]); r = np.clip(r, lo, hi)
    df = pd.DataFrame({"r": r, "hour": z["hour"], "day": z["day"], "x": z["x"], "dtype": z["dtype"], "gas": z["gas"]}); df["week"] = (df.day - df.day.min()) // 7; df["month"] = (df.day - df.day.min()) // 30
    df["xb"] = np.clip(((df.x + 10) // 10).astype(int), 0, 7); df["hb"] = df.hour // 3
    print(f"\n{tag}: {len(df)} hours, error of the curve: MAE {np.abs(r).mean():.2f} ct, sd {r.std():.2f}")
    cur = df.r.copy(); v0 = cur.var()
    def step(label, keys, minn=1):
        global cur
        g = cur.groupby([df[k] for k in keys]); m = g.transform("mean"); n = g.transform("size"); m = m * n / (n + minn); new = cur - m
        print(f"  remove {label:<44} variance share {100 * (cur.var() - new.var()) / v0:5.1f}%   MAE after {np.abs(new).mean():.2f}"); cur = new
    step("level per month (30 d)", ["month"]); step("level per week", ["week"]); step("hour profile per month (3-h blocks)", ["month", "hb"], 5)
    step("curve shape per month (10-GW bins of x)", ["month", "xb"], 5); step("level per day", ["day"], 2); step("hour profile per week (3-h blocks)", ["week", "hb"], 3)
    print(f"  left over{'':<40} variance share {100 * cur.var() / v0:5.1f}%")
    # persistence of the raw error and of the error after the monthly terms
    P = df.pivot_table(index="day", columns="hour", values="r", aggfunc="mean").reindex(range(df.day.min(), df.day.max() + 1))
    for lab, M in (("raw error", P),):
        A = M.to_numpy(); c = lambda k: np.nanmean([np.corrcoef(A[k:, h][np.isfinite(A[k:, h]) & np.isfinite(A[:-k, h])], A[:-k, h][np.isfinite(A[k:, h]) & np.isfinite(A[:-k, h])])[0, 1] for h in range(24)])
        print(f"  correlation of the {lab} with the same hour k days earlier: " + "  ".join(f"k={k}: {c(k):.2f}" for k in (1, 2, 3, 7, 14, 28, 56)))
    dm = P.mean(1).to_numpy(); ok = lambda k: np.isfinite(dm[k:]) & np.isfinite(dm[:-k])
    print("  correlation of the day mean error k days earlier:                  " + "  ".join(f"k={k}: {np.corrcoef(dm[k:][ok(k)], dm[:-k][ok(k)])[0, 1]:.2f}" for k in (1, 2, 3, 7, 14, 28, 56)))
    # last 120 days: mean error by hour block and by season of the training window
    last = df[df.day > df.day.max() - 60]; print("  mean error, last 60 days, by 3-h block: " + " ".join(f"{v:+.1f}" for v in last.groupby("hb").r.mean()))
    yr = df[(df.day > df.day.max() - 365 - 60) & (df.day <= df.day.max() - 365)]; print("  same 60 days one year earlier:          " + " ".join(f"{v:+.1f}" for v in yr.groupby("hb").r.mean()))
