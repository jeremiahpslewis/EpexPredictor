"""Live arm (Timestral loads): errors of the structural model (+ its layer), of Timestral's price forecast and of their plain average, by the structural
model's own effective residual load for the forecast hour (x from Timestral loads, known at the origin). Is one of the two better in some range?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
CFG = {"summer": ("ys", "ssn_ts", "median", "two_summer_ts"), "autumn": ("ya", "tsl_123_8k", "vint", "two_autumn_ts_d1"), "winter": ("yw", "win_ts", "median", "two_winter_ts")}; rows = []
for period, (run, tsf, tsk, two) in CFG.items():
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); tags = [str(c)[:10] for c in I["cutoffs"]]; s = np.load(f"{SP}/out/structfc_{run}_ts_dyn2xxh.npz")["vint"].astype(float); m = np.load(f"{SP}/out/{tsf}.npz")[tsk].astype(float); t = np.load(f"{SP}/out/{two}.npz")["vint"].astype(float)
    for j in range(len(a)):
        z = np.load(f"{SP}/out/struct_{run}/{tags[j]}.npz"); x = z["x_ts"] if "x_ts" in z.files else z["x_known"]
        rows.append(pd.DataFrame({"period": period, "j": j, "slot": np.arange(56), "x": x[:56], "a": a[j, :56], "s": s[j, :56], "m": m[j, :56], "t": t[j, :56]}))
df = pd.concat(rows, ignore_index=True); df["c"] = 0.5 * (df.s + df.m); df.to_parquet(f"{SP}/out/blend_look.parquet"); pd.set_option("display.width", 250)
bins = [-60, 5, 15, 25, 35, 45, 100]
for hz, lab in ((32, "night"), (56, "first 14 h")):
    for period in CFG:
        d = df[(df.period == period) & (df.slot < hz)]; g = d.groupby(pd.cut(d.x, bins), observed=True)
        out = pd.DataFrame({"n": g.size(), "structural": g.apply(lambda v: (v.s - v.a).abs().mean()), "Timestral price": g.apply(lambda v: (v.m - v.a).abs().mean()), "average": g.apply(lambda v: (v.c - v.a).abs().mean()), "trees": g.apply(lambda v: (v.t - v.a).abs().mean()),
                            "best fixed weight on structural": g.apply(lambda v: min(np.arange(0, 1.01, 0.1), key=lambda w: (w * v.s + (1 - w) * v.m - v.a).abs().mean())), "MAE at that weight": g.apply(lambda v: min((w * v.s + (1 - w) * v.m - v.a).abs().mean() for w in np.arange(0, 1.01, 0.1)))})
        tot = sum(len(v) * min((w * v.s + (1 - w) * v.m - v.a).abs().mean() for w in np.arange(0, 1.01, 0.1)) for _, v in g) / len(d)
        print(f"\n{period}, {lab}, live: all hours: structural {(d.s - d.a).abs().mean():.2f}, Timestral price {(d.m - d.a).abs().mean():.2f}, average {(d.c - d.a).abs().mean():.2f}, trees {(d.t - d.a).abs().mean():.2f}; with the best weight per range (hindsight, upper bound) {tot:.2f}"); print(out.round(2).to_string())
