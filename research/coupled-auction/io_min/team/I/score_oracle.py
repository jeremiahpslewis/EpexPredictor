"""Score the true-RL oracle (out_oracle) against the champion (out_strict) on the same days; per member, by window,
with the paired per-day differences. usage: score_oracle.py SP [ORACLE_DIR]"""
import glob, os, sys
import numpy as np, pandas as pd

SP = sys.argv[1]
OD = sys.argv[2] if len(sys.argv) > 2 else f"{SP}/team/I/out_oracle"
MEM = ["trees", "timestral", "market", "issued"]
WIN = [-1, 5, 9, 15, 20, 23]; WL = ["00-06", "06-10", "10-16", "16-21", "21-24"]
rows = []
for f in sorted(glob.glob(f"{OD}/*.npz")):
    day = os.path.basename(f)[:-4]
    o, b = np.load(f), np.load(f"{SP}/faithful/out_strict/{day}.npz")
    assert np.allclose(o["actual"], b["actual"], equal_nan=True)
    t = pd.to_datetime(b["idx"], unit="us", utc=True).tz_convert("Europe/Berlin")
    df = pd.DataFrame({"day": day, "h": t.hour, "actual": b["actual"],
                       "rl_err_base": b["rl_trees"] - b["rl_true"], "rlmix_err_base": b["rl_mixed"] - b["rl_true"]})
    for m in MEM:
        df[f"b_{m}"] = np.abs(b[m] - b["actual"]); df[f"o_{m}"] = np.abs(o[m] - o["actual"])
        df[f"bias_b_{m}"] = b[m] - b["actual"]; df[f"bias_o_{m}"] = o[m] - o["actual"]
    df["o_rl_trees_err"] = o["rl_trees"] - o["rl_true"]
    # blend of the oracle's trees with the champion's Timestral and market: trees-only RL bound on the blend
    df["o_blend_treesonly"] = np.abs((o["trees"] + b["timestral"] + b["market"]) / 3 - b["actual"])
    rows.append(df)
X = pd.concat(rows, ignore_index=True)
X["win"] = pd.cut(X.h, WIN, labels=WL)
print(f"days ({X.day.nunique()}): {', '.join(sorted(X.day.unique()))}")
print(f"max |oracle tree RL - true| on D: {X.o_rl_trees_err.abs().max():.2e} GW")
D = X.groupby("day").mean(numeric_only=True)
rng = np.random.default_rng(0)


def boot(v, n=4000):   # day-level bootstrap (days are 2+ apart mostly; ~1 per week-block)
    v = np.asarray(v); idx = rng.integers(0, len(v), (n, len(v)))
    return np.percentile(v[idx].mean(1), [5, 95])


print("\nprice MAE ct/kWh, oracle (true RL) - champion, per-day bootstrap 90%")
for m in MEM + ["blend_treesonly"]:
    bcol = "b_issued" if m == "blend_treesonly" else f"b_{m}"
    d = D[f"o_{m}"] - D[bcol]
    lo, hi = boot(d)
    print(f"  {m:16s} champ {D[bcol].mean():.3f} oracle {D[f'o_{m}'].mean():.3f} diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"({100 * d.mean() / D[bcol].mean():+.1f}%), days better {(d < 0).sum()}/{len(d)}")
print("\nby window (diff oracle - champion, ct/kWh)")
W = X.groupby("win", observed=True).mean(numeric_only=True)
print(pd.DataFrame({m: W[f"o_{m}"] - W[f"b_{m}"] for m in MEM} | {"champ_issued": W.b_issued, "RL|err| base": W.rl_err_base.abs()}).round(3).to_string())
print("\nbias (forecast - actual), champion -> oracle")
print(pd.DataFrame({m: [X[f"bias_b_{m}"].mean(), X[f"bias_o_{m}"].mean()] for m in MEM}, index=["champ", "oracle"]).round(3).to_string())
print("\nper day: RL MAE base (trees/mixed) GW, issued champ -> oracle, trees champ -> oracle")
for day, r in D.iterrows():
    dd = X[X.day == day]
    print(f"  {day}: RL {dd.rl_err_base.abs().mean():.2f}/{dd.rlmix_err_base.abs().mean():.2f}  issued {r.b_issued:.3f}->{r.o_issued:.3f}  "
          f"trees {r.b_trees:.3f}->{r.o_trees:.3f}  timestral {r.b_timestral:.3f}->{r.o_timestral:.3f}  market {r.b_market:.3f}->{r.o_market:.3f}")
