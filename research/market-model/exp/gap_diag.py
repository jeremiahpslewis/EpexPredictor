"""Where the structural model (published residual load known, online correction) loses to the two-stage trees at night."""
import numpy as np, pandas as pd, live_common as lc, warnings; warnings.filterwarnings("ignore")
SP = lc.SP; F = lc.load(); D = F["DE"]; local = D.index.tz_convert("Europe/Berlin"); hour_all = local.hour.to_numpy(); wd = D["weekday"].to_numpy(); hol = D["holiday"].to_numpy()
dt_all = np.where((hol >= 0.5) | (wd == 6), 2, np.where(wd == 5, 1, 0)); price_all = D["price"].to_numpy(float)
CFG = {"autumn": ("ha", "two_autumn_known_d1"), "winter": ("hw2", "two_winter_known"), "summer": ("hs2", "two_summer_known")}
import os, sys
for kv in os.environ.get("RUNS", "").split(","):
    if kv: CFG[kv.split("=")[0]] = (kv.split("=")[1], CFG[kv.split("=")[0]][1])
BRIEF = os.environ.get("BRIEF", "0") == "1"
rows = []
for period, (run, two) in CFG.items():
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); cuts = I["cuts"]; tags = [str(c)[:10] for c in I["cutoffs"]]
    S = np.load(f"{SP}/out/structfc_{run}_known_dyn2xxh.npz"); s = S["vint"].astype(float); c0 = S["curve"].astype(float); t = np.load(f"{SP}/out/{two}.npz")["vint"].astype(float)
    for j in range(len(a)):
        z = np.load(f"{SP}/out/struct_{run}/{tags[j]}.npz"); d0 = int(cuts[j]); sl = np.arange(32); pos = d0 + sl
        last = price_all[d0 - 96:d0]                                                  # prices of D+1, known at issue
        rows.append(pd.DataFrame({"period": period, "j": j, "slot": sl, "hour": hour_all[pos], "dtype": dt_all[pos], "x": z["x_known"][:32], "rl": I["rl_true"][j, :32], "wind": I["wind_true"][j, :32], "load": I["load_true"][j, :32],
                                  "temp": I["eu"][j, :32, 3], "gas": float(I["gas"][j]), "a": a[j, :32], "s": s[j, :32], "c": c0[j, :32], "t": t[j, :32], "prev": last[:32], "prev_dm": np.nanmean(last),
                                  "drl": I["rl_true"][j, :32] - np.nanmean(I["rl_true"][j, :32])}))
df = pd.concat(rows); df["es"] = (df.s - df.a).abs(); df["et"] = (df.t - df.a).abs(); df["ec"] = (df.c - df.a).abs(); df["bs"] = df.s - df.a; df["bt"] = df.t - df.a; df["gap"] = df.es - df.et
pd.set_option("display.width", 250)
def tab(key, bins=None, labels=None):
    for period in CFG:
        d = df[df.period == period]; k = d[key] if bins is None else (pd.qcut(d[key], bins, duplicates="drop") if isinstance(bins, int) else pd.cut(d[key], bins))
        g = d.groupby(k, observed=True); out = pd.DataFrame({"n": g.size(), "struct": g.es.mean(), "trees": g.et.mean(), "gap": g.gap.mean(), "share of gap %": 100 * g.gap.sum() / d.gap.sum(), "bias struct": g.bs.mean(), "bias trees": g.bt.mean(), "curve only": g.ec.mean()})
        print(f"\n{period} by {key}:\n" + out.round(2).to_string())
print("night (first 8 h), MAE ct/kWh:"); print(df.groupby("period").agg(struct=("es", "mean"), trees=("et", "mean"), gap=("gap", "mean"), bias_struct=("bs", "mean"), bias_trees=("bt", "mean"), curve_only=("ec", "mean")).round(2).to_string())
tab("x", [-50, 15, 25, 35, 45, 100])
if not BRIEF: tab("hour", [-1, 1, 3, 5, 7]); tab("dtype"); tab("wind", 3); tab("temp", 3); tab("a", 4)
# per-forecast: is the gap a level error of whole nights?
for period in CFG:
    d = df[df.period == period]; g = d.groupby("j"); ls = g.bs.mean(); lt = g.bt.mean()
    ws = (d.bs - d.j.map(ls)).abs().mean(); wt = (d.bt - d.j.map(lt)).abs().mean()
    print(f"\n{period}: night-level error per forecast: struct mean |.| {ls.abs().mean():.2f}, trees {lt.abs().mean():.2f}; within-night shape error: struct {ws:.2f}, trees {wt:.2f}; corr of the two models' level errors {np.corrcoef(ls, lt)[0, 1]:.2f}")
    worst = (g.es.mean() - g.et.mean()).sort_values(ascending=False); tags = [str(c)[:10] for c in np.load(f"{SP}/out/inputs_{period}.npz")["cutoffs"]]
    print("   forecasts with the largest gap: " + ", ".join(f"{tags[j]} {v:+.2f}" for j, v in worst.head(5).items()) + f"; top 5 carry {100 * worst.head(5).sum() / worst.sum():.0f}% of the gap; struct better in {100 * (worst < 0).mean():.0f}% of forecasts")

