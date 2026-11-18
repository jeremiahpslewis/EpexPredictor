"""The low-load regime at night (effective residual load x <= 15 GW), where the structural model loses to the trees: what separates the hours it gets wrong,
and what are the trees using? Published loads known; structural = earlier curve with per-zone neighbours + its online layer (runs ys / ya / yw)."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import live_common as lc
import oc
SP = lc.SP; F = lc.load(); D = F["DE"]; hour_all = D.index.tz_convert("Europe/Berlin").hour.to_numpy(); price_all = D["price"].to_numpy(float)
CFG = {"autumn": ("ya", "two_autumn_known_d1"), "winter": ("yw", "two_winter_known"), "summer": ("ys", "two_summer_known")}; rows = []
for period, (run, two) in CFG.items():
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); cuts = I["cuts"]; tags = [str(c)[:10] for c in I["cutoffs"]]
    S_ = np.load(f"{SP}/out/structfc_{run}_known_dyn2xxh.npz"); s = S_["vint"].astype(float); c0 = S_["curve"].astype(float); t = np.load(f"{SP}/out/{two}.npz")["vint"].astype(float)
    for j in range(len(a)):
        z = np.load(f"{SP}/out/struct_{run}/{tags[j]}.npz"); d0 = int(cuts[j]); sl = np.arange(32); pos = d0 + sl; h = pos // 4; last = price_all[d0 - 96:d0]; nb = oc.RL[h, 1:] - oc.RLZM[None]
        r = {"period": period, "j": j, "slot": sl, "hour": hour_all[pos], "x": z["x_known"][:32], "rl": I["rl_true"][j, :32], "wind": I["wind_true"][j, :32], "gas": float(I["gas"][j]), "a": a[j, :32], "s": s[j, :32], "c": c0[j, :32], "t": t[j, :32],
             "prev": last[:32] / 10 if np.nanmax(np.abs(last)) > 60 else last[:32], "prev_min": np.nanmin(last) / (10 if np.nanmax(np.abs(last)) > 60 else 1), "rl_prev": I["rl_true"][j - 1, :32] if j > 0 else np.nan, "nbsum": nb.sum(1), "rl_min_night": np.nanmin(I["rl_true"][j, :32])}
        for k, zn in enumerate(oc.NB): r["nb_" + zn] = nb[:, k]
        rows.append(pd.DataFrame(r))
df = pd.concat(rows, ignore_index=True); df["es"] = (df.s - df.a).abs(); df["et"] = (df.t - df.a).abs(); df["ec"] = (df.c - df.a).abs(); df["rs"] = df.a - df.s; df["rt"] = df.a - df.t; df["ts"] = df.t - df.s; df["ps"] = df.prev - df.s; df.to_parquet(f"{SP}/out/low_look.parquet")
pd.set_option("display.width", 250); print("unit check: actual mean", df.a.mean().round(2), "prev mean", df.prev.mean().round(2))
lo = df[df.x <= 15]
for period in CFG:
    d = lo[lo.period == period]; print(f"\n{period}: {len(d)} night quarter-hours with x <= 15 GW in {d.j.nunique()} forecasts; MAE struct {d.es.mean():.2f}, trees {d.et.mean():.2f}, curve only {d.ec.mean():.2f}; mean actual {d.a.mean():.2f}, struct {d.s.mean():.2f}, trees {d.t.mean():.2f}")
    k = pd.cut(d.x, [-60, 0, 5, 10, 15]); g = d.groupby(k, observed=True); print("   by x:            " + "  ".join(f"{i}: n {len(v)}, struct {v.es.mean():.2f}, trees {v.et.mean():.2f}, actual {v.a.mean():.1f}, s {v.s.mean():.1f}, t {v.t.mean():.1f}" for i, v in g))
    k = pd.cut(d.a, [-60, 0.5, 3, 6, 9, 60]); g = d.groupby(k, observed=True); print("   by actual price: " + "  ".join(f"{i}: n {len(v)}, struct {v.es.mean():.2f} (bias {(-v.rs).mean():+.2f}), trees {v.et.mean():.2f} (bias {(-v.rt).mean():+.2f})" for i, v in g))
    X = d[["ps", "nbsum", "wind", "x"]].copy(); ok = X.notna().all(1) & d.rs.notna()
    print("   correlation of the structural error (actual - struct) with: " + ", ".join(f"{nm} {np.corrcoef(d[c][ok], d.rs[ok])[0, 1]:+.2f}" for nm, c in (("yesterday's price at that slot minus struct", "ps"), ("neighbours' residual load (sum of deviations)", "nbsum"), ("German wind", "wind"), ("x", "x"))) + " | per neighbour: " + " ".join(f"{zn} {np.corrcoef(d['nb_' + zn][ok], d.rs[ok])[0, 1]:+.2f}" for zn in oc.NB))
    print("   correlation of what the trees add (trees - struct) with: " + ", ".join(f"{nm} {np.corrcoef(d[c][ok], d.ts[ok])[0, 1]:+.2f}" for nm, c in (("yesterday's price minus struct", "ps"), ("neighbours' residual load", "nbsum"), ("German wind", "wind"), ("x", "x"))) + f" | of trees - struct with actual - struct: {np.corrcoef(d.ts[ok], d.rs[ok])[0, 1]:+.2f}")
    for b in (0.25, 0.5):
        print(f"   diagnostic only: struct + {b} x (yesterday's price at that slot - struct): MAE {(d.a - (d.s + b * d.ps)).abs().mean():.2f}; struct + {b} x (trees - struct): {(d.a - (d.s + b * d.ts)).abs().mean():.2f}")
    # per forecast: is it whole nights?
    g = d.groupby("j"); lvl_s = g.rs.mean(); lvl_t = g.rt.mean(); print(f"   per forecast (nights with low-load hours): mean |night-level error| struct {lvl_s.abs().mean():.2f}, trees {lvl_t.abs().mean():.2f}; within-night: struct {(d.rs - d.j.map(lvl_s)).abs().mean():.2f}, trees {(d.rt - d.j.map(lvl_t)).abs().mean():.2f}")
