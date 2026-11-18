"""Before building the two-zone model: does the neighbours' residual load order the curve's error at low German
residual load at night, also when it is only a forecast? Is there a stable neighbour price curve?"""
import numpy as np, pandas as pd, os, live_common as lc, warnings; warnings.filterwarnings("ignore")
SP = lc.SP; F = lc.load(); idx = F["DE"].index; local = idx.tz_convert("Europe/Berlin"); hour_all = local.hour.to_numpy(); month_all = local.month.to_numpy()
nb = pd.read_parquet(f"{SP}/out/nb_rl.parquet"); nbs = nb[[c for c in nb.columns if c.startswith("rl_")]].sum(axis=1, min_count=8).reindex(idx).to_numpy(float)
west = np.nanmean(np.stack([F[z]["price"].to_numpy(float) for z in ("NL", "BE", "FR")]), 0); de = F["DE"]["price"].to_numpy(float)
seas = lambda m: np.where(np.isin(m, [11, 12, 1, 2]), "Nov-Feb", np.where(np.isin(m, [5, 6, 7, 8]), "May-Aug", "spring/autumn"))
print("A. in-sample error of the frozen curve (actual - curve, ct), night hours with effective residual load < 15 GW, by tercile of the neighbours' residual load")
for tag in ("2026-02-26", "2026-11-06"):
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); df = pd.DataFrame({"r": z["r"] / 10, "x": z["x"], "hour": z["hour"], "pos": z["pos"]}); df["nb"] = nbs[df.pos]; df["season"] = seas(month_all[df.pos]); s = df[(df.hour < 8) & (df.x < 15) & np.isfinite(df.nb)]
    for k, v in s.groupby("season"):
        if len(v) < 60: continue
        q = pd.qcut(v.nb, 3, labels=["low", "mid", "high"]); g = v.groupby(q, observed=True)
        print(f"   fit {tag} {k:<14} n {len(v):4d}  " + "  ".join(f"{a}: nb {g.nb.mean()[a]:.0f} GW err {g.r.mean()[a]:+.2f}" for a in ("low", "mid", "high")) + f"   corr {np.corrcoef(v.nb, v.r)[0, 1]:+.2f}")
print("\nB. forecasts: curve-only error at night (first 8 h), slots with x < 15 GW, by tercile of the neighbours' residual load: published / Timestral forecast")
for period, run in (("autumn", "ha"), ("winter", "hw2"), ("summer", "hs2")):
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); tags = [str(c)[:10] for c in I["cutoffs"]]; rows = []
    for j, t in enumerate(tags):
        z = np.load(f"{SP}/out/struct_{run}/{t}.npz"); d0 = int(I["cuts"][j]); fn = f"{SP}/out/tsnb_{period}/{t}.npz"; pes = np.load(fn)["pes"][:32] if os.path.exists(fn) else np.full(32, np.nan)
        rows.append(pd.DataFrame({"r": a[j, :32] - z["known_shape"][:32], "x": z["x_known"][:32], "nb": nbs[d0:d0 + 32], "nbf": pes}))
    s = pd.concat(rows); s = s[(s.x < 15) & np.isfinite(s.nb) & np.isfinite(s.nbf)]
    if len(s) < 60: print(f"   {period}: too few slots ({len(s)})"); continue
    out = f"   {period:<7} n {len(s):4d}"
    for col, lab in (("nb", "published"), ("nbf", "Timestral")):
        q = pd.qcut(s[col], 3, labels=["low", "mid", "high"]); g = s.groupby(q, observed=True); out += f" | {lab}: " + " ".join(f"{g[col].mean()[k]:.0f} GW {g.r.mean()[k]:+.2f}" for k in ("low", "mid", "high")) + f" corr {np.corrcoef(s[col], s.r)[0, 1]:+.2f}"
    print(out + f" | forecast error of the regional figure here: MAE {np.abs(s.nbf - s.nb).mean():.1f} GW")
print("\nC. price of the western neighbours (mean NL, BE, FR; ct) at night against the neighbours' residual load, by season, last 18 months; and how often DE = NL")
m = (hour_all < 8) & np.isfinite(nbs) & np.isfinite(west) & (np.arange(len(idx)) > len(idx) - 96 * 540); d = pd.DataFrame({"nb": nbs[m], "west": west[m], "season": seas(month_all[m]), "gas": F["DE"]["gasprice"].to_numpy(float)[m]})
bins = [0, 60, 70, 80, 90, 100, 110, 200]
print(d.groupby([pd.cut(d.nb, bins), "season"], observed=True).west.agg(["mean", "size"]).unstack().round(1).to_string())
nl = F["NL"]["price"].to_numpy(float); n2 = (np.arange(len(idx)) > len(idx) - 96 * 540) & np.isfinite(nl) & np.isfinite(de)
for lab, mm in (("all hours", n2), ("night", n2 & (hour_all < 8))): print(f"   DE within 0.2 ct of NL: {lab} {100 * np.mean(np.abs(de[mm] - nl[mm]) < 0.2):.0f}%; within 1 ct {100 * np.mean(np.abs(de[mm] - nl[mm]) < 1.0):.0f}%")
