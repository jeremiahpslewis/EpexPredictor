"""Per border: how the scheduled exchange relates to the price difference, and what German weather adds. Hourly, hours with exchange data."""
import numpy as np, pandas as pd, live_common as lc
SP = lc.SP; D = pd.read_parquet(f"{SP}/out/star_data.parquet"); F = lc.load()["DE"]
D["wind"] = F[[c for c in F.columns if c.startswith("wind_")]].mean(1).values; D["sun"] = F[[c for c in F.columns if c.startswith("irradiance_")]].mean(1).values
H = D.resample("1h").mean(); NB = ["fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]
w = (H["wind"] - H["wind"].mean()) / H["wind"].std(); s = (H["sun"] - H["sun"].mean()) / H["sun"].std()
print("exports from Germany (GW, + = export) against the neighbour-minus-German price difference dp (EUR/MWh)")
for z in NB:
    x = -H[f"x_{z}"]; dp = H[f"p_{z}"] - H["p_de"]; k = x.notna() & dp.notna(); x, dpv, wv, sv = x[k].values, dp[k].values, w[k].values, s[k].values
    sh = lambda a: 100 * np.mean(np.abs(dpv) <= a)
    line = f"{z}: n {k.sum()}; |dp| <= 0.5: {sh(0.5):.0f}%, <= 2: {sh(2):.0f}%, <= 5: {sh(5):.0f}%, > 20: {100 - sh(20):.0f}% | exports 1/50/99%: {np.percentile(x, 1):+.1f} {np.percentile(x, 50):+.1f} {np.percentile(x, 99):+.1f}"
    bins = [(-1e9, -20), (-20, -5), (-5, -0.5), (-0.5, 0.5), (0.5, 5), (5, 20), (20, 1e9)]; med = []
    for a, b in bins:
        kk = (dpv > a) & (dpv <= b); med.append(f"{np.median(x[kk]):+.1f} [{np.percentile(x[kk], 10):+.1f},{np.percentile(x[kk], 90):+.1f}]" if kk.sum() > 30 else "  .  ")
    print(line); print("     median [10%,90%] by dp bin (<-20, -20..-5, -5..-0.5, coupled, 0.5..5, 5..20, >20):  " + "  ".join(med))
    # what weather adds: within the saturated bins, regression of exports on wind and sun
    for lab, kk in (("dp < -5 (importing side)", dpv < -5), ("dp > 5 (exporting side)", dpv > 5), ("coupled", np.abs(dpv) <= 0.5)):
        if kk.sum() > 200:
            X = np.column_stack([np.ones(kk.sum()), wv[kk], sv[kk]]); b, *_ = np.linalg.lstsq(X, x[kk], rcond=None); r2 = 1 - np.var(x[kk] - X @ b) / np.var(x[kk])
            print(f"     {lab:26s} n {kk.sum():5d}: exports = {b[0]:+.2f} {b[1]:+.2f} per sd wind {b[2]:+.2f} per sd sun; share of variance {100 * r2:.0f}%")
