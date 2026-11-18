"""P11: elasticity of the price level to gas-fired cost by time scale (slow = 28-day trailing mean, fast = the day's deviation), hours at 25-50 GW."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import live_common as lc
SP = lc.SP; T = "2026-09-18"; d = np.load(f"{SP}/out/struct_pch/debug_{T}.npz"); z = np.load(f"{SP}/out/struct_pch/{T}.npz"); idx = lc.load()["DE"].index; t = idx[d["pos"]].tz_convert("Europe/Berlin"); eua = np.load(f"{SP}/out/eua_15min.npz")["eua"][d["pos"]]
gref = float(z["gref"]); shape = np.interp(d["x"], z["grid"], z["b_curve"]) + gref * np.interp(d["x"], z["grid"], z["h_curve"]); k = (d["x"] >= 25) & (d["x"] < 50) & (d["y"] > 10) & (shape > 10)
H = pd.DataFrame({"day": t.normalize()[k], "y": np.log(d["y"][k]) - np.log(shape[k]), "L": (1.85 * d["gas"] + 0.37 * eua)[k]}); D = H.groupby("day").agg(y=("y", "mean"), L=("L", "mean"), n=("y", "size")); D = D[D.n >= 6]
allL = pd.DataFrame({"day": t.normalize(), "L": 1.85 * d["gas"] + 0.37 * eua}).groupby("day")["L"].mean(); slow = np.log(allL).rolling(28, min_periods=14).mean(); D["slow"] = slow.reindex(D.index); D["fast"] = np.log(D["L"]) - D["slow"]; D = D.dropna()
X = np.column_stack([np.ones(len(D)), D.slow - D.slow.mean(), D.fast]); b = np.linalg.lstsq(X, D.y.values, rcond=None)[0]; wk = (np.arange(len(D)) // 7); rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    pick = rng.choice(wk.max() + 1, wk.max() + 1); ii = np.concatenate([np.flatnonzero(wk == w) for w in pick]); bs.append(np.linalg.lstsq(X[ii], D.y.values[ii], rcond=None)[0])
bs = np.array(bs); D["part_fast"] = D.y - b[0] - b[1] * (D.slow - D.slow.mean()); D["part_slow"] = D.y - b[2] * D.fast; D["mo"] = D.index.month + 12 * (D.index.year - 2025); D.to_parquet(f"{SP}/out/fit_p11.parquet")
print(f"{len(D)} days, hours at 25-50 GW; log gas-fired cost: slow part sd {D.slow.std():.3f}, fast part sd {D.fast.std():.3f}, correlation {np.corrcoef(D.slow, D.fast)[0, 1]:+.2f}")
print(f"elasticity of the day's price level to the SLOW part: {b[1]:.2f} [{np.percentile(bs[:, 1], 5):.2f}..{np.percentile(bs[:, 1], 95):.2f}]; to the FAST part: {b[2]:.2f} [{np.percentile(bs[:, 2], 5):.2f}..{np.percentile(bs[:, 2], 95):.2f}]  (week-block bootstrap, 5-95%)")
for win in (7, 14, 56):
    s2 = np.log(allL).rolling(win, min_periods=win // 2).mean().reindex(D.index); f2 = np.log(D["L"]) - s2; ok = s2.notna(); X2 = np.column_stack([np.ones(ok.sum()), (s2 - s2.mean())[ok], f2[ok]]); b2 = np.linalg.lstsq(X2, D.y.values[ok], rcond=None)[0]; print(f"   with a {win}-day mean as the slow part: slow {b2[1]:.2f}, fast {b2[2]:.2f} (fast sd {f2[ok].std():.3f})")
np.savez(f"{SP}/out/fit_p11.npz", b=b, lo=np.percentile(bs, 5, 0), hi=np.percentile(bs, 95, 0))
