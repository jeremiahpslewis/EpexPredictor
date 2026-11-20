"""The gap between realised and published residual load, against the price (notes 21.147)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
H = pd.read_parquet(f"{SP}/out/market_hourly.parquet").dropna(subset=["p", "rl", "rlreal"]); loc = H.index.tz_convert("Europe/Berlin")
H["gap"] = H.rlreal - H.rl; H["cell"] = pd.qcut(H.rl, 10, labels=False).astype(str) + "_" + loc.hour.astype(str); H["g_r"] = H.gap - H.groupby("cell").gap.transform("mean")
H["half"] = loc.year.astype(str) + np.where(loc.month <= 6, "H1", "H2")
bands = [-600, -20, 0, 1, 30, 60, 90, 120, 150, 200, 1000]; H["band"] = pd.cut(H.p, bands)
T = H.groupby("band").agg(hours=("g_r", "size"), gap=("g_r", "mean"), se=("g_r", lambda x: x.std() / np.sqrt(max(len(x) / 24, 1))))
print("gap realised - published residual load, GW, net of published-level x hour means (se assumes 24 hours per independent draw)"); print(T.round(2).to_string())
print("\nby half year, three price bands:"); H["b3"] = pd.cut(H.p, [-600, 0, 150, 1000], labels=["below 0", "0 to 150", "above 150"]); print(H.pivot_table(index="b3", columns="half", values="g_r", aggfunc="mean").round(2).to_string())
print(H.pivot_table(index="b3", columns="half", values="g_r", aggfunc="size").to_string())
# a linear slope where the bulk of hours are, 30 to 150 EUR/MWh
m = (H.p > 30) & (H.p < 150); b = np.polyfit(H.p[m], H.g_r[m], 1)[0]; print(f"\nlinear slope of the gap on price, 30-150 EUR/MWh: {10 * b:+.3f} GW per 10 EUR/MWh")
T.to_csv(f"{SP}/out/demand_gap.csv")
