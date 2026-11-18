"""Low load (published residual load 5-15 GW and 15-25 GW): why did the German price relative to gas-fired cost rise from late 2025?  By half-year."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); D["eua"] = ed.reindex(loc.normalize().tz_localize(None)).values
D["L"] = 1.85 * D["gas"] + 0.37 * D["eua"]; half = np.array([f"{y}H{1 if m <= 6 else 2}" for y, m in zip(loc.year, loc.month)]); hr = loc.hour.values
D["ps"] = D["g_de_psgen"] + D["g_de_pump"]; NB = ["fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]; D["pbar"] = D["p_de"].groupby(loc.normalize()).transform("mean").values
for a, b in ((5, 15), (15, 25)):
    print(f"\npublished residual load {a}-{b} GW, by half-year: hours | price/L median | share of hours 10-16h | net exports GW (scheduled) | share of hours within 2 EUR of at least four neighbours | median neighbour price / L: fr nl at ch dk | pumped storage net GW | thermal above 2% floor GW | day mean price / L | lignite+coal output")
    for h in sorted(set(half)):
        k = (half == h) & (D["rl_de"] >= a).values & (D["rl_de"] < b).values & D["p_de"].notna().values
        if k.sum() < 100: continue
        x = D[k]; near = sum((x[f"p_{z}"] - x["p_de"]).abs() <= 2 for z in NB)
        print(f"  {h}: {k.sum():4d} | {np.median(x['p_de'] / x['L']):5.2f} | {100 * np.mean((hr[k] >= 10) & (hr[k] < 16)):3.0f}% | {-x['x_sum'].mean():+5.1f} | {100 * np.mean(near >= 4):3.0f}% | " + " ".join(f"{np.median(x['p_' + z] / x['L']):5.2f}" for z in ("fr", "nl", "at", "ch", "dk")) + f" | {x['ps'].mean():+5.1f} | {(x['g_de_lignite'] + x['g_de_coal'] + x['g_de_gas']).mean():5.1f} | {np.median(x['pbar'] / x['L']):5.2f} | {(x['g_de_lignite'] + x['g_de_coal']).mean():5.1f}")
