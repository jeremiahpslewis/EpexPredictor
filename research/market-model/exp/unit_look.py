"""Lignite unit by unit (ENTSO-E actual generation per unit): in the hours when the fleet is part-loaded, are units off or throttled, and at what price?"""
import sys, glob, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
fuel = sys.argv[1] if len(sys.argv) > 1 else "lignite"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
for t0 in ("2025-10-06", "2026-10-05"):
    parts = []
    for f in sorted(glob.glob(f"{SP}/entsoe/units/*_{fuel}_{t0}.parquet")):
        x = pd.read_parquet(f); x = x[[c for c in x.columns if "Consumption" not in c]]; x.columns = [c.split(" | ")[0] for c in x.columns]; parts.append(x.resample("1h").mean())
    if not parts: print(t0, "no files"); continue
    U = pd.concat(parts, axis=1).fillna(0.0) / 1000.0; U = U.loc[:, U.max() > 0.05]; U.index = U.index.tz_convert("UTC")
    p = D["p_de"].reindex(U.index); tot = D[{"lignite": "g_de_lignite", "coal": "g_de_coal", "gas": "g_de_gas"}[fuel]].reindex(U.index); eua = float(E.loc[:t0].iloc[-1]); umax = U.max()
    on = U > 0.15 * umax; load = (U / umax).where(on)
    print(f"\n=== {fuel}, two weeks from {t0}: {U.shape[1]} units reporting, sum of unit maxima {umax.sum():.1f} GW; their total output mean {U.sum(1).mean():.1f} GW against {tot.mean():.1f} GW for the whole fuel; carbon {eua:.0f} EUR/t; price mean {p.mean():.0f}")
    print("   by price band: hours | units on (of %d) | mean load of the units that are on | total output of reporting units GW" % U.shape[1])
    for a, b in ((-500, 60), (60, 90), (90, 110), (110, 130), (130, 160), (160, 200), (200, 1000)):
        k = ((p > a) & (p <= b)).values
        if k.sum() >= 5: print(f"   {a:5d}..{b:4d} EUR/MWh: {k.sum():4d} | {on[k].sum(1).mean():5.1f} | {np.nanmean(load[k].values):.2f} | {U[k].sum(1).mean():5.1f}")
    # per unit: maximum, share of hours on, mean load when on, and load in the price band just above cost
    band = ((p > eua + 20) & (p <= eua + 60)).values; hi = (p > eua + 80).values
    T = pd.DataFrame({"max GW": umax.round(2), "hours on %": (100 * on.mean()).round(0), "load when on": load.mean().round(2), f"load at price {eua + 20:.0f}-{eua + 60:.0f}": (U[band] / umax).mean().round(2), f"load at price > {eua + 80:.0f}": (U[hi] / umax).mean().round(2) if hi.sum() else np.nan}).sort_values("max GW", ascending=False)
    print(T.to_string()); U.to_parquet(f"{SP}/out/units_{fuel}_{t0}.parquet")
