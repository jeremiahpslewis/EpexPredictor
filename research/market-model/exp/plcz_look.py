"""Is the 2026 change in solid-fuel dispatch German only? Polish and Czech lignite and hard coal, same season in three years, against the margin over carbon cost in their own price."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); G = pd.read_parquet(f"{SP}/out/plcz_gen.parquet").reindex(D.index); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); eua = ed.reindex(loc.normalize().tz_localize(None)).values
md_ = loc.strftime("%m-%d"); seas = (md_ >= "08-01") & (md_ <= "11-09"); bands = [(-1e9, -20), (-20, 0), (0, 20), (20, 50), (50, 100), (100, 1e9)]
for name, col, pcol, ef, c0 in (("German lignite", None, "p_de", 1.0, 12.0), ("Polish lignite", "pl_lignite", "p_pl", 1.0, 12.0), ("Czech lignite", "cz_lignite", "p_cz", 1.0, 12.0), ("German hard coal", None, "p_de", 0.85, 25.0), ("Polish hard coal", "pl_hcoal", "p_pl", 0.85, 25.0), ("Czech hard coal", "cz_hcoal", "p_cz", 0.85, 25.0)):
    g = (D["g_de_lignite"] if name == "German lignite" else D["g_de_coal"] if name == "German hard coal" else G[col])
    if g.notna().sum() < 1000: print(name, "no data"); continue
    avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24); u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3).values; m = (D[pcol] - ef * eua - c0).values
    print(f"\n{name}: median utilisation (hours) by margin of the zone's price over carbon cost + {c0:.0f}, Aug-Nov; season mean output and maximum GW")
    print("            " + "".join(f"{('<-20' if a < -1e8 else ('>100' if b > 1e8 else f'{a}..{b}')):>13s}" for a, b in bands))
    for y in (2024, 2025, 2026):
        k = seas & (loc.year == y) & np.isfinite(u) & np.isfinite(m); row = ""
        for a, b in bands:
            kk = k & (m >= a) & (m < b); row += f"{np.median(u[kk]):6.2f} ({kk.sum():3d})" if kk.sum() >= 15 else "      .      "
        print(f"   {y}     {row}   | {g[seas & (loc.year == y)].mean():.1f} / {g[seas & (loc.year == y)].max():.1f}; price {D[pcol][seas & (loc.year == y)].mean():.0f}")
