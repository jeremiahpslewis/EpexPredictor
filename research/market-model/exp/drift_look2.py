"""Is plant output a stable function of TWO prices: the hour's margin (dispatch between minimum load and full) and the day's margin (whether units are on at all)?
Median utilisation in cells of (hour margin, day margin), early periods against Aug-Nov 2026."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet"); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
H = D.resample("1h").mean(); loc = H.index.tz_convert("Europe/Berlin"); day = loc.normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1)
H["eua"] = ed.reindex(day.tz_localize(None)).values; H["pbar"] = H["p_de"].groupby(day).transform("mean").values
# also the mean over the surrounding three days (yesterday, today, tomorrow) as a slower commitment signal
dm = H["p_de"].groupby(day).mean(); H["pbar3"] = dm.rolling(3, center=True, min_periods=1).mean().reindex(day).values
FUEL = {"lignite": ("g_de_lignite", 1.0, 0.0), "hard coal": ("g_de_coal", 0.85, 0.0), "gas": ("g_de_gas", 0.37, 1.85)}
ym = loc.strftime("%Y-%m"); A = (ym >= "2024-07") & (ym <= "2026-03"); B = (ym >= "2026-08")
hb = [-1e9, -60, -30, -10, 10, 30, 60, 120, 1e9]; db = [-1e9, -10, 15, 40, 80, 1e9]
for name, (col, ef, hr) in FUEL.items():
    g = H[col]; avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24)
    u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3).values; cost = (ef * H["eua"] + hr * H["gas"]).values; mh = H["p_de"].values - cost; md = H["pbar"].values - cost
    print(f"\n{name}: median utilisation (count). rows: day-mean price minus cost; columns: hour price minus cost [EUR/MWh]. First line of each row = Jul 2024-Mar 2026, second = Aug-Nov 2026")
    print("            " + "".join(f"{('<' + str(hb[1])) if i == 0 else ('>' + str(hb[i])) if i == len(hb) - 2 else f'{hb[i]}..{hb[i + 1]}':>13s}" for i in range(len(hb) - 1)))
    for j in range(len(db) - 1):
        for lab, per in (("early", A), ("late ", B)):
            row = ""
            for i in range(len(hb) - 1):
                k = per & (mh >= hb[i]) & (mh < hb[i + 1]) & (md >= db[j]) & (md < db[j + 1]) & np.isfinite(u)
                row += f"{np.median(u[k]):7.2f} ({k.sum():3d})" if k.sum() >= 15 else "        .    "
            print(f"{('<' + str(db[1])) if j == 0 else ('>' + str(db[j])) if j == len(db) - 2 else f'{db[j]}..{db[j + 1]}':>7s} {lab}" + row)
