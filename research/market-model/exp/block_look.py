"""Plant blocks one by one (SMARD production per block): how did each lignite / hard coal block's response to price change from 2025 to 2026, same season,
and does the change line up with owner size (market power), with age and so carbon intensity (a carbon-like cost), or with nothing (a common constraint)?
Per block and year: loading (output / block maximum) by margin band, and the margin at which loading is half its plateau (logistic fit)."""
import sys, os, json, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index); loc = D.index.tz_convert("Europe/Berlin")
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); eua = pd.Series(ed.reindex(loc.normalize().tz_localize(None)).values, index=D.index)
W = pd.read_csv(f"{SP}/out/wb_coal.csv", index_col=0, parse_dates=True)["Coal, South African **"] / 6.978 * 0.92; mon = pd.DatetimeIndex(loc.tz_localize(None)).to_period("M").to_timestamp() - pd.DateOffset(months=1)
coal = pd.Series(W.reindex(W.index.union(mon.unique())).sort_index().ffill().reindex(mon).values, index=D.index); p = D["p_de"]; gas = D["gas"]
COST = {"Braunkohle": 12.0 + 1.0 * eua, "Steinkohle": 4.0 + 2.6 * coal + 0.85 * eua, "Erdgas": 3.0 + 1.85 * gas + 0.37 * eua}
sel = json.load(open(f"{SP}/ext/smard/blocks_sel.json")); md = loc.strftime("%m-%d"); S1 = sys.argv[1] if len(sys.argv) > 1 else "04-01"; S2 = sys.argv[2] if len(sys.argv) > 2 else "11-08"; seas = (md >= S1) & (md <= S2); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -50, 50)))
BIG = {"Lausitz Energie Kraftwerke AG": "LEAG", "RWE Power AG": "RWE", "EnBW Energie Baden-Württemberg AG": "EnBW", "Saale Energie GmbH": "Saale (EPH)"}
rows = []
for b in sel:
    fn = f"{SP}/ext/smard/blocks/{b['id']}.parquet"
    if not os.path.exists(fn): continue
    s = pd.read_parquet(fn)["mw"].reindex(D.index)
    if s.notna().sum() < 2000 or s.max() < 50: continue
    cap = np.nanpercentile(s[s > 0], 99.5) if (s > 0).sum() > 100 else np.nan; m = (p - COST[b["fuel"]]); r = dict(id=b["id"], fuel=b["fuel"], plant=b["plant"], name=b["name"], company=b["company"], owner=BIG.get(b["company"], "other"), power=b["power"], cap=cap, year=b["year"], chp=b["chp"], status=b["status"][:22])
    for y in (2025, 2026):
        k = seas & (loc.year == y) & s.notna().values & m.notna().values
        if k.sum() < 1500: continue
        ld = (s[k] / cap).clip(0, 1.2).values; mm = m[k].values; r[f"on{y}"] = float(np.mean(ld > 0.15)); r[f"mean{y}"] = float(ld.mean())
        for a_, b_ in ((0, 30), (30, 60), (60, 120), (120, 1e9)):
            kk = (mm >= a_) & (mm < b_); r[f"l{y}_{a_}"] = float(ld[kk].mean()) if kk.sum() >= 40 else np.nan
        plateau = np.nanmean(ld[mm > np.nanpercentile(mm, 90)]); r[f"plat{y}"] = float(plateau); best = (1e9, np.nan, np.nan)
        if plateau > 0.25:
            for w_ in (8, 14, 22, 35):
                for m0 in np.arange(-60, 201, 4.0):
                    e_ = np.mean(np.abs(ld - plateau * sg((mm - m0) / w_)))
                    if e_ < best[0]: best = (e_, m0, w_)
        r[f"m0_{y}"] = best[1]; r[f"w_{y}"] = best[2]
    rows.append(r)
T = pd.DataFrame(rows); T["shift"] = T["m0_2026"] - T["m0_2025"]; T["d30"] = T["l2026_30"] - T["l2025_30"]; T["d60"] = T["l2026_60"] - T["l2025_60"]; T.to_csv(f"{SP}/out/block_look.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
for fuel in ("Braunkohle", "Steinkohle", "Erdgas"):
    x = T[(T["fuel"] == fuel) & T["m0_2025"].notna() & T["m0_2026"].notna()].sort_values("shift")
    print(f"\n=== {fuel}: {len(x)} blocks with both years ({S1} to {S2}). Loading at margin 30-60 and 60-120 EUR/MWh over the fuel's cost, share of hours on, half-loading margin")
    for _, r in x.iterrows():
        print(f"  {str(r['plant'])[:22]:22s} {str(r['name'])[-9:]:9s} {r['owner']:11s} {str(r['company'])[:24]:24s} {r['cap']:5.0f} MW  built {r['year']}  CHP {r['chp']:4s} | 30-60: {r['l2025_30']:.2f} -> {r['l2026_30']:.2f} | 60-120: {r['l2025_60']:.2f} -> {r['l2026_60']:.2f} | on {r['on2025']:.2f} -> {r['on2026']:.2f} | half-loading margin {r['m0_2025']:+5.0f} -> {r['m0_2026']:+5.0f} (shift {r['shift']:+4.0f})")
    if len(x) >= 4:
        wgt = x["cap"].values; print(f"  capacity-weighted: shift {np.average(x['shift'], weights=wgt):+.0f} EUR/MWh; loading at 30-60: {np.average(x['l2025_30'].fillna(0), weights=wgt):.2f} -> {np.average(x['l2026_30'].fillna(0), weights=wgt):.2f}; at 60-120: {np.average(x['l2025_60'].fillna(0), weights=wgt):.2f} -> {np.average(x['l2026_60'].fillna(0), weights=wgt):.2f}")
        for lab, kk in (("large portfolios (LEAG, RWE, EnBW, Saale)", x["owner"] != "other"), ("other owners", x["owner"] == "other")):
            if kk.sum() >= 2: print(f"     {lab:42s} n {int(kk.sum()):2d}, {x['cap'][kk].sum() / 1000:4.1f} GW: shift {np.average(x['shift'][kk], weights=x['cap'][kk]):+4.0f}; loading 30-60: {np.average(x['l2025_30'][kk].fillna(0), weights=x['cap'][kk]):.2f} -> {np.average(x['l2026_30'][kk].fillna(0), weights=x['cap'][kk]):.2f}; 60-120: {np.average(x['l2025_60'][kk].fillna(0), weights=x['cap'][kk]):.2f} -> {np.average(x['l2026_60'][kk].fillna(0), weights=x['cap'][kk]):.2f}")
        yr = pd.to_numeric(x["year"], errors="coerce"); kk = yr.notna()
        if kk.sum() >= 5: print(f"     correlation of the shift with year built: {np.corrcoef(yr[kk], x['shift'][kk])[0, 1]:+.2f}; with block size: {np.corrcoef(x['cap'][kk], x['shift'][kk])[0, 1]:+.2f}; CHP yes n {int((x['chp'] == 'Ja').sum())}: shift {x['shift'][x['chp'] == 'Ja'].mean():+.0f}, CHP no n {int((x['chp'] != 'Ja').sum())}: {x['shift'][x['chp'] != 'Ja'].mean() if (x['chp'] != 'Ja').sum() else float('nan'):+.0f}")
print("\nblocks present in one year only or barely running:")
for _, r in T[T["m0_2025"].isna() | T["m0_2026"].isna()].iterrows(): print(f"  {r['fuel']:10s} {str(r['plant'])[:24]:24s} {r['owner']:8s} {r['cap']:5.0f} MW status {r['status']}: mean loading {r.get('mean2025', np.nan):.2f} -> {r.get('mean2026', np.nan):.2f}")
