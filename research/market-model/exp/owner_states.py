"""Owner-level supply states from SMARD production per plant block (published five days late, so everything is built from days <= d-7).
Per owner group and day d: output in the hours when the fuel was clearly in the money (margin 30..120 EUR/MWh), mean over the 14 days ending d-7, and the
14-day maximum of the group's hourly output. Writes out/owner_states.parquet on the hourly grid of the model data."""
import os, json, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index); loc = D.index.tz_convert("Europe/Berlin"); day = loc.normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); eua = pd.Series(ed.reindex(day.tz_localize(None)).values, index=D.index); p = D["p_de"]; gas = D["gas"]
W = pd.read_csv(f"{SP}/out/wb_coal.csv", index_col=0, parse_dates=True)["Coal, South African **"] / 6.978 * 0.92; mon = pd.DatetimeIndex(loc.tz_localize(None)).to_period("M").to_timestamp() - pd.DateOffset(months=1); coal = pd.Series(W.reindex(W.index.union(mon.unique())).sort_index().ffill().reindex(mon).values, index=D.index)
COST = {"Braunkohle": 12.0 + eua, "Steinkohle": 4.0 + 2.6 * coal + 0.85 * eua, "Erdgas": 3.0 + 1.85 * gas + 0.37 * eua}
GRP = {"Lausitz Energie Kraftwerke AG": "leag", "RWE Power AG": "rwe", "RWE Generation SE": "rwe", "EnBW Energie Baden-Württemberg AG": "enbw", "Saale Energie GmbH": "saale"}; FN = {"Braunkohle": "lig", "Steinkohle": "coal", "Erdgas": "gas"}
sel = json.load(open(f"{SP}/ext/smard/blocks_sel.json")); G = {}; CAP = {}
for b in sel:
    fn = f"{SP}/ext/smard/blocks/{b['id']}.parquet"
    if not os.path.exists(fn) or "etrieb" not in b["status"]: continue
    s = pd.read_parquet(fn)["mw"].reindex(D.index) / 1000.0
    if (s > 0).sum() < 500: continue
    o = GRP.get(b["company"], "other"); o = o if not (b["fuel"] == "Steinkohle" and o not in ("enbw",)) else "other"; o = o if not (b["fuel"] == "Erdgas" and o != "rwe") else "other"; key = f"{FN[b['fuel']]}_{o}"
    G[key] = G.get(key, 0) + s.fillna(0); CAP[key] = CAP.get(key, 0) + float(np.nanpercentile(s[s > 0], 99.5))
    G.setdefault(key + "_n", 0); G[key + "_n"] = G[key + "_n"] + s.notna().astype(float)
out = {}; LAG = 7
for key in [k for k in G if not k.endswith("_n")]:
    g = G[key].where(G[key + "_n"] > 0); fuel = {"lig": "Braunkohle", "coal": "Steinkohle", "gas": "Erdgas"}[key.split("_")[0]]; m = p - COST[fuel]; inm = (m >= 30) & (m < 120) & g.notna()
    num = g.where(inm).groupby(day).sum().rolling(14, min_periods=3).sum(); den = inm.groupby(day).sum().rolling(14, min_periods=3).sum(); z = (num / den.where(den >= 12)).ffill().shift(LAG); mx = g.groupby(day).max().rolling(14, min_periods=7).max().shift(LAG)
    out[f"zin_{key}"] = z.reindex(day).values; out[f"max_{key}"] = mx.reindex(day).values; out[f"raw_{key}"] = g.values
    print(f"{key:12s} capacity {CAP[key]:4.1f} GW; mean output {g.mean():4.1f}; in-the-money output, mean by half-year: " + "  ".join(f"{h}: {np.nanmean(z.reindex(day).values[(loc.strftime('%Y') + 'H' + np.where(loc.month <= 6, '1', '2')) == h]):.1f}" for h in ("2024H2", "2025H1", "2025H2", "2026H1", "2026H2")))
O = pd.DataFrame(out, index=D.index); O.to_parquet(f"{SP}/out/owner_states.parquet"); json.dump(CAP, open(f"{SP}/out/owner_caps.json", "w")); print("blocks cover", O["raw_lig_rwe"].notna().mean().round(2), "of hours; first", O["raw_lig_rwe"].dropna().index.min())
# coverage check: owner sums against the fleet series
for f, col in (("lig", "g_de_lignite"), ("coal", "g_de_coal")):
    tot = sum(O[c] for c in O.columns if c.startswith(f"raw_{f}_")); k = tot.notna() & D[col].notna() & (D.index >= "2024-08-01"); print(f"  {f}: blocks sum to {tot[k].mean():.1f} GW of the fleet's {D[col][k].mean():.1f} GW; correlation {np.corrcoef(tot[k], D[col][k])[0, 1]:.3f}")
