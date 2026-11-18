"""Data for the coupled-market model: per zone price [EUR/MWh] and published residual load [GW]; gas; realised output by
fuel (for measured availability and as a check); scheduled exchanges per border (check only). Index: the 15-minute grid."""
import json, glob, numpy as np, pandas as pd, live_common as lc, dense_common as dc
SP = lc.SP; F = lc.load(); idx = F["DE"].index; out = pd.DataFrame(index=idx)
ZN = ["de", "fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]
for z, src in (("de", "DE"), ("fr", "FR"), ("nl", "NL"), ("be", "BE"), ("at", "AT"), ("dk", "DK1")): out[f"p_{z}"] = F[src]["price"].to_numpy(float) * 10
for z in ("pl", "cz", "ch"):
    d = json.load(open(f"{SP}/ext/ecz/price_{z.upper()}.json")); s = pd.Series(pd.to_numeric(pd.Series(d["price"]), errors="coerce").to_numpy(), index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)); s = s[~s.index.duplicated()]
    out[f"p_{z}"] = s.reindex(idx.union(s.index)).sort_index().ffill(limit=3).reindex(idx).to_numpy()
nb = pd.read_parquet(f"{SP}/out/nb_rl.parquet").reindex(idx); out["rl_de"] = dc.tso(idx)["rl"].to_numpy(float)
for z in ZN[1:]: out[f"rl_{z}"] = nb[f"rl_{z}"].to_numpy(float)
out["gas"] = F["DE"]["gasprice"].to_numpy(float)
def pp(pattern):
    fr = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f)); fr.append(pd.DataFrame({p["name"]: pd.to_numeric(pd.Series(p["data"]), errors="coerce").to_numpy() for p in d["production_types"]}, index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)))
    M = pd.concat(fr).sort_index(); M = M[~M.index.duplicated()]; return (M.reindex(idx.union(M.index)).sort_index().ffill(limit=3).reindex(idx)) / 1000
De = pp(f"{SP}/ext/ecpp/pp_[abcd]*.json"); Fr = pp(f"{SP}/ext/ecpp/pp_f*.json")
for c, n in (("g_de_lignite", "Fossil brown coal / lignite"), ("g_de_coal", "Fossil hard coal"), ("g_de_gas", "Fossil gas"), ("g_de_pump", "Hydro pumped storage consumption"), ("g_de_psgen", "Hydro pumped storage"), ("g_de_trade", "Cross border electricity trading")): out[c] = De[n].to_numpy()
for c, n in (("g_fr_nuclear", "Nuclear"), ("g_fr_gas", "Fossil gas"), ("g_fr_reservoir", "Hydro water reservoir"), ("g_fr_ror", "Hydro Run-of-River")): out[c] = Fr[n].to_numpy()
d = json.load(open(f"{SP}/ext/ecz/cbet_a.json")); ci = pd.to_datetime(d["unix_seconds"], unit="s", utc=True); NAME = {"Austria": "at", "Belgium": "be", "Czech Republic": "cz", "Denmark": "dk", "France": "fr", "Netherlands": "nl", "Poland": "pl", "Switzerland": "ch", "sum": "sum"}
for c in d["countries"]:
    if c["name"] in NAME: out[f"x_{NAME[c['name']]}"] = pd.Series(pd.to_numeric(pd.Series(c["data"]), errors="coerce").to_numpy(), index=ci)[lambda s: ~s.index.duplicated()].reindex(idx).to_numpy()
out.to_parquet(f"{SP}/out/star_data.parquet")
print(out.notna().mean().round(2).to_dict())
r = out.iloc[-96 * 540:]; print("last 540 days: share of hours with |p_z - p_de| <= 1.5 EUR/MWh / neighbour dearer / cheaper; corr(exchange with z, price difference)")
for z in ZN[1:]:
    dp = r[f"p_{z}"] - r["p_de"]; x = r.get(f"x_{z}"); print(f"   {z}: {100 * (dp.abs() <= 1.5).mean():.0f}% / {100 * (dp > 1.5).mean():.0f}% / {100 * (dp < -1.5).mean():.0f}%   exchange mean {x.mean():+.2f} GW, 5%..95% {x.quantile(.05):+.1f}..{x.quantile(.95):+.1f}; mean when neighbour dearer {x[dp > 1.5].mean():+.2f}, cheaper {x[dp < -1.5].mean():+.2f}")
print("   sum of exchanges vs public_power trade: corr", round(float(np.corrcoef(r["x_sum"].fillna(0), r["g_de_trade"].fillna(0))[0, 1]), 2), "; FR nuclear output 5%..95%:", r["g_fr_nuclear"].quantile(.05).round(1), r["g_fr_nuclear"].quantile(.95).round(1), "; DE lignite", r["g_de_lignite"].quantile(.05).round(1), r["g_de_lignite"].quantile(.95).round(1), "coal", r["g_de_coal"].quantile(.05).round(1), r["g_de_coal"].quantile(.95).round(1), "gas", r["g_de_gas"].quantile(.05).round(1), r["g_de_gas"].quantile(.95).round(1))
