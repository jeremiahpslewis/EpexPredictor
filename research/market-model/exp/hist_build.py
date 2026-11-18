"""2021-2023 hourly frame for Germany: price, output by fuel, carbon (EEX auctions), monthly gas and coal (World Bank). Reads downloaded JSON only."""
import sys, json, glob, numpy as np, pandas as pd
src, eua_p, wb_p, out = sys.argv[1:5]; parts = []
for f in sorted(glob.glob(src + "/pp_*.json")):
    j = json.load(open(f)); t = pd.to_datetime(j["unix_seconds"], unit="s", utc=True); d = {}
    for pt in j["production_types"]:
        n = pt["name"].lower()
        for key, pat in (("lignite", "brown coal"), ("hcoal", "hard coal"), ("gas_out", "fossil gas"), ("load", "load"), ("resload", "residual load"), ("wind_on", "wind onshore"), ("wind_off", "wind offshore"), ("solar", "solar"), ("pump", "pumped storage consumption"), ("psgen", "pumped storage"), ("xb", "cross border"), ("nuclear", "nuclear")):
            if n == pat or (key in ("lignite", "hcoal", "gas_out") and pat in n and "coal-derived" not in n) or (key == "xb" and pat in n) or (key == "psgen" and n == "hydro pumped storage") or (key == "pump" and "pumped storage consumption" in n): d[key] = np.array(pt["data"], float)
    parts.append(pd.DataFrame(d, index=t))
    if f.endswith("2021a.json"): print([pt["name"] for pt in j["production_types"]])
G = pd.concat(parts).sort_index(); G = G[~G.index.duplicated()] / 1000.0; pr = []
for f in sorted(glob.glob(src + "/price_*.json")):
    j = json.load(open(f)); pr.append(pd.Series(j["price"], index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True), dtype=float))
p = pd.concat(pr).sort_index(); p = p[~p.index.duplicated()]
H = G.resample("1h").mean(); H["p"] = p.resample("1h").mean().reindex(H.index)
E = pd.read_parquet(eua_p)["eua"]; E.index = pd.to_datetime(E.index); ed = E.reindex(pd.date_range(E.index.min(), "2023-12-31", freq="D")).ffill().shift(1)
loc = H.index.tz_convert("Europe/Berlin"); H["eua"] = ed.reindex(loc.normalize().tz_localize(None)).values
W = pd.read_csv(wb_p, index_col=0, parse_dates=True); FX = {2021: 0.845, 2022: 0.951, 2023: 0.925}                     # EUR per USD, annual averages
mon = pd.DatetimeIndex(loc.tz_localize(None)).to_period("M").to_timestamp(); fx = np.array([FX.get(y, 0.92) for y in loc.year])
H["gas"] = W["Natural gas, Europe"].reindex(mon).values / 0.29307 * fx; H["coal"] = W["Coal, South African **"].reindex(mon).values / 6.978 * fx        # EUR per MWh thermal
print(H.resample("QS").mean()[["p", "gas", "coal", "eua", "lignite", "hcoal", "resload"]].round(1).to_string()); H.to_parquet(out)
