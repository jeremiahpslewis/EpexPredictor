"""
Hourly two-zone panel for the minimal structural model (io_min). Usage: build_panel.py OLD_SP OUT_DIR
Reads only from OLD_SP (the earlier session's data); writes OUT_DIR/panel.parquet and OUT_DIR/fleet.json.
Zone DE = DE_LU. Zone N = the ten SDAC neighbours coupled to DE_LU (FR NL BE AT CZ PL DK_1 DK_2 SE_4 NO_2); CH is an
exogenous schedule. Information set at the auction (noon D-1): gas = THE price of D-1, carbon = last EUA before noon
D-1, unavailability documents created before noon D-1 (same rule as de_fleet.py), published load / wind / solar forecasts.
"""
import json, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
T0, T1 = pd.Timestamp("2025-10-01", tz="Europe/Berlin"), pd.Timestamp("2026-11-08", tz="Europe/Berlin")
Z = ["FR", "NL", "BE", "AT", "CZ", "PL", "DK_1", "DK_2", "SE_4", "NO_2"]
D = SP / "entsoe/neighbours/derived"
hr = lambda s: s.resample("1h").mean().tz_convert("UTC")

H = pd.read_parquet(SP / "out/recon_hourly.parquet")
M = pd.read_parquet(SP / "out/market_hourly.parquet")
idx = pd.date_range(T0.tz_convert("UTC"), (T1 + pd.Timedelta(days=1)).tz_convert("UTC"), freq="1h", inclusive="left")
X = pd.DataFrame(index=idx)
X["p"] = H.p.reindex(idx)
price = pd.concat({z: hr(pd.read_parquet(D / f"{z}_price_15min.parquet").iloc[:, 0]) for z in Z}, axis=1).reindex(idx)
load = pd.concat({z: hr(pd.read_parquet(D / f"{z}_load_15min.parquet").iloc[:, 0]) for z in Z}, axis=1).reindex(idx)
ws = pd.concat({z: hr(pd.read_parquet(D / f"{z}_ws_15min.parquet").fillna(0).sum(axis=1)) for z in Z}, axis=1).reindex(idx)
X["pN"] = (price * load).sum(axis=1, min_count=10) / load.sum(axis=1, min_count=10)
X["DN"] = (load - ws).sum(axis=1, min_count=10) / 1000                        # GW, neighbours' published residual load
ne = pd.read_parquet(D / "DE_LU_sched_net_export_15min.parquet").resample("1h").mean(); ne.index = ne.index.tz_convert("UTC")
X["m"] = -ne[Z].sum(axis=1, min_count=10).reindex(idx) / 1000                  # GW, DE scheduled net import from SDAC10
X["ch"] = ne["CH"].reindex(idx) / 1000                                         # GW, DE scheduled net export to CH (exogenous)
for c in ["rl", "rlreal", "lig", "coal", "gasg", "other", "psgen", "pump"]: X[c] = H[c].reindex(idx)
X["DDE"] = X.rl - X.other - X.psgen - X.pump + X.ch                            # GW, residual load to be met by thermal + imports
X["gap"] = X.rlreal - X.rl
# costs known at noon D-1
day = idx.tz_convert("Europe/Berlin").tz_localize(None).normalize()
gas_d = M.gas.copy(); gas_d.index = gas_d.index.tz_convert("Europe/Berlin").tz_localize(None).normalize(); gas_d = gas_d.groupby(level=0).mean()
X["Pg"] = gas_d.reindex(day - pd.Timedelta(days=1)).to_numpy()
eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
X["tau"] = eua.reindex(day - pd.Timedelta(days=1), method="ffill").to_numpy()
w = pd.read_json(SP / "appdata/weather_v2_DE_LU.json.gz", compression="gzip")
if pd.api.types.is_integer_dtype(w.index.dtype): w.index = pd.to_datetime(w.index, unit="ms", utc=True)
elif w.index.tz is None: w.index = w.index.tz_localize("UTC")
t = w[[f"temp_{i}" for i in range(6)]].mean(axis=1).tz_convert("Europe/Berlin")
tday = t.groupby(t.index.tz_localize(None).normalize()).mean()
X["hdd"] = (15.0 - tday).clip(lower=0).reindex(day).to_numpy()
X["hour"] = idx.tz_convert("Europe/Berlin").hour; X["day"] = day

# fleet (12 months before T0) and published unavailability per fuel and hour, as de_fleet.py
FUELS = {"lignite": ("lig", "Fossil Brown coal/Lignite"), "coal": ("coal", "Fossil Hard coal"), "gas": ("gasg", "Fossil Gas")}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
lo, hi = (T0 - pd.DateOffset(months=12)).strftime("%Y-%m"), (T0 - pd.DateOffset(months=1)).strftime("%Y-%m")
fleet, names = {}, {}
for fuel, (col, _) in FUELS.items():
    fr = []
    for f in sorted(glob.glob(str(SP / f"entsoe/units_all/DE_*_{fuel}_*.parquet"))):
        mo = re.search(r"_(\d{4}-\d{2})\.parquet", f).group(1)
        if lo <= mo <= hi:
            d = pd.read_parquet(f); d.columns = [c.split(" | ")[0] for c in d.columns]; fr.append(d)
    U = pd.concat(fr, axis=1); U = U.T.groupby(level=0).max().T; U = U.groupby(U.index).max()
    umax = U.max(); act = umax[umax > 50]
    tot = U[act.index].sum(axis=1, min_count=1).dropna(); tot.index = tot.index.tz_convert("UTC")
    r = H[col].reindex(tot.index); ok = r.notna()
    share = float((tot[ok] / 1000).sum() / r[ok].sum())
    fleet[fuel] = dict(units=int(len(act)), unit_MW=float(act.sum()), coverage=round(share, 3),
                       fleet_MW=float(act.sum() / (share if fuel == "gas" else 1.0)))
    names[fuel] = {norm(n): float(v) for n, v in act.items()}
O = pd.read_parquet(SP / "entsoe/unavail_gen_DE_LU.parquet").reset_index()
O = O[O.plant_type.isin([v[1] for v in FUELS.values()])]
un = {f: np.zeros(len(idx)) for f in FUELS}
loc = idx.tz_convert("Europe/Berlin")
for dd in pd.date_range(T0, T1, freq="D"):
    gate = (dd - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=12)
    sel = np.where((loc >= dd) & (loc < (dd + pd.DateOffset(days=1)).normalize()))[0]
    hrs = loc[sel]
    o = O[O.created_doc_time < gate].sort_values(["mrid", "revision", "created_doc_time"]).groupby("mrid").tail(1)
    o = o[(o.docstatus != "Cancelled") & (o.start < hrs[-1] + pd.Timedelta(hours=1)) & (o.end > hrs[0])]
    for fuel, (_, pt) in FUELS.items():
        per = {}
        for _, row in o[o.plant_type == pt].iterrows():
            key = norm(str(row.production_resource_psr_name) or str(row.production_resource_name))
            mt = next((n for n in names[fuel] if n and (n in key or key in n)), None)
            if mt is None: continue
            lost = np.where((hrs >= row.start) & (hrs < row.end), min(float(row.nominal_power) - float(row.avail_qty), names[fuel][mt]), 0.0)
            per[mt] = np.maximum(per.get(mt, 0.0), lost)
        if per: un[fuel][sel] = np.sum(list(per.values()), axis=0)
for fuel in FUELS: X[f"K_{fuel}"] = (fleet[fuel]["fleet_MW"] - un[fuel]) / 1000   # GW, available before derating
n0 = len(X); X = X.dropna()
print(f"rows {len(X)} of {n0}"); print(json.dumps(fleet, indent=1))
print(X.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
X.to_parquet(OUT / "panel.parquet"); json.dump(fleet, open(OUT / "fleet.json", "w"), indent=1)
