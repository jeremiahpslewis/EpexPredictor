"""
Zone A's fleet from public data dated before the window (notes 21.162). Usage: de_fleet.py DATA_DIR FIRST LAST OUT.json
- Market fleet per fuel: each unit's largest hourly output in the 12 months before FIRST (ENTSO-E actual generation per
  unit, units of 100 MW and more), summed over units that produced more than 50 MW at least once. Gas is scaled up by the
  units' share of total gas output (the small units are missing from the unit data).
- Published unavailability per fuel and quarter of each delivery day (ENTSO-E generation-unit unavailability): only
  documents created before noon the day before, the latest revision of each document, cancelled ones dropped, and per
  unit the largest unavailability among overlapping documents (no double counting). Counted only for units in the
  market fleet (matched by EIC code via the unit data's names is not possible, so by unit name; unmatched units listed).
"""
import json, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

SP, FIRST, LAST, OUT = Path(sys.argv[1]), pd.Timestamp(sys.argv[2], tz="Europe/Berlin"), pd.Timestamp(sys.argv[3], tz="Europe/Berlin"), Path(sys.argv[4])
X = pd.read_parquet(SP / "out/recon_hourly.parquet")
lo, hi = (FIRST - pd.DateOffset(months=12)).strftime("%Y-%m"), (FIRST - pd.DateOffset(months=1)).strftime("%Y-%m")
FUELS = {"lignite": ("lig", "Fossil Brown coal/Lignite"), "coal": ("coal", "Fossil Hard coal"), "gas": ("gasg", "Fossil Gas")}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
fleet, names = {}, {}
for fuel, (col, _) in FUELS.items():
    frames = []
    for f in sorted(glob.glob(str(SP / f"entsoe/units_all/DE_*_{fuel}_*.parquet"))):
        m = re.search(r"_(\d{4}-\d{2})\.parquet", f).group(1)
        if lo <= m <= hi:
            d = pd.read_parquet(f); d.columns = [c.split(" | ")[0] for c in d.columns]; frames.append(d)
    U = pd.concat(frames, axis=1); U = U.T.groupby(level=0).max().T; U = U.groupby(U.index).max()
    umax = U.max(); act = umax[umax > 50]
    tot = U[act.index].sum(axis=1, min_count=1).dropna(); tot.index = tot.index.tz_convert("UTC")
    r = X[col].reindex(tot.index); ok = r.notna()
    share = float((tot[ok] / 1000).sum() / r[ok].sum())
    fleet[fuel] = dict(units=int(len(act)), unit_MW=float(act.sum()), coverage=round(share, 3), fleet_MW=float(act.sum() / (share if fuel == "gas" else 1.0)))
    names[fuel] = {norm(n): float(v) for n, v in act.items()}

O = pd.read_parquet(SP / "entsoe/unavail_gen_DE_LU.parquet").reset_index()
O = O[O.plant_type.isin([v[1] for v in FUELS.values()])]
days = pd.date_range(FIRST, LAST, freq="D"); out = {"fleet": fleet, "days": {}, "unmatched": {}}
for d in days:
    gate = (d - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=12)
    idx = pd.date_range(d, (d + pd.DateOffset(days=1)).normalize(), freq="15min", inclusive="left")
    o = O[O.created_doc_time < gate].sort_values(["mrid", "revision", "created_doc_time"]).groupby("mrid").tail(1)
    o = o[(o.docstatus != "Cancelled") & (o.start < idx[-1] + pd.Timedelta(minutes=15)) & (o.end > idx[0])]
    day = {}
    for fuel, (_, pt) in FUELS.items():
        oo = o[o.plant_type == pt]; per_unit = {}
        for _, row in oo.iterrows():
            key = norm(str(row.production_resource_psr_name) or str(row.production_resource_name))
            match = next((n for n in names[fuel] if n and (n in key or key in n)), None)
            if match is None:
                out["unmatched"].setdefault(fuel, set()).add(str(row.production_resource_psr_name)); continue
            lost = np.where((idx >= row.start) & (idx < row.end), min(float(row.nominal_power) - float(row.avail_qty), names[fuel][match]), 0.0)
            per_unit[match] = np.maximum(per_unit.get(match, 0.0), lost)
        day[fuel] = (np.sum(list(per_unit.values()), axis=0) if per_unit else np.zeros(len(idx))).tolist()
    out["days"][str(d.date())] = day
out["unmatched"] = {k: sorted(v) for k, v in out["unmatched"].items()}
json.dump(out, open(OUT, "w"))
print(json.dumps(fleet, indent=1))
for d, v in out["days"].items(): print(d, {k: round(float(np.mean(x)) / 1000, 2) for k, x in v.items()})
print("unmatched outage units:", {k: len(v) for k, v in out["unmatched"].items()}, {k: v[:6] for k, v in out["unmatched"].items()})
