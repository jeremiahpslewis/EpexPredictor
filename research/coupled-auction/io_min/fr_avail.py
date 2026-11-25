"""French nuclear unavailability series from ENTSO-E generation-unit notices. Usage: fr_avail.py SP OUT.parquet
Cleaning: drop cancelled documents; per notice (mrid) keep the latest revision; per unit and time, the unavailable MW is
the largest (nominal - available) among notices active then (overlapping notices are not summed). Two daily series at
noon UTC: 'all' (every notice, whenever published) and 'pregate' (outages already started by 12:00 the day before; the API keeps only the latest revision, so this approximates the gate view)."""
import sys, glob
import numpy as np, pandas as pd
SP, OUT = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/entsoe/fr_unavail_gen/*.parquet")) if pd.read_parquet(f).shape[0]])
d = d[d.plant_type.astype(str) == "Nuclear"]; d = d[d.docstatus.astype(str) != "Cancelled"]
d["created"] = pd.to_datetime(d["created_doc_time"], utc=True); d["start"] = pd.to_datetime(d["start"], utc=True); d["end"] = pd.to_datetime(d["end"], utc=True)
d["rev"] = pd.to_numeric(d["revision"], errors="coerce").fillna(0)
d["lost"] = (pd.to_numeric(d.nominal_power, errors="coerce") - pd.to_numeric(d.avail_qty, errors="coerce")).clip(lower=0)
d = d.sort_values(["mrid", "rev", "created"]); allrev = d                      # every revision, for the pre-gate view
d = d.drop_duplicates("mrid", keep="last")                                    # latest revision, for the 'all' view
days = pd.date_range(pd.Timestamp("2025-09-01", tz="UTC"), min(d.end.max().normalize(), pd.Timestamp("2026-11-16", tz="UTC")), freq="D") + pd.Timedelta(hours=12)
rows = []
for t in days:
    act = d[(d.start <= t) & (d.end > t)]
    g = t.normalize() - pd.Timedelta(hours=12)                                  # D-1 noon gate
    # the API returns only each notice's latest revision, so the gate view is approximated by outages that had
    # already started at the gate (their current end date comes from the latest revision; disclosed)
    pre = act[act.start <= g]
    rows.append(dict(t=t, all=act.groupby("production_resource_name").lost.max().sum() / 1000, pregate=pre.groupby("production_resource_name").lost.max().sum() / 1000,
                     units_out=int((act.groupby("production_resource_name").lost.max() > 0).sum())))
S = pd.DataFrame(rows).set_index("t"); S.to_parquet(OUT)
print("notices kept", len(d), " units", d.production_resource_name.nunique(), " nominal fleet (sum of unit nominal, GW)", round(d.groupby("production_resource_name").nominal_power.max().astype(float).sum() / 1000, 1))
print(S.resample("MS").mean().round(1).to_string())
