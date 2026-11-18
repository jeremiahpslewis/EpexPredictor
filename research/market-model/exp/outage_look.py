"""Published unavailability of German generation units (ENTSO-E) by fuel: does it account for the capacity that stays idle at high margins?"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
parts = []
for f in ("unavail_gen_DE_LU", "bt_unavail_prod_DE_LU"):
    d = pd.read_parquet(f"{SP}/entsoe/{f}.parquet").reset_index(); d["src"] = f; parts.append(d)
    print(f, len(d), "created", d["created_doc_time"].min(), "..", d["created_doc_time"].max(), "| start", d["start"].min(), "..", d["start"].max(), "| end max", d["end"].max())
    print("   plant types:", d["plant_type"].value_counts().to_dict()); print("   docstatus:", d["docstatus"].value_counts(dropna=False).to_dict(), "| businesstype:", d["businesstype"].value_counts().to_dict())
    print("   records per start year-quarter:", d.groupby(d["start"].dt.tz_convert("UTC").dt.to_period("Q")).size().to_dict())
grid = pd.date_range("2024-03-01", "2026-11-09", freq="1h", tz="UTC")
FU = {"lignite": "Fossil Brown coal/Lignite", "hard coal": "Fossil Hard coal", "gas": "Fossil Gas"}
def series(d, ptype):
    d = d[d["plant_type"] == ptype]; d = d[d["revision"] == d.groupby("mrid")["revision"].transform("max")]; d = d[d["docstatus"].astype(str) != "Cancelled"].copy()
    d["un"] = (d["nominal_power"] - pd.to_numeric(d["avail_qty"], errors="coerce")).clip(lower=0); tot = np.zeros(len(grid)); pl = np.zeros(len(grid)); nunit = d["production_resource_psr_name"].nunique()
    for unit, g in d.groupby("production_resource_psr_name"):
        u = np.zeros(len(grid)); up = np.zeros(len(grid))
        for s, e, un, bt in zip(g["start"], g["end"], g["un"], g["businesstype"]):
            i, j = grid.searchsorted(s.tz_convert("UTC")), grid.searchsorted(e.tz_convert("UTC"))
            if j > i and np.isfinite(un):
                u[i:j] = np.maximum(u[i:j], un)
                if bt == "Planned maintenance": up[i:j] = np.maximum(up[i:j], un)
        tot += u; pl += up
    return pd.Series(tot / 1000, index=grid), pd.Series(pl / 1000, index=grid), nunit, d
G = parts[0]; out = {}
for name, pt in FU.items():
    s, p, n, d = series(G, pt); out[name] = s; out[name + " planned"] = p
    cap = d.groupby("production_resource_psr_name")["nominal_power"].max().sum() / 1000
    print(f"\n{name}: {n} units ever reported, their nominal power {cap:.1f} GW; unavailable GW by month (mean):")
    m = s.resample("MS").mean(); print("   " + "  ".join(f"{k.strftime('%y-%m')} {v:.1f}" for k, v in m.items()))
O = pd.DataFrame(out); O.to_parquet(f"{SP}/out/de_outage_hourly.parquet")
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean().reindex(grid)
s = slice("2026-10-05", "2026-10-19"); dd = pd.concat([D.loc[s, ["p_de", "g_de_lignite", "g_de_coal", "g_de_gas"]], O.loc[s]], axis=1).resample("1D").mean()
pd.set_option("display.width", 250); print("\ndaily means, 5-19 Oct 2026:"); print(dd.round(1).to_string())
