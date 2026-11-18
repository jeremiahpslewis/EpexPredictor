import json, os, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
grid = pd.date_range("2026-08-25", "2026-11-12", freq="15min", tz="UTC")
def series(path, keep):
    """unavailable MW per quarter hour; latest revision per record, cancelled dropped, overlaps per unit not double counted"""
    d = pd.read_parquet(path).reset_index()
    d = d[d["revision"] == d.groupby("mrid")["revision"].transform("max")]
    d = d[(d["docstatus"] != "Cancelled") & keep(d)]
    d["un"] = (d["nominal_power"] - pd.to_numeric(d["avail_qty"], errors="coerce")).clip(lower=0)
    tot = {"all": np.zeros(len(grid)), "Unplanned outage": np.zeros(len(grid)), "Planned maintenance": np.zeros(len(grid))}
    for unit, g in d.groupby("production_resource_id" if "production_resource_id" in d else "mrid"):
        u = {k: np.zeros(len(grid)) for k in tot}
        for s, e, un, bt in zip(g["start"], g["end"], g["un"], g["businesstype"]):
            i, j = grid.searchsorted(s.tz_convert("UTC")), grid.searchsorted(e.tz_convert("UTC"))
            if j > i and np.isfinite(un):
                u["all"][i:j] = np.maximum(u["all"][i:j], un)
                if bt in u: u[bt][i:j] = np.maximum(u[bt][i:j], un)
        for k in tot: tot[k] += u[k]
    return {k: pd.Series(v / 1000, index=grid) for k, v in tot.items()}, len(d)
S = {}
def add(name, fn, keep):
    p = f"{SP}/entsoe/{fn}.parquet"
    if os.path.exists(p):
        s, n = series(p, keep); S[name] = s["all"]; S[name + ", unplanned"] = s["Unplanned outage"]; print(f"{name}: {n} records, mean {s['all'].mean():.1f} GW out, range {s['all'].min():.1f}-{s['all'].max():.1f}; unplanned mean {s['Unplanned outage'].mean():.1f}")
nohy = lambda d: ~d["plant_type"].str.contains("Hydro", na=False)
add("FR nuclear", "bt_unavail_gen_FR", lambda d: d["plant_type"] == "Nuclear")
add("FR all but hydro", "bt_unavail_gen_FR", nohy)
add("FR production units", "bt_unavail_prod_FR", nohy)
add("DE generation units", "unavail_gen_DE_LU", nohy)
add("DE production units", "bt_unavail_prod_DE_LU", nohy)
add("DE gas+coal+lignite units", "unavail_gen_DE_LU", lambda d: d["plant_type"].str.contains("Fossil", na=False))
z = np.load(f"{SP}/out/live_ho.npz"); a = z["actual"].astype(float); f = z["vint"].astype(float)
t = (z["cutoffs"].astype("datetime64[ns]")[:, None] + (np.arange(96) * np.timedelta64(15, "m"))[None, :]).ravel()
t = pd.DatetimeIndex(t).tz_localize("UTC")
err = pd.Series((a - f)[:, :96].ravel(), index=t); price = pd.Series(a[:, :96].ravel(), index=t)
print(f"\nerrors: {len(err)} quarter hours of the first forecast day, live setup (actual minus forecast, ct/kWh)")
hi = price >= price.quantile(0.8); big = err > 5; dip = err < -5
dow_hour = err.index.dayofweek.to_numpy() * 24 + err.index.hour.to_numpy()
print(f"{'':<34}corr with error  same, top-20% price hours  same, daily means   GW out when under-forecast >5 ct / otherwise")
for name, s in S.items():
    x = s.reindex(err.index)
    dx, de = x.groupby(x.index.date).mean(), err.groupby(err.index.date).mean()
    print(f"{name:<34}{np.corrcoef(x, err)[0, 1]:+.2f}             {np.corrcoef(x[hi], err[hi])[0, 1]:+.2f}                     {np.corrcoef(dx, de)[0, 1]:+.2f}                {x[big].mean():.1f} / {x[~big].mean():.1f}")
# out of sample: does the best series explain any error? fit on the first half, test on the second
half = len(err) // 2
for name in S:
    x = S[name].reindex(err.index).to_numpy(); e = err.to_numpy()
    b = np.polyfit(x[:half], e[:half], 1); r = e[half:] - np.polyval(b, x[half:])
    print(f"  linear correction from {name}: MAE second half {np.abs(e[half:]).mean():.2f} -> {np.abs(r).mean():.2f}")
