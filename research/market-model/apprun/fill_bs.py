"""Append the coming days from Bright Sky to the app's weather stores, scaled to the stores' own history on the overlap."""
import sys, os, json, asyncio
sys.path.insert(0, "/home/user/EpexPredictor")
import numpy as np, pandas as pd
from predictor.model.priceregion import PriceRegionName
from predictor.model.weatherstore import WeatherStore
from predictor.model.residualload import GridWeatherStore
BS, DATA = sys.argv[1], sys.argv[2]
FIELD = {"wind": ("wind_speed", 1.0, "scale"), "temp": ("temperature", 1.0, "offset"), "irradiance": ("solar", 1000.0, "scale"), "radiation": ("solar", 1000.0, "scale"), "pressure": ("pressure_msl", 1.0, "offset"), "humidity": ("relative_humidity", 1.0, "offset")}
def point(la, lo):
    w = json.load(open(f"{BS}/p_{la}_{lo}.json"))["weather"]; df = pd.DataFrame(w); df.index = pd.to_datetime(df["timestamp"], utc=True); return df[~df.index.duplicated()].sort_index()
async def fill(store):
    await store.load(); d = store.data; last = d.index[-1]; step = d.index[-1] - d.index[-2]; new = {}; sc = {}; cors = []
    for i, (la, lo) in enumerate(zip(store.latitudes(), store.longitudes())):
        p = point(la, lo)
        for name in store.VARIABLES.values():
            f, mult, kind = FIELD[name]; s = pd.to_numeric(p[f], errors="coerce") * mult
            idx = pd.date_range(s.index[0], s.index[-1], freq=step); s = s.reindex(s.index.union(idx)).interpolate(method="time").reindex(idx)
            own = d[f"{name}_{i}"]; common = own.index.intersection(s.index); a, b = own[common].to_numpy(float), s[common].to_numpy(float); ok = np.isfinite(a) & np.isfinite(b)
            if kind == "scale": k = float(np.clip((a[ok] * b[ok]).sum() / max((b[ok] ** 2).sum(), 1e-9), 0.3, 4.0)); v = k * s
            else: k = float((a[ok] - b[ok]).mean()); v = s + k
            if name == "humidity": v = v.clip(0, 100)
            sc.setdefault(name, []).append(k); cors.append((name, float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan)); new[f"{name}_{i}"] = v[v.index > last]
    add = pd.DataFrame(new)
    for name in store.VARIABLES.values():                       # a location without this field at its station: the mean of the other locations
        cols = [c for c in add.columns if c.rsplit("_", 1)[0] == name]; empty = [c for c in cols if add[c].isna().all()]
        for c in empty: add[c] = add[[k for k in cols if k not in empty]].mean(axis=1); print("   ", store.region.bidding_zone_entsoe, c, "not at the station - mean of the other locations used", flush=True)
    add = add.dropna(); add.index.name = "time"; store._update_data(add); store.data.sort_index(inplace=True); await store.serialize()
    print(f"{store.STORAGE_PREFIX} {store.region.bidding_zone_entsoe}: was to {last}, added {len(add)} rows to {store.data.index[-1]} | overlap rows {int(ok.sum())} |", {n: f"{np.median(v):.2f} ({kind_})" for n, v in sc.items() for kind_ in [FIELD[n][2]]}, "| correlation with own history:", {n: round(float(np.nanmedian([c for m, c in cors if m == n])), 2) for n in sc}, flush=True)
async def main():
    for n in sys.argv[3].split(","): await fill(WeatherStore(PriceRegionName[n].to_region(), DATA))
asyncio.run(main())
