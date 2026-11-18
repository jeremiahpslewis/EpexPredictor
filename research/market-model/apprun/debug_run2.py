import sys, os, asyncio, aiohttp
_Orig = aiohttp.ClientSession
class _ProxySession(_Orig):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True); super().__init__(*a, **k)
aiohttp.ClientSession = _ProxySession
sys.path.insert(0, "/home/user/EpexPredictor")
import numpy as np, pandas as pd
from predictor.api import priceapi as api
from predictor.model.timestralblend import TimestralBlend, SLOT
from predictor.model.marketroute import MarketRoute
from predictor.model.forecastlog import ForecastLog
from predictor.model.priceregion import PriceRegionName
async def main():
    await api.base_models.ensure_loaded(); base = api.base_models.get_base(PriceRegionName.DE); real_last = base.pricestore.get_last_known(); last = pd.Timestamp("2026-11-07T22:45:00Z"); base.pricestore.get_last_known = lambda: last; first = last + SLOT
    from datetime import timedelta
    await base.residual_load.train(last - timedelta(days=120), last)
    base.residual_load.generation.horizon_cutoff = last
    import time as _t; t0 = _t.time(); plain = await base.prepare_dataframe(first, last + 192 * SLOT); mixed = await api.timestral.inputs(base, first, last + 192 * SLOT); truth = base.residual_load.generation.data; tr = (truth.tso_load - truth.tso_wind - truth.tso_solar).reindex(plain.index)
    print('load forecast configured:', api.timestral.load is not None, '| seconds for the load run', round(_t.time() - t0, 1), '| residual load, first day after the last published figure, MAE GW: trees', round(float((plain.residual_load - tr).abs().iloc[:96].mean()), 2), 'handed over', round(float((mixed.residual_load - tr).abs().iloc[:96].mean()), 2), '| second day equal:', bool(np.allclose(plain.residual_load.iloc[97:189], mixed.residual_load.iloc[97:189])), '| first slot trees', round(float(plain.residual_load.iloc[0]), 2), 'handed over', round(float(mixed.residual_load.iloc[0]), 2), 'published', round(float(tr.iloc[0]), 2))
    cmp_ = await api.timestral.load.comparison(base.residual_load); ll = ForecastLog(os.environ['EPEXPREDICTOR_DATADIR'], 'DE_LU_smoketest', 'loadforecastlog', ('trees', 'timestral')); print('load log written:', ll.append(cmp_[0], cmp_[1]), '| rows', len(ForecastLog.read(ll.path)), '| columns', list(ForecastLog.read(ll.path).columns))
    df = await base.prepare_dataframe(first - 8192 * SLOT, last + 192 * SLOT); hist = df.loc[:last]; fut = df.loc[first:]
    print("pretend last known price:", last, "| history rows", len(hist), "| missing in history: price", round(float(hist.price.isna().mean()), 3), "residual load", round(float(hist.residual_load.isna().mean()), 3) if "residual_load" in df else "NO COLUMN", "gas price", round(float(hist.gasprice.isna().mean()), 3))
    if "residual_load" in df: m = hist.residual_load.isna(); print("   residual load gaps: first", m.idxmax() if m.any() else None, "last", m[::-1].idxmax() if m.any() else None, "| longest run of quarter hours", int(m.astype(int).groupby((~m).cumsum()).sum().max()), "| future known", int((~fut.residual_load.isna()).sum()), "of", len(fut))
    shift = MarketRoute.supply_shift(hist); print("   supply shift GW: mean", round(float(np.nanmean(shift)), 2), "sd", round(float(np.nanstd(shift)), 2), "| missing", round(float(np.isnan(shift).mean()), 3))
    ts, mr = api.timestral, api.market_route; print("forecasters configured:", ts is not None, mr is not None)
    f1 = await ts.forecast(base); f2 = await mr.forecast(base)
    for nm, f in (("timestral", f1), ("market", f2)): print(f"   {nm}: " + ("None" if f is None else f"{len(f)} quarter hours from {f.index[0]}, mean {f.mean():.2f} ct/kWh, min {f.min():.2f}, max {f.max():.2f}, finite {bool(np.isfinite(f).all())}"))
    pred = pd.DataFrame({"price": 12.0}, index=df.index); trees = pred["price"].copy(); ok = await ts.blend(base, [pred], {"market": mr}); print("blend returned", ok, "| forecasters mixed:", sorted(ts.last_forecasts), "| first slot: trees 12.00 ->", round(float(pred.loc[first, "price"]), 3), "| expected third each:", round(float((12.0 + ts.last_forecasts["timestral"][first] + ts.last_forecasts["market"][first]) / 3), 3) if len(ts.last_forecasts) == 2 else "n/a")
    fl = ForecastLog(os.environ["EPEXPREDICTOR_DATADIR"], "DE_LU_smoketest"); print("forecast log written:", fl.append(last, {"trees": trees, **ts.last_forecasts, "price": pred["price"]}), "| rows read back:", len(ForecastLog.read(fl.path)), "| columns:", list(ForecastLog.read(fl.path).columns))
asyncio.run(main())
import resource; print("peak memory MB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
