"""End-to-end run of the app's model classes on live data sources: stage 1 for eight zones, stage 2 for Germany."""
import asyncio, logging, sys, time
from datetime import datetime, timedelta, timezone
sys.path.insert(0, "/home/user/EpexPredictor")
import aiohttp
_Orig = aiohttp.ClientSession
class _ProxySession(_Orig):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True); super().__init__(*a, **k)
aiohttp.ClientSession = _ProxySession
import pandas as pd
from predictor.model.europeweather import EuropeWeather
from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s"); logging.getLogger("predictor.model.weatherstore").setLevel(logging.WARNING)
D = sys.argv[1]; ZONES = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]
async def main():
    t0 = time.time(); base = {z: await PricePredictor(PriceRegionName(z).to_region(), D).load_from_persistence() for z in ZONES}
    eu = EuropeWeather([p.weatherstore for p in base.values()])
    for p in base.values(): p.europe_weather = eu
    now = datetime(2026, 11, 8, tzinfo=timezone.utc); start, end = datetime(2025, 10, 5, tzinfo=timezone.utc), now + timedelta(days=2)
    for p in base.values(): p.set_horizon_cutoff(now)
    for z in ZONES:
        for attempt in range(8):
            try: await base[z].train(start, end); break
            except Exception as e: print(z, "retry", attempt, repr(e)[:150], flush=True); await asyncio.sleep(60)
        print(z, "stage 1 trained", round(time.time() - t0), "s", "columns", len(base[z].traindata.columns), "rows", len(base[z].traindata), flush=True)
    de = PricePredictor(PriceRegionName.DE.to_region(), D, cross_predictors=list(base.values())); de.use_datastores_from(base["DE"])
    t1 = time.time(); await de.train(start, end); print("stage 2 trained in", round(time.time() - t1), "s; residual_load in training data:", "residual_load" in de.traindata.columns, flush=True)
    rl = base["DE"].residual_load; print("grid weather", rl.gridweather.data.shape, rl.gridweather.data.index.min(), rl.gridweather.data.index.max()); print("TSO figures", rl.generation.data.shape, rl.generation.data.index.min(), rl.generation.data.index.max())
    pred = await de.predict(now - timedelta(days=2), end, fill_known=False)
    for p in base.values(): p.set_horizon_cutoff(None)
    known = await de.pricestore.get_data(now - timedelta(days=2), end); tso = await rl.generation.get_data(now - timedelta(days=2), end); tso = (tso["tso_load"] - tso["tso_wind"] - tso["tso_solar"]).rename("tso_residual_load")
    for p in base.values(): p.set_horizon_cutoff(now)
    r = await de.residual_load.get_data(now - timedelta(days=2), end)
    out = pd.concat([pred.rename(columns={"price": "forecast"}), known.rename(columns={"price": "actual"}), r, tso], axis=1); out.to_parquet(f"{D}/e2e_result.parquet")
    print(out.resample("6h").mean().round(2).to_string()); f = out.loc[now + timedelta(minutes=15):]; print("MAE price", (f["forecast"] - f["actual"]).abs().mean().round(2), " MAE residual load vs TSO", (f["residual_load"] - f["tso_residual_load"]).abs().mean().round(2))
    print("forecast until", pred["price"].dropna().index.max(), " last known price", known.index.max(), " total", round(time.time() - t0), "s")
asyncio.run(main())
