import asyncio, logging, sys, time
from datetime import datetime, timedelta, timezone
sys.path.insert(0, "/home/user/EpexPredictor")
import pandas as pd
from predictor.model.europeweather import EuropeWeather
from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName
from predictor.model.timestralblend import TimestralBlend
logging.basicConfig(level=logging.WARNING)
D = sys.argv[1]
async def main():
    base = {z: await PricePredictor(PriceRegionName(z).to_region(), D).load_from_persistence() for z in ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]}
    eu = EuropeWeather([p.weatherstore for p in base.values()])
    for p in base.values(): p.europe_weather = eu
    now = datetime(2026, 11, 8, tzinfo=timezone.utc); de = base["DE"]; de.set_horizon_cutoff(now)
    blend = TimestralBlend(sys.argv[2]); t0 = time.time(); fc = await blend.forecast(de); print("Timestral forecast:", None if fc is None else (len(fc), fc.index[0], fc.index[-1]), "in", round(time.time() - t0), "s")
    r = pd.read_parquet(f"{D}/e2e_result.parquet"); trees = r[["forecast"]].rename(columns={"forecast": "price"}).copy(); both = trees.copy()
    print("blended:", await blend.blend(de, [both]))
    out = pd.concat([trees["price"].rename("trees"), fc.rename("timestral"), both["price"].rename("blend"), r["actual"]], axis=1).loc[fc.index[0]:fc.index[-1]]
    print(out.resample("6h").mean().round(2).to_string()); print("MAE", {c: round(float((out[c] - out["actual"]).abs().mean()), 2) for c in ("trees", "timestral", "blend")})
asyncio.run(main())
