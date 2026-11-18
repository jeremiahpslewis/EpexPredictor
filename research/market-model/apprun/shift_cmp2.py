import sys, os, asyncio
sys.path.insert(0, "/home/user/EpexPredictor")
import numpy as np, pandas as pd
from datetime import timedelta
from predictor.api import priceapi as api
from predictor.model import marketmodel as mm
from predictor.model.marketroute import MarketRoute
from predictor.model.timestralblend import SLOT
from predictor.model.priceregion import PriceRegionName
SP = os.environ["SP"]
async def main():
    await api.base_models.ensure_loaded(); base = api.base_models.get_base(PriceRegionName.DE)
    last = pd.Timestamp("2026-11-07T22:45:00Z"); base.pricestore.get_last_known = lambda: last
    await base.residual_load.train(last - timedelta(days=120), last)
    df = await base.prepare_dataframe(last - 8192 * SLOT, last); app = pd.Series(MarketRoute.supply_shift(df), index=df.index)
    z = np.load(f"{SP}/out/shift15_lite.npz"); idx = pd.date_range("2024-03-01T00:00Z", periods=len(z["p"]), freq="15min")
    ru = pd.Series(z["rl"] - z["other"] - z["Sp"] - z["nbterm"], index=idx).reindex(df.index)
    H = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); rg = H.gas.reindex(df.index, method="ffill")
    d = (df.gasprice - rg); print("gas at quarter hours, app - research EUR/MWh: mean abs", round(float(d.abs().mean()), 3), "| share equal", round(float((d.abs() < 1e-6).mean()), 3), "| by UTC hour, share unequal:", (d.abs() > 1e-6).groupby(df.index.hour).mean().round(2).to_dict())
    alt = pd.Series(mm.supply_shift(df.residual_load.to_numpy() - 8.0, df.price.to_numpy() * 10, rg.to_numpy(), 75.0), index=df.index); alt = app
    m = ru.notna() & alt.notna(); print("app code with research gas alignment - research shift GW: mean abs", round(float((alt - ru)[m].abs().mean()), 4), "max", round(float((alt - ru)[m].abs().max()), 4))
    print("app shift - research GW by UTC hour (mean abs):", (app - ru)[m].abs().groupby(df.index[m.to_numpy()].hour).mean().round(2).to_dict())
    g = await base.gasstore.get_data(last - timedelta(days=4), last); print(g.tail(5).to_string()); print(H.gas["2026-11-05":"2026-11-07"].drop_duplicates().to_string())
asyncio.run(main())
