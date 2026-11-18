import sys, os, asyncio, time
sys.path.insert(0, "/home/user/EpexPredictor")
import numpy as np, pandas as pd
from datetime import timedelta
from predictor.api import priceapi as api
from predictor.model.timestralblend import SLOT
from predictor.model.timestralload import TimestralLoad
from predictor.model.priceregion import PriceRegionName
SP = os.environ["SP"]; N = int(os.environ.get("N", 40)); H = TimestralLoad.HORIZON
async def main():
    await api.base_models.ensure_loaded(); base = api.base_models.get_base(PriceRegionName.DE); rl = base.residual_load.with_shared_stores(); gen = rl.generation
    truth = gen.data.copy(); truth_rl = truth.tso_load - truth.tso_wind - truth.tso_solar; tl = TimestralLoad(api.timestral)
    cuts = [pd.Timestamp("2026-11-06T22:45Z") - timedelta(days=k) for k in range(N)][::-1]; out = {"cut": [], "trees": [], "ts": [], "true": []}; t0 = time.time()
    for c in cuts:
        gen.horizon_cutoff = c
        await rl.train(c - timedelta(days=api.TRAINING_DAYS), c)
        idx = pd.date_range(c + SLOT, periods=H, freq="15min")
        trees = (await rl.get_data(c + SLOT, c + H * SLOT))["residual_load"].reindex(idx)
        fig = await tl.residual_load(rl); ts = fig.reindex(idx) if fig is not None else pd.Series(np.nan, index=idx)
        tr = truth_rl.reindex(idx)
        out["cut"].append(str(c)); out["trees"].append(trees.to_numpy()); out["ts"].append(ts.to_numpy()); out["true"].append(tr.to_numpy())
        print(c.date(), f"{time.time() - t0:.0f}s", "MAE day1 trees", round(float(np.nanmean(np.abs(trees - tr)[:96])), 2), "timestral", round(float(np.nanmean(np.abs(ts - tr)[:96])), 2), "| missing trees", int(trees.isna().sum()), "ts", int(ts.isna().sum()), "true", int(tr.isna().sum()), flush=True)
        np.savez(f"{SP}/out/load_bt.npz", cut=np.array(out["cut"]), trees=np.array(out["trees"]), ts=np.array(out["ts"]), true=np.array(out["true"]))
    print("finished", flush=True)
asyncio.run(main())
