"""Reference run: the repo's own performance_testing.py loop (same calls, same order) on a configurable
window/region set, additionally dumping the raw predictions so the fast harness can be checked against it."""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, "/home/user/EpexPredictor/predictor")

import model.pricepredictor as pred  # noqa: E402
from model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR, OUT = sys.argv[1], sys.argv[2]
START = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
END = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc)
CROSS_REGIONS = [PriceRegionName(r) for r in sys.argv[5].split(",")]
EVAL_REGIONS = [PriceRegionName(r) for r in sys.argv[6].split(",")]
TWO_STAGE = sys.argv[7] == "2"
LEARN_DAYS = int(sys.argv[8]) if len(sys.argv) > 8 else 180

logging.basicConfig(format="%(message)s", level=logging.WARNING)


async def main():
    stage1 = {}
    for name in CROSS_REGIONS:
        stage1[name] = await pred.PricePredictor(name.to_region(), DATA_DIR).load_from_persistence()
    from model.europeweather import EuropeWeather
    europe_weather = EuropeWeather([p.weatherstore for p in stage1.values()])
    for p in stage1.values():
        p.europe_weather = europe_weather
    stage2 = {}
    if TWO_STAGE:
        for name in EVAL_REGIONS:
            p = pred.PricePredictor(name.to_region(), DATA_DIR, cross_predictors=[stage1[n] for n in CROSS_REGIONS])
            p.use_datastores_from(stage1[name])
            stage2[name] = p

    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START
    out = {f"s1_{n.value}": [] for n in EVAL_REGIONS} | {f"s2_{n.value}": [] for n in EVAL_REGIONS} | {
        f"actual_{n.value}": [] for n in EVAL_REGIONS}
    cutoffs = []
    while learn_end < END - timedelta(days=3):
        d0, d3 = learn_end, learn_end + timedelta(days=3)
        for name in CROSS_REGIONS:
            stage1[name].pricestore.horizon_cutoff = learn_end
            stage1[name].gasstore.horizon_cutoff = learn_end
        await asyncio.gather(*[stage1[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in CROSS_REGIONS])
        if TWO_STAGE:
            await asyncio.gather(*[stage2[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in EVAL_REGIONS])
        for name in EVAL_REGIONS:
            out[f"s1_{name.value}"].append((await stage1[name].predict(d0, d3, False))["price"].to_numpy())
            if TWO_STAGE:
                out[f"s2_{name.value}"].append((await stage2[name].predict(d0, d3, False))["price"].to_numpy())
        for name in EVAL_REGIONS:
            stage1[name].pricestore.horizon_cutoff = None
            stage1[name].gasstore.horizon_cutoff = None
            out[f"actual_{name.value}"].append((await stage1[name].pricestore.get_data(d0, d3))["price"].to_numpy())
            stage1[name].pricestore.horizon_cutoff = learn_end
            stage1[name].gasstore.horizon_cutoff = learn_end
        if len(cutoffs) == 0:
            td = stage1[EVAL_REGIONS[0]].traindata
            print("stage-1 train rows:", len(td), "cols:", len(td.columns) - 1, td.index.min(), td.index.max())
            if TWO_STAGE:
                td = stage2[EVAL_REGIONS[0]].traindata
                print("stage-2 train rows:", len(td), "cols:", len(td.columns) - 1, list(td.columns)[-10:])
        cutoffs.append(np.datetime64(learn_end.replace(tzinfo=None)))
        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        print(".", end="", flush=True)
    print()
    np.savez_compressed(OUT, cutoffs=np.array(cutoffs), **{k: np.stack(v) for k, v in out.items() if len(v)})


asyncio.run(main())
