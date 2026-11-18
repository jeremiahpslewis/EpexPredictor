"""Peak memory of a full training cycle with the real code (stage 1 + stage 2 for all available zones)."""
import asyncio
import resource
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.europeweather import EuropeWeather  # noqa: E402
from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR = sys.argv[1]
END = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
ZONES = [PriceRegionName(r) for r in "DE,AT,BE,FR,NL,SE4,DK1,DK2".split(",")]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


async def main():
    t0 = time.time()
    base = {n: await PricePredictor(n.to_region(), DATA_DIR).load_from_persistence() for n in ZONES}
    europe = EuropeWeather([p.weatherstore for p in base.values()])
    for p in base.values():
        p.europe_weather = europe
        p.pricestore.horizon_cutoff = END
        p.gasstore.horizon_cutoff = END
    print(f"stores loaded: peak RSS {rss_mb():.0f} MB")
    start = END - timedelta(days=540)
    pred_end = END + timedelta(days=3)
    await asyncio.gather(*[p.train(start, pred_end) for p in base.values()])
    print(f"stage 1 trained ({len(base)} zones, in parallel): peak RSS {rss_mb():.0f} MB, {time.time() - t0:.0f}s")
    stage2 = {}
    for n in ZONES:
        p = PricePredictor(n.to_region(), DATA_DIR, cross_predictors=list(base.values()))
        p.use_datastores_from(base[n])
        await p.train(start, pred_end)
        await p.predict(start, pred_end)
        await p.predict(start, pred_end, fill_known=False)
        stage2[n] = p
    print(f"stage 2 trained and predicted ({len(stage2)} zones): peak RSS {rss_mb():.0f} MB, {time.time() - t0:.0f}s total")


asyncio.run(main())
