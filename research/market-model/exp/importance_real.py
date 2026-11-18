"""Feature usage of the real (modified) stage-2 DE model at one cutoff."""
import asyncio
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.pricepredictor import PricePredictor  # noqa: E402
from predictor.model.priceregion import PriceRegionName  # noqa: E402

DATA_DIR = sys.argv[1]
END = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
ZONES = [PriceRegionName(r) for r in "DE,AT,BE,FR,NL,SE4,DK1,DK2".split(",")]


async def main():
    base = {n: await PricePredictor(n.to_region(), DATA_DIR).load_from_persistence() for n in ZONES}
    start = END - timedelta(days=540)
    for p in base.values():
        p.pricestore.horizon_cutoff = END
        p.gasstore.horizon_cutoff = END
    await asyncio.gather(*[p.train(start, END - timedelta(minutes=15)) for p in base.values()])
    de = PricePredictor(PriceRegionName.DE.to_region(), DATA_DIR, cross_predictors=list(base.values()))
    de.use_datastores_from(base[PriceRegionName.DE])
    await de.train(start, END - timedelta(minutes=15))
    for label, booster in [("price model", de.predictor), ("gas-relative model", de.gas_predictor)]:
        imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        imp = (imp / imp.sum()).sort_values(ascending=False)
        grp = imp.groupby(lambda c: "Europe-wide weather" if c.startswith("eu_") else "cross-region price forecasts" if c.startswith("cross_price_") else "own inputs").sum()
        print(f"{label}: " + ", ".join(f"{k} {100*v:.1f}%" for k, v in grp.sort_values(ascending=False).items()))
        print("   top inputs: " + ", ".join(f"{k} {100*v:.1f}%" for k, v in imp.head(8).items()))
        print("   Europe-wide: " + ", ".join(f"{k} {100*v:.2f}% (rank {list(imp.index).index(k) + 1} of {len(imp)})" for k in imp.index if k.startswith("eu_")))


asyncio.run(main())
