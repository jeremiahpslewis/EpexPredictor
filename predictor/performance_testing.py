#!/usr/bin/python3

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import os

import model.pricepredictor as pred
from backtest_metrics import Metrics, last_week, table
from model.europeweather import EuropeWeather
from model.priceregion import PriceRegion, PriceRegionName


START: datetime = datetime.fromisoformat("2025-09-01T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-09-01T00:00:00Z")
CROSS_REGIONS = [
    PriceRegionName.DE,
    PriceRegionName.AT,
    PriceRegionName.BE,
    PriceRegionName.NL,
    PriceRegionName.SE1,
    PriceRegionName.SE2,
    PriceRegionName.SE3,
    PriceRegionName.SE4,
    PriceRegionName.DK1,
    PriceRegionName.DK2,
    PriceRegionName.ES,
    PriceRegionName.PT,
    PriceRegionName.FR
]

# Regions to evaluate (get metrics for). Usually a subset of CROSS_REGIONS for debugging.
EVAL_REGIONS = CROSS_REGIONS

LEARN_DAYS : int = 540

PARALLELIZE = True

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def load_data(p : pred.PricePredictor):
    """
    preload data for whole time range to reduce individual http requests
    """
    learn_start = START - timedelta(days=LEARN_DAYS)
    await asyncio.gather(
        p.weatherstore.get_data(learn_start, END),
        p.pricestore.get_data(learn_start, END),
        p.entsoestore.get_data(learn_start, END),
        p.auxstore.get_data(learn_start, END),
        p.gasstore.get_data(learn_start, END)
    )

# Set to False to fall back to the original single-stage (per-region) evaluation
TWO_STAGE = True


async def run_many(coros):
    if PARALLELIZE:
        await asyncio.gather(*coros)
    else:
        for c in coros:
            await c



async def main():
    data_dir = os.getenv("EPEXPREDICTOR_DATADIR", "./data")

    # Stage 1: baseline models for ALL cross regions (needed as features for stage-2)
    stage1 = {}
    for name in CROSS_REGIONS:
        region = name.to_region()
        stage1[name] = await pred.PricePredictor(region, data_dir).load_from_persistence()

    # Large-scale weather features, built from the weather of all regions (stage-2 inherits them with the stores)
    europe_weather = EuropeWeather([p.weatherstore for p in stage1.values()])
    for p in stage1.values():
        p.europe_weather = europe_weather

    # Stage 2: only for eval regions, with stage-1 forecasts of ALL cross regions as features.
    # Each stage-2 predictor shares its data stores with its stage-1 counterpart.
    stage2 = {}
    if TWO_STAGE:
        for name in EVAL_REGIONS:
            region = name.to_region()
            p = pred.PricePredictor(region, data_dir, cross_predictors=[stage1[n] for n in CROSS_REGIONS])
            p.use_datastores_from(stage1[name])
            stage2[name] = p

    # Preload data for all cross regions (stage-2 shares stores with stage-1)
    for name in CROSS_REGIONS:
        await load_data(stage1[name])

    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START

    models = ("stage1", "stage2", "last_week") if TWO_STAGE else ("stage1", "last_week")
    metrics = {name: Metrics(models) for name in EVAL_REGIONS}

    iterations = 0
    while learn_end < END - timedelta(days=3):

        # intervals to predict and check. Could be done nicer but w/e
        d0 = learn_end
        d1 = learn_end + timedelta(days=1)
        d2 = learn_end + timedelta(days=2)
        d3 = learn_end + timedelta(days=3)

        # Make sure training/prediction doesn't "cheat" with data that is known during
        # performance testing, but not for actual forecasts. Must cap ALL cross regions
        # since stage-2 queries their stores via cross-predictors.
        for name in CROSS_REGIONS:
            stage1[name].set_horizon_cutoff(learn_end)

        # Stage 1: train every cross region's baseline model for this window
        await run_many([stage1[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in CROSS_REGIONS])

        # Stage 2: train every region's model, using the (freshly trained) stage-1 forecasts
        # of all regions as extra features. Must run after all stage-1 models are trained.
        if TWO_STAGE:
            await run_many([stage2[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in EVAL_REGIONS])

        # Predict the upcoming days (cutoffs still at learn_end -> no cheating)
        preds1 = {}
        for name in EVAL_REGIONS:
            preds1[name] = await stage1[name].predict(d0, d3, False)
        preds2 = {}
        if TWO_STAGE:
            for name in EVAL_REGIONS:
                preds2[name] = await stage2[name].predict(d0, d3, False)

        # Fetch actuals (needs full data, so lift the cutoff)
        actuals = {}
        for name in EVAL_REGIONS:
            stage1[name].set_horizon_cutoff(None)
            # from a week before d0, for the same-slot-last-week baseline (those prices are known at d0)
            actuals[name] = (await stage1[name].pricestore.get_data(d0 - timedelta(days=7), d3))["price"]
            stage1[name].set_horizon_cutoff(learn_end)

        for name in EVAL_REGIONS:
            predictions = {"stage1": preds1[name]["price"], "last_week": last_week(actuals[name], d0, d3)}
            if TWO_STAGE:
                predictions["stage2"] = preds2[name]["price"]
            metrics[name].add(actuals[name], predictions, d0, d1, d2, d3)

        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        iterations += 1
        print('.', end='')

    print()

    print(f"Rolling backtest with historical weather: windows starting {START.date()} to {(learn_end - timedelta(days=1)).date()} "
          f"(UTC midnight), {iterations} windows, {LEARN_DAYS}-day training window, "
          f"{'two-stage' if TWO_STAGE else 'single-stage'}. Day k = the 24 hours [d0 + k - 1, d0 + k) UTC after the training end.")
    print("Prices and errors in ct/kWh. Errors on the quarter hours where every model and the actual price are there (Common).")
    print()
    print(table({name.value: metrics[name] for name in EVAL_REGIONS}))


if __name__ == "__main__":
    asyncio.run(main())
