"""Agent P: dump the DE stage-2 tree design matrices from the faithful harness, so tree variants can be screened offline.

Imports faithful_bt (unchanged) and drives its Harness: stage_day(D) stages the as-of-gate stores; the 8 stage-1 base
models (incl. DE's residual-load trees) are retrained on [cutoff - 540 d, cutoff] every --train-every days (1 = the
champion's daily cadence); the stage-2 frame is built exactly as PricePredictor.prepare_dataframe builds it for the
deployed stage-2 model (cross_price_* = stage-1 forecasts). No stage-2 booster is trained here.

Per target day D, SP/P/dump/<tag>/<D>.D.parquet: the stage-2 frame on D's slots (price NaN: hidden) as the app predicts
it (range cutoff-14d+1 .. end of D+1, sliced to D). On a retrain day also <D>.train.parquet: the training frame
[cutoff-540 d, cutoff] (rows with any NaN kept; the app drops them). float32.

Usage: dump.py --days 2026-09-08,... [--train-every 1] [--tag daily]
"""
import argparse, asyncio, os, sys, time, logging
from datetime import date, timedelta
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "faithful"))
import faithful_bt as fb  # noqa: E402  (sets up the environment and monkeypatches, as the harness does)

log = logging.getLogger("P.dump")


def f32(df):  # name kept; stores float64
    """float32 except the time index feature ts (float32 would round it to ~2 min)"""
    return df.astype("float64")   # float32 moved tree splits (up to 0.4 ct on a day); kept exact


async def run(days, train_every, out, threads, rl_variant="base"):
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(vintage="strict", latency=6.0, rl_variant=rl_variant, out=out)
    fb.RLV.variant = rl_variant
    if rl_variant in fb.MATCHED_VARIANTS:
        fb.RLV.d2 = fb.load_d2_history()
    h = fb.Harness(args)
    await h.setup()
    if threads:
        import lightgbm  # noqa: F401  (LightGBM reads OMP_NUM_THREADS; set it in the environment)
    last_train = None
    for i, d in enumerate(days):
        fnD = f"{out}/{d}.D.parquet"
        need_train = last_train is None or (d - last_train).days >= train_every
        if os.path.exists(fnD) and not need_train:
            continue
        t0 = time.time()
        gate, cutoff, info = h.stage_day(d)
        start, end = cutoff - pd.Timedelta(days=fb.api.TRAINING_DAYS), cutoff
        if need_train:
            await asyncio.gather(*(p.train(start.to_pydatetime(), end.to_pydatetime()) for p in h.bases.values()))
            last_train = d
        stage2 = fb.pp.PricePredictor(h.de.region, storage_dir=fb.api.EPEXPREDICTOR_DATADIR,
                                      cross_predictors=fb.api.base_models.get_cross_predictors())
        stage2.use_datastores_from(h.de)
        # stage2.train() would retrain its own residual-load boosters on the same window and data (the harness asserts
        # they equal DE's to 1e-6); share DE's instead of training them again
        stage2.residual_load.models = h.de.residual_load.models
        stage2.residual_load.extra = getattr(h.de.residual_load, "extra", None)
        t1 = time.time()
        if need_train:
            tr = await stage2.prepare_dataframe(start.to_pydatetime(), end.to_pydatetime())
            f32(tr).to_parquet(f"{out}/{d}.train.parquet")
        D = fb.day_index(d)
        await h.assert_no_leak(d, cutoff, stage2)
        pstart, pend = cutoff - pd.Timedelta(days=14) + fb.SLOT, fb.day_index(d + timedelta(days=1))[-1]
        # the variant's frame (<D>.D.parquet) and, for a non-base variant, the base RL frame too (<D>.Dbase.parquet):
        # training frames are the same for both (rows up to the cutoff carry the published RL)
        for var, fn in [(rl_variant, fnD)] + ([("base", f"{out}/{d}.Dbase.parquet")] if rl_variant != "base" else []):
            fb.RLV.variant = var
            dfD = await stage2.prepare_dataframe(pstart.to_pydatetime(), pend.to_pydatetime())
            dfD = dfD.reindex(D)
            assert dfD["price"].isna().all()
            assert "residual_load" in dfD.columns and dfD["residual_load"].notna().all(), "stage-2 frame without residual load"
            dfD["actual"] = h.actual.reindex(D).to_numpy()
            dfD["rl_true"] = h.rl_true.reindex(D).to_numpy()
            dfD.attrs = {}
            f32(dfD).to_parquet(fn + ".tmp")
            os.replace(fn + ".tmp", fn)
        fb.RLV.variant = rl_variant
        log.warning(f"{d}: train={need_train} stage1 {t1 - t0:.0f}s total {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--train-every", type=int, default=1)
    ap.add_argument("--tag", default="daily")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rl-variant", default="base", choices=("base", "matched"))
    a = ap.parse_args()
    days = [date.fromisoformat(x) for x in a.days.split(",")]
    out = f"{fb.SP}/P/dump/{a.tag}"
    asyncio.run(run(days, a.train_every, out, a.threads, a.rl_variant))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
