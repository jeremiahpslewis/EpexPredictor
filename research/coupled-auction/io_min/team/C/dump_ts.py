"""Agent C: dump Timestral's exact inputs per target day from the faithful harness (faithful_bt imported unchanged),
so Timestral variants (context, covariates, model size, extra series) can be screened offline.

Per day D: stage_day(D) (as-of-gate stores, strict vintage, base RL), stage-1 training of the base models (DE's residual
load included), then api.timestral.forecast(DE base) exactly as TimestralBlend.blend calls it, with _forecast wrapped to
capture (prices, covariates, weekday) and its output. Also saved: neighbours' price histories up to the cutoff on the
same index, the full DE input frame over the Timestral range, the actual on D. Output SP/C/tsdump/<D>.npz (+ .parquet).
Usage: dump_ts.py --days 2026-09-10,... [--de-only]
"""
import argparse, asyncio, os, sys, time, logging
from datetime import date, timedelta
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "faithful"))
import faithful_bt as fb  # noqa: E402
log = logging.getLogger("C.dump_ts")

def cov_names(pred, df, nrows):
    """row names of TimestralBlend.covariates, same order"""
    wind = df.filter(regex=r"^wind_\d+$"); onshore = len(pred.region.latitudes)
    n = ["wind_on", "wind_pow", "irr", "temp", "holiday"] + (["wind_off"] if wind.shape[1] > onshore else [])
    n += [c for c in ("eu_wind", "eu_wind_northsea", "eu_irradiance", "gasprice") if c in df.columns]
    if nrows == len(n) + 1: n.append("residual_load")
    assert len(n) == nrows, (n, nrows)
    return n


async def run(days, out, de_only, context, also=(), tag=""):
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(vintage="strict", latency=6.0, rl_variant="base", out=out)
    fb.RLV.variant = "base"
    h = fb.Harness(args); await h.setup()
    tb = fb.api.timestral; cap = {}
    tb.CONTEXT = context   # instance attribute only: longer history captured, cut back offline (8192 = deployed)
    orig = tb._forecast
    def spy(prices, covariates, weekday):
        v = orig(prices, covariates, weekday); cap.update(prices=prices.copy(), cov=covariates.copy(), wd=weekday.copy(), out=np.asarray(v).copy()); return v
    tb._forecast = spy
    orig_inputs = type(tb).inputs
    async def spy_inputs(predictor, start, end):
        df = await orig_inputs(tb, predictor, start, end); cap["df"] = df.copy(); return df
    tb.inputs = spy_inputs
    for d in days:
        fn = f"{out}/{d}.npz"
        if os.path.exists(fn): continue
        t0 = time.time(); cap.clear()
        gate, cutoff, info = h.stage_day(d)
        start, end = cutoff - pd.Timedelta(days=fb.api.TRAINING_DAYS), cutoff
        ps = [h.de] if de_only else list(h.bases.values())
        await asyncio.gather(*(p.train(start.to_pydatetime(), end.to_pydatetime()) for p in ps))
        t1 = time.time()
        fc = await tb.forecast(h.de)
        D = fb.day_index(d)
        df = cap["df"]; n = len(cap["prices"]); hidx = df.index[:n]
        names = cov_names(h.de, df, cap["cov"].shape[0])
        nb = {}
        # leak assertions: the DE context ends exactly at the cutoff (D-1 23:45 local); every neighbour store is cut at
        # the cutoff (no slot of D, which comes from the same coupled auction); no forward fill (plain reindex, NaN kept)
        assert hidx[-1] == cutoff and len(hidx) == n, (hidx[-1], cutoff)
        for name, p in h.bases.items():
            ps = p.pricestore.data["price"]
            assert ps.index.max() <= cutoff, f"{name}: price store reaches {ps.index.max()} > cutoff {cutoff}"
            nb[str(getattr(name, "value", name))] = ps.reindex(hidx).to_numpy(float)
        np.savez_compressed(fn + ".tmp.npz", prices=cap["prices"], cov=cap["cov"], wd=cap["wd"], out=cap["out"],
                            idx=df.index.asi8, names=np.array(names), n_hist=n, fc_D=fc.reindex(D).to_numpy(float), D=D.asi8,
                            actual=h.actual.reindex(D).to_numpy(float), cutoff=np.int64(cutoff.value), nb_names=np.array(list(nb)), nb=np.stack(list(nb.values())))
        os.replace(fn + ".tmp.npz", fn)
        df.astype("float32").to_parquet(f"{out}/{d}.parquet")
        if also:   # replay variants right away on the loaded model (confirmation waves: no second pass)
            import ts_variants as V
            V._M[tb.model_name] = tb.forecaster
            z = V.load(fn); z["_path"] = fn; vs = V.variants(z)
            for v in also:
                fo = f"{fb.SP}/C/tsvar_{tag}/{v}_{d}.npy"; os.makedirs(os.path.dirname(fo), exist_ok=True); np.save(fo, vs[v]())
        log.warning(f"{d}: stage1 {t1 - t0:.0f}s total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--days", required=True); ap.add_argument("--de-only", action="store_true")
    ap.add_argument("--tag", default="base"); ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--also", default=""); a = ap.parse_args()
    asyncio.run(run([date.fromisoformat(x) for x in a.days.split(",")], f"{fb.SP}/C/tsdump/{a.tag}", a.de_only, a.context, tuple(x for x in a.also.split(",") if x), a.tag))
