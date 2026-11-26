"""Agent P: market route with Timestral re-run on a substituted supply-shift history (C's per-day input frames).

Reproduces MarketRoute.forecast from C's dump of the Timestral-mixed DE input frame (SP/C/tsdump/<tag>/<D>.parquet):
history = rows <= cutoff (price known), demand = residual_load - 8, shift = marketmodel.supply_shift with gas lagged
192 qh, Timestral(shift, demand covariate, weekday), cleared on D. Variants:
  repro    : as the app (checked against the stored faithful market route)
  gas1     : gas lagged 96 qh (gas day D-1 on D, known at the gate) in the shift history and the clearing
  consist  : the RL of past days in the history replaced by the forecast-time RL stored for those days (rl_mixed of
             SP/faithful/out_strict / out_fresh_strict, i.e. what the route itself read at each earlier gate); the
             published RL stays where no stored day exists. Shift history and covariate both use it.
Usage: market_ts.py VARIANT:DAY[,DAY..] [...]   (writes SP/P/market_ts/<variant>/<D>.npz)
"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model import marketmodel as mm  # noqa: E402

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
SLOT = pd.Timedelta(minutes=15)
CARBON, OTHER, EUR = 75.0, 8.0, 10.0
_fc = None


def forecaster():
    global _fc
    if _fc is None:
        import torch
        torch.set_num_threads(1)
        os.environ["HF_HUB_OFFLINE"] = "1"
        from predictor.model.timestralblend import TESTED_REVISIONS
        from timestral import TimestralForecaster
        name = "mistralai/Timestral-123M"   # as deployed, at the app's pinned revision
        _fc = TimestralForecaster.from_pretrained(name, revision=TESTED_REVISIONS[name])
    return _fc


def frame(day):
    for tag in ("ctx8k", "fresh8k"):
        fn = f"{SP}/C/tsdump/{tag}/{day}.parquet"
        if os.path.exists(fn):
            return pd.read_parquet(fn)
    raise FileNotFoundError(day)


def stored_rl():
    """forecast-time RL (Timestral-mixed) the market route read for each stored day, on that day's slots"""
    parts = []
    for o in ("out_strict", "out_fresh_strict"):
        for fn in sorted(glob.glob(f"{SP}/faithful/{o}/*.npz")):
            z = np.load(fn); iv = np.asarray(z["idx"])
            idx = pd.to_datetime(iv, unit="us" if np.abs(iv).max() < 1e17 else "ns", utc=True)
            parts.append(pd.Series(np.asarray(z["rl_mixed"], float), index=idx))
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index()


def lag(series, n):
    out = series.copy(); out.index = out.index + n * SLOT
    return out.reindex(series.index).fillna(series.dropna().iloc[0]).to_numpy(float)


def forecast(day, variant, rl_hist=None):
    import torch
    df = frame(day)
    D = pd.date_range(pd.Timestamp(day, tz="Europe/Berlin"), periods=96, freq="15min").tz_convert("UTC")
    cutoff = D[0] - SLOT
    hist = df.loc[:cutoff]
    n = len(hist)
    assert hist.index[-1] == cutoff and df.loc[D[0]:, "price"].isna().all(), "price known after the cutoff"
    rl = df["residual_load"].astype("float64").copy()
    if variant == "consist":
        past = rl_hist.reindex(rl.index[:n])
        past = past[past.index < D[0]]                     # only earlier days (stored at their own gates)
        rl.loc[past.dropna().index] = past.dropna()
    gas = lag(df["gasprice"], 96 if variant == "gas1" else 192)
    demand = rl.to_numpy(float) - OTHER
    shift = mm.supply_shift(demand[:n], df["price"].to_numpy(float)[:n] * EUR, gas[:n], CARBON)
    inputs = np.vstack([demand, gas])
    ok = np.isfinite(inputs).all(axis=0)
    horizon = int(np.argmin(ok[n:])) if not ok[n:].all() else len(df) - n
    rows = n + horizon
    with torch.no_grad():
        f = forecaster().forecast(shift[None, None, :], prediction_length=horizon,
                                  known_covariates=demand[None, None, :rows],
                                  categorical_known_covariates=df["weekday"].to_numpy(float)[None, None, :rows])
    s_fc = np.asarray(f.median)[0, 0]
    price = mm.clearing_price(demand[n:rows], gas[n:rows], CARBON, shift=s_fc) / EUR
    return pd.Series(price, index=df.index[n:rows]).reindex(D).to_numpy()


def main():
    rl_hist = stored_rl()
    for job in sys.argv[1:]:
        variant, days = job.split(":")
        os.makedirs(f"{SP}/P/market_ts/{variant}", exist_ok=True)
        for day in days.split(","):
            fn = f"{SP}/P/market_ts/{variant}/{day}.npz"
            if os.path.exists(fn):
                continue
            t = time.time()
            p = forecast(day, variant, rl_hist)
            np.savez(fn, market=p)
            print(f"{variant} {day} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
