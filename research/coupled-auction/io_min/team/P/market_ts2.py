"""Agent P: market route with Timestral on a substituted shift history, frames rebuilt from the stores (no harness).

What the route reads, per target day D, rebuilt from the cached stores (SP/fresh_data/stores) and the stored faithful
day files:
  history (8192 qh ending at the cutoff D-1 23:45 local): known DE price; residual load = the published TSO figures
    (tso_load - tso_wind - tso_solar, as ResidualLoad.get_data serves the past); THE gas ffilled; local weekday.
  D (96 qh): residual load = the stored Timestral-mixed RL the route read at D's gate (rl_mixed of D's faithful file);
    gas ffilled from gas day D-1; price unknown.
The app's route also passes D+1 as horizon; here the horizon is D only (96 qh), for both variants, so the paired
comparison repro96 vs consist96 is like for like. repro96 vs the stored route measures the horizon effect.
Variants: repro96 (as the app, horizon 96) and consist96 (past days' RL in history and covariate = the stored
forecast-time rl_mixed of each earlier day d < D, i.e. what the route read at d's gate; published RL where no stored
day exists).
Usage: market_ts2.py check DAY,..   (compare the rebuilt history with C's dumped frame)
       market_ts2.py run OUTTAG DAY,..
"""
import os, sys, glob, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model import marketmodel as mm  # noqa: E402

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
ST = f"{SP}/fresh_data/stores"
SLOT = pd.Timedelta(minutes=15)
CONTEXT, OTHER, CARBON, EUR = 8192, 8.0, 75.0, 10.0
DAYFILES = ("out_strict", "out_fresh_strict")


def store(name, col=None):
    s = pd.read_json(f"{ST}/{name}.json.gz")
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    s = s.sort_index()
    return s[col] if col else s


def dayfile(d):
    for o in DAYFILES:
        fn = f"{SP}/faithful/{o}/{d}.npz"
        if os.path.exists(fn):
            return np.load(fn)
    return None


def stored_rl():
    """day -> Series of the rl_mixed the route read at that day's gate, on that day's slots"""
    out = {}
    for o in DAYFILES:
        for fn in sorted(glob.glob(f"{SP}/faithful/{o}/*.npz")):
            z = np.load(fn); iv = np.asarray(z["idx"])
            idx = pd.to_datetime(iv, unit="us" if np.abs(iv).max() < 1e17 else "ns", utc=True)
            out[os.path.basename(fn)[:10]] = pd.Series(np.asarray(z["rl_mixed"], float), index=idx)
    return out


class Data:
    def __init__(self):
        self.price = store("prices_v3_DE_LU", "price")
        self.gas = store("gasprices_the_v1_DE_LU", "gasprice")
        g = store("generation_v1_DE_LU")
        self.rl_pub = g.tso_load - g.tso_wind - g.tso_solar
        self.rl_log = stored_rl()


def build(day, data, variant):
    D = pd.date_range(pd.Timestamp(day, tz="Europe/Berlin"), periods=200, freq="15min").tz_convert("UTC")
    D = D[D.tz_convert("Europe/Berlin").normalize() == pd.Timestamp(day, tz="Europe/Berlin")]
    cutoff = D[0] - SLOT
    H = pd.date_range(cutoff - (CONTEXT - 1) * SLOT, cutoff, freq="15min")
    idx = H.append(D)
    z = dayfile(day)
    price = data.price.reindex(H).to_numpy(float)
    rl = pd.Series(np.r_[data.rl_pub.reindex(H).to_numpy(float), np.asarray(z["rl_mixed"], float)], index=idx)
    nlog = 0
    if variant == "consist96":
        used = []
        for d, s in data.rl_log.items():
            if d < day:
                s = s[s.index.isin(H)]
                rl.loc[s.index] = s.to_numpy()
                nlog += len(s) > 0; used.append(s.index.max() if len(s) else None)
        # leak assertion: every forecast-time value used belongs to a day before D (issued at that day's gate)
        assert all(u is None or u < D[0] for u in used), day
    gas = data.gas[:cutoff].reindex(idx, method="ffill")       # gas known at the cutoff, ffilled into D
    assert data.gas[:cutoff].index.max() <= cutoff
    wd = idx.tz_convert("Europe/Berlin").weekday.to_numpy(float)
    return dict(idx=idx, H=H, D=D, price=price, rl=rl.to_numpy(float), gas=gas, wd=wd, nlog=nlog)


_fc = None


def forecaster():
    global _fc
    if _fc is None:
        import torch
        torch.set_num_threads(1)
        os.environ["HF_HUB_OFFLINE"] = "1"
        from predictor.model.timestralblend import TESTED_REVISIONS
        from timestral import TimestralForecaster
        name = "mistralai/Timestral-123M"
        _fc = TimestralForecaster.from_pretrained(name, revision=TESTED_REVISIONS[name])
    return _fc


def lag2(gas):
    s = gas.copy(); s.index = s.index + 192 * SLOT
    return s.reindex(gas.index).fillna(gas.dropna().iloc[0]).to_numpy(float)


def forecast(f):
    import torch
    n = len(f["H"])
    demand = f["rl"] - OTHER
    g = lag2(f["gas"])
    shift = mm.supply_shift(demand[:n], f["price"] * EUR, g[:n], CARBON)
    with torch.no_grad():
        out = forecaster().forecast(shift[None, None, :], prediction_length=len(f["D"]),
                                    known_covariates=demand[None, None, :], categorical_known_covariates=f["wd"][None, None, :])
    s_fc = np.asarray(out.median)[0, 0]
    return mm.clearing_price(demand[n:], g[n:], CARBON, shift=s_fc) / EUR


def check(days):
    data = Data()
    for day in days:
        f = build(day, data, "repro96")
        for tag in ("ctx8k", "fresh8k"):
            fn = f"{SP}/C/tsdump/{tag}/{day}.parquet"
            if os.path.exists(fn):
                c = pd.read_parquet(fn).reindex(f["idx"])
        n = len(f["H"])
        print(day, "max |price| diff", np.nanmax(np.abs(c["price"].to_numpy()[:n] - f["price"])),
              "| RL hist", np.nanmax(np.abs(c["residual_load"].to_numpy()[:n] - f["rl"][:n])),
              "| RL D", np.nanmax(np.abs(c["residual_load"].to_numpy()[n:] - f["rl"][n:])),
              "| gas", np.nanmax(np.abs(c["gasprice"].to_numpy() - f["gas"].to_numpy())),
              "| wd", np.nanmax(np.abs(c["weekday"].to_numpy() - f["wd"])),
              "| NaN price hist", int(np.isnan(f["price"]).sum()))


def run(tag, days):
    data = Data()
    out = f"{SP}/P/market_ts2/{tag}"
    os.makedirs(out, exist_ok=True)
    for day in days:
        fn = f"{out}/{day}.npz"
        if os.path.exists(fn):
            continue
        t = time.time(); res = {}
        for v in ("repro96", "consist96"):
            f = build(day, data, v)
            res[v] = forecast(f); res[f"nlog_{v}"] = f["nlog"]
        np.savez(fn, **res)
        print(day, f"log days {res['nlog_consist96']}", f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2].split(","))
    else:
        run(sys.argv[2], sys.argv[3].split(","))
