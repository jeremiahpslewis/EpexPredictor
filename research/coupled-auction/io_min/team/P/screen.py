"""Agent P: screen tree variants on a dump (dump.py). Stage 2 is refit at each dump origin (a day with a .train
frame) and forecasts every dumped D up to the next origin. Writes SP/P/screen/<tag>/<variant>.npz (day -> forecast).
Usage: screen.py TAG VARIANT[,VARIANT...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/EpexPredictor')
from treelab import SP, load_dump, fit_predict, price_series

TZ = "Europe/Berlin"


def add_price_lags(X, pr, prefix="p", samedaytype=False):
    """price features known at the gate for every slot of D: same slot 1 and 7 days earlier, and the D-1 local-day mean
    (all D-1 slots are known at the gate: D-1 prices come from the D-2 auction)"""
    idx = X.index
    X = X.copy()
    X[f"{prefix}_lag1"] = pr.reindex(idx - pd.Timedelta(days=1)).to_numpy()
    X[f"{prefix}_lag7"] = pr.reindex(idx - pd.Timedelta(days=7)).to_numpy()
    loc = idx.tz_convert(TZ)
    dm = pr.groupby(pr.index.tz_convert(TZ).normalize()).mean()
    prev = loc.normalize() - pd.Timedelta(days=1)
    X[f"{prefix}_d1mean"] = dm.reindex(prev).to_numpy()
    if samedaytype:
        # same slot on the latest earlier day of the same type (Mon-Fri vs Sat/Sun): Friday for a Monday,
        # D-1 for Tue-Fri, D-1 for Sunday, D-6 (last Sunday) for Saturday; always known at the gate
        wd = loc.weekday.to_numpy()
        back = np.where(wd == 0, 3, np.where(wd == 5, 6, 1))
        X[f"{prefix}_sametype"] = pr.reindex(idx - pd.to_timedelta(back, unit="D")).to_numpy()
        X[f"{prefix}_sametype_mean"] = dm.reindex(loc.normalize() - pd.to_timedelta(back, unit="D")).to_numpy()
    return X


def variant(name, Xtr, ytr, XD, ctx):
    one = {"num_threads": 1}
    if name == "base":
        return fit_predict(Xtr, ytr, XD, one)
    if name == "l1":
        return fit_predict(Xtr, ytr, XD, {**one, "objective": "l1"}, obj_scale=1)
    if name == "huber":
        return fit_predict(Xtr, ytr, XD, {**one, "objective": "huber", "alpha": 2.0}, obj_scale=1)
    if name == "slow":
        return fit_predict(Xtr, ytr, XD, {**one, "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 50,
                                          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1}, rounds=500)
    if name == "slow_l1":
        return fit_predict(Xtr, ytr, XD, {**one, "objective": "l1", "learning_rate": 0.03, "num_leaves": 63,
                                          "min_data_in_leaf": 50, "feature_fraction": 0.7, "bagging_fraction": 0.8,
                                          "bagging_freq": 1}, rounds=500, obj_scale=1)
    if name.startswith("recency"):        # recencyN: weight exp(-age / N days)
        tau = float(name[7:])
        age = (Xtr["ts"].max() - Xtr["ts"]) / 86400.0
        return fit_predict(Xtr, ytr, XD, one, weight=np.exp(-age / tau).to_numpy())
    if name == "nots":
        return fit_predict(Xtr.drop(columns=["ts"]), ytr, XD.drop(columns=["ts"]), one)
    if name == "gasonly":
        return fit_predict(Xtr, ytr, XD, one, raw_model=False)
    if name == "lags":
        pr = ctx["pr"]
        return fit_predict(add_price_lags(Xtr, pr), ytr, add_price_lags(XD, pr), one)
    if name == "lagsst":
        pr = ctx["pr"]
        return fit_predict(add_price_lags(Xtr, pr, samedaytype=True), ytr, add_price_lags(XD, pr, samedaytype=True), one)
    if name.startswith("avg_"):            # avg_<a>+<b>: mean of two variants' prices
        a, b = name[4:].split("+")
        return 0.5 * (variant(a, Xtr, ytr, XD, ctx) + variant(b, Xtr, ytr, XD, ctx))
    if name.startswith("shift") and "_lags" in name:
        pr = ctx["pr"]
        return variant(name.replace("_lags", ""), add_price_lags(Xtr, pr), ytr, add_price_lags(XD, pr), ctx)
    if name.startswith("shift"):
        # structural target: the supply shift [GW] of marketmodel (as the market route defines it, with the frame's
        # gas price and residual load) is learned by the trees and cleared back to a price on D
        import lightgbm as lgb
        from predictor.model import marketmodel as mm
        def shift_of(X, y):
            return mm.supply_shift(X["residual_load"].to_numpy() - 8.0, y.to_numpy() * 10.0, X["gasprice"].to_numpy(), 75.0)
        s = shift_of(Xtr, ytr)
        ok = np.isfinite(s) & (np.abs(ytr.to_numpy()) > 0.02)
        p = {"num_threads": 1, "force_col_wise": True, "verbosity": -1}
        rounds = 100
        if "l1" in name:
            p["objective"] = "l1"
        if "slow" in name:
            p.update({"learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.7,
                      "bagging_fraction": 0.8, "bagging_freq": 1}); rounds = 500
        m = lgb.train(p, lgb.Dataset(Xtr[ok], label=s[ok]), num_boost_round=rounds)
        sh = m.predict(XD)
        return mm.clearing_price(XD["residual_load"].to_numpy() - 8.0, XD["gasprice"].to_numpy(), 75.0, shift=sh) / 10.0
    raise ValueError(name)


def main(tag, names, dfile="D"):
    Ds, Ts = load_dump(tag, dfile)
    origins = sorted(Ts)
    ctx = {"pr": price_series()}
    out = f"{SP}/P/screen/{tag}" + ("" if dfile == "D" else f"_{dfile}")
    os.makedirs(out, exist_ok=True)
    for name in names:
        fn = f"{out}/{name}.npz"
        if os.path.exists(fn):
            continue
        t = time.time()
        res = {}
        for i, o in enumerate(origins):
            nxt = origins[i + 1] if i + 1 < len(origins) else "9999"
            days = [d for d in Ds if o <= d < nxt]
            tr = pd.read_parquet(Ts[o]).astype("float64").dropna()
            Xtr, ytr = tr.drop(columns=["price"]), tr["price"]
            XD = pd.concat([pd.read_parquet(Ds[d]) for d in days]).astype("float64")
            act = XD.pop("actual"); XD = XD[Xtr.columns]
            p = variant(name, Xtr, ytr, XD, ctx)
            for d in days:
                m = (XD.index.tz_convert(TZ).strftime("%Y-%m-%d") == d)
                res[d] = np.c_[XD.index.asi8[m], p[m], act.to_numpy()[m]]
        np.savez(fn, **res)
        e = np.mean([np.abs(v[:96, 1] - v[:96, 2]).mean() for v in res.values()])
        print(f"{tag} {name}: MAE {e:.3f} over {len(res)} days ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else "D")
