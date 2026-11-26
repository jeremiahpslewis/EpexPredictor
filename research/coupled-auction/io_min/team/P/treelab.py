"""Agent P: offline tree-variant lab on the dumped stage-2 design matrices (dump.py).

baseline() replicates PricePredictor.train_models/predict_with for the deployed stage-2 model: default LightGBM on the
raw price plus a gas-relative model (price / gas, cross prices / gas, weight gas^2), averaged. Variants change the
target, the features, the objective, the weights or the parameters; every variant gets the same rows.
"""
import os, glob
import numpy as np, pandas as pd, lightgbm as lgb

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
MIN_GAS = 1.0
BASE = {"force_col_wise": True, "verbosity": -1}


def load_dump(tag, dfile="D"):
    d = f"{SP}/P/dump/{tag}"
    Ds = {os.path.basename(f)[:10]: f for f in sorted(glob.glob(f"{d}/*.{dfile}.parquet"))}
    Ts = {os.path.basename(f)[:10]: f for f in sorted(glob.glob(f"{d}/*.train.parquet"))}
    return Ds, Ts


def price_series():
    """actual DE price (ct/kWh), UTC 15-min, from the harness working store"""
    s = pd.read_json(f"{SP}/fresh_data/stores/prices_v3_DE_LU.json.gz")["price"]
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    return s.sort_index()


def gas_rel(X, gas):
    X = X.copy()
    cp = [c for c in X.columns if c.startswith("cross_price_")]
    X[cp] = X[cp].div(gas, axis=0)
    return X


def fit_predict(Xtr, ytr, Xte, params=None, rounds=100, gas_model=True, raw_model=True, weight=None, obj_scale=None):
    """the app's pair of models; params override the defaults. Returns the averaged prediction on Xte."""
    p = {**BASE, **(params or {})}
    preds = []
    w = None if weight is None else np.asarray(weight)
    if raw_model:
        m = lgb.train(p, lgb.Dataset(Xtr, label=ytr, weight=w), num_boost_round=rounds)
        preds.append(m.predict(Xte))
    if gas_model and "gasprice" in Xtr.columns:
        g = Xtr["gasprice"].clip(lower=MIN_GAS); gt = Xte["gasprice"].clip(lower=MIN_GAS)
        # the app weights by gas^2 to keep a squared error in price units; for L1 the matching weight is gas
        gw = g.to_numpy() ** (obj_scale if obj_scale is not None else 2)
        if w is not None:
            gw = gw * w
        m = lgb.train(p, lgb.Dataset(gas_rel(Xtr, g), label=ytr / g, weight=gw), num_boost_round=rounds)
        preds.append(m.predict(gas_rel(Xte, gt)) * gt.to_numpy())
    return np.mean(preds, axis=0)
