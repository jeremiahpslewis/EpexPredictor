"""One RL-forecast candidate, one protocol, one period. Saves per-cutoff component forecasts for several test weather vintages
and the forecast of D-1 (for the residual correction, scored in score.py).
usage: run.py CAND PROTO [PERIOD]   PROTO rolling|static, PERIOD autumn|spring
"""
import sys, time
import numpy as np, pandas as pd
import common as c

CANDS = {
    "base":    dict(train="d0"),
    "trd2":    dict(train="d2h"),
    "stack":   dict(train="stack"),
    "trd1":    dict(train="hyb1"),
    "w80":     dict(train="d0", w80=True),
    "split":   dict(train="d0", split=True),
    "recency": dict(train="d0", hl=90.0),
    # combinations, defined only after the single candidates are scored (see EXPECTATIONS.md)
    "nots":    dict(train="d0", nots=True),
    "doy":     dict(train="d0", nots=True, doy=True),
    "trd2_nots": dict(train="d2h", nots=True),
    "trd1_nots": dict(train="hyb1", nots=True),
    "stack_nots": dict(train="stack", nots=True),
    "trd2_rec":  dict(train="d2h", hl=90.0),
    "trd1_rec":  dict(train="hyb1", hl=90.0),
}
cand, proto = sys.argv[1], sys.argv[2]; period = sys.argv[3] if len(sys.argv) > 3 else "autumn"; C = CANDS[cand]
A = c.load_all(); idx, aux, fc, W = A["idx"], A["aux"], A["fc"], A["W"]
if period == "autumn":
    cuts = A["cuts"]; stride = 7; tests = ("gate", "d1", "d2", "d0")
else:
    cuts = []; d = pd.Timestamp("2026-04-01", tz="Europe/Berlin")
    while d <= pd.Timestamp("2026-08-28", tz="Europe/Berlin"):
        cuts.append(idx.get_loc(d.tz_convert("UTC"))); d += pd.DateOffset(days=1)
    cuts = np.array(cuts); stride = 14; tests = ("d2h", "d0")
w80 = C.get("w80", False)
X = {k: c.feats(W[k], aux, w80=w80) for k in ("d0", "d2h", "d1", "d2", "d3")}
if C.get("nots"):
    loc = idx.tz_convert("Europe/Berlin"); ang = 2 * np.pi * (loc.dayofyear.to_numpy() - 1) / 365.25
    for k in X:
        X[k] = X[k].drop(columns=["ts"])
        if C.get("doy"): X[k]["doy_sin"] = np.sin(ang); X[k]["doy_cos"] = np.cos(ang)
cols = X["d0"].columns
TG = ["wind_onshore", "wind_offshore", "solar", "load"] if C.get("split") else ["wind", "solar", "load"]
start = idx.get_loc(pd.Timestamp("2025-10-01", tz="UTC"))


def train_set(end):
    rows = np.arange(start, end); tw = C["train"]
    if tw == "d0": Xs = [X["d0"].to_numpy()[rows]]; R = [rows]
    elif tw == "d2h": Xs = [X["d2h"].to_numpy()[rows]]; R = [rows]
    elif tw == "stack": Xs = [X["d0"].to_numpy()[rows], X["d2h"].to_numpy()[rows]]; R = [rows, rows]
    elif tw == "hyb1":
        a = X["d1"].to_numpy()[rows]; b = X["d2h"].to_numpy()[rows]; has = ~np.isnan(a).any(1)
        Xs = [np.where(has[:, None], a, b)]; R = [rows]
    Xt = np.vstack(Xs); rr = np.concatenate(R)
    w = None
    if C.get("hl"):
        age = (end - rr) / 96.0; w = 0.5 ** (age / C["hl"])
    return Xt, rr, w


def models(end):
    Xt, rr, w = train_set(end)
    return {t: c.fit(Xt, fc[t].to_numpy()[rr], w) for t in TG}


def predict(M, Xr):
    p = {t: M[t].predict(Xr) for t in TG}
    if "wind_onshore" in p: p["wind"] = p.pop("wind_onshore") + p.pop("wind_offshore")
    return p


def rows_for(kind, d0):
    if kind == "gate":
        a = X["d1"].to_numpy()[d0:d0 + c.HZ]; b = X["d2"].to_numpy()[d0:d0 + c.HZ]
        return np.where((np.arange(c.HZ) < c.GATE_D1)[:, None], a, b)
    return X[kind].to_numpy()[d0:d0 + c.HZ]


out = {}; t0 = time.time(); M = None
static_end = idx.get_loc(pd.Timestamp("2026-08-31 22:00", tz="UTC"))     # up to the end of 31 Aug local
for j, d0 in enumerate(cuts):
    d0 = int(d0)
    if proto == "static":
        if M is None: M = models(static_end)
    elif j % stride == 0:
        M = models(d0 - 144)           # up to noon of D-2 of the refit day, so D-1 of every later day is out of sample
    for k in tests:
        for t, v in predict(M, rows_for(k, d0)).items(): out.setdefault(f"{t}_{k}", []).append(v)
    for k in (("d1", "d0") if period == "autumn" else ("d2h", "d0")):        # forecast of D-1 (residual correction)
        for t, v in predict(M, X[k].to_numpy()[d0 - 96:d0]).items(): out.setdefault(f"{t}_prev_{k}", []).append(v)
    for t in ("wind", "solar", "load"):
        out.setdefault(f"{t}_true", []).append(fc[t].to_numpy()[d0:d0 + c.HZ]); out.setdefault(f"{t}_prevtrue", []).append(fc[t].to_numpy()[d0 - 96:d0])
    if j % 10 == 0: print(cand, proto, period, j, f"{time.time() - t0:.0f}s", flush=True)
R = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
np.savez_compressed(f"{c.OUT}/{cand}_{proto}_{period}.npz", cuts=np.asarray(cuts), hours=np.stack([c.local_hour(idx, int(d)) for d in cuts]), **R)
rl = R["load_true"] - R["wind_true"] - R["solar_true"]
for k in tests:
    f = R[f"load_{k}"] - R[f"wind_{k}"] - R[f"solar_{k}"]; print(cand, proto, period, k, {a: round(b, 3) for a, b in c.mae_seg(f, rl).items()}, flush=True)
print("done", f"{time.time() - t0:.0f}s")
