"""MIDNIGHT ISSUE (notes 21.29): as ts_gen.py, but the context runs to the start of the target day, because at midnight the published figures
for the day before the target are out. Horizon = the target day onwards.
Timestral for the TSO day-ahead wind, solar and load figures (live setup).
Published figures are known through the end of the issue day D (they appear in the evening for the next day), so the
context ends there and the horizon is D+1 and D+2 (plus the start of D+3); D+2 onwards is scored.
Covariates: a few aggregates of the dense weather grid, as forecast (2 days old in the context; in the horizon the
freshest vintage that exists at 14:00 on D), holiday flag, weekday.
usage: ts_gen.py NAME PERIOD MODE [MODEL]   MODE joint (wind, solar, load as three variates) | rl (residual load directly)"""
import os, sys, time, glob
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc, dense_common as dc
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))
name, period, mode = sys.argv[1], sys.argv[2], sys.argv[3]; MODEL = sys.argv[4] if len(sys.argv) > 4 else "mistralai/Timestral-123M"; CTX, LEAD = 8192, 0
OUT = f"{lc.SP}/out/tsgen_{name}"; assert name.startswith("mid_"); os.makedirs(OUT, exist_ok=True)
F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx); I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = I["cuts"]
H = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{lc.SP}/dense/hist_*.parquet"))]).sort_index(); H = H[~H.index.duplicated()]; H15 = dc.to15(H, idx)
W2 = H15[[c for c in H15.columns if c.endswith("_d2")]].rename(columns=lambda c: c[:-3])
PREV = f"{lc.SP}/dense/prev.parquet"; P15 = dc.to15(pd.read_parquet(PREV), idx) if period == "autumn" else None
AGG = ["on_all", "on_north", "on_south", "off_ns", "off_baltic", "sw_all", "sw_south", "t_all"]
NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]; _nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); assert len(_nb) == len(idx)
T = {**{c: _nb[c].ffill().to_numpy(float) for c in NBC}, "wind": fc["wind"].to_numpy(float), "solar": fc["solar"].to_numpy(float), "load": fc["load"].to_numpy(float), "rl": fc["rl"].to_numpy(float)}
wd = D["weekday"].to_numpy(float); hol = (D["holiday"].to_numpy(float) >= 0.5).astype(float)
f = TimestralForecaster.from_pretrained(MODEL, revision={"mistralai/Timestral-123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7"}.get(MODEL))
t0 = time.time()
for j, d0 in enumerate(cuts):
    tag = str(idx[d0].date())
    if os.path.exists(f"{OUT}/{tag}.npz"): continue
    e = int(d0) - LEAD; a, b = e - CTX, int(d0) + lc.HZ; n = b - a; hz = b - e
    W = W2.iloc[a:b].copy()
    if P15 is not None:                                   # horizon: freshest forecast that exists at 14:00 on D
        r = np.arange(hz); k = np.where(r < 96, 1, np.where(r < 192, 2, 3)); k = np.minimum(k + int(os.environ.get("VSH", 0)), 3)   # VSH=1 (21.55): the 1-day-old forecast for the target day does not exist yet at a midnight issue
        for c in W.columns:
            W.iloc[CTX:, W.columns.get_loc(c)] = np.select([k == 1, k == 2, k == 3], [P15[f"{c}_d{q}"].to_numpy()[e:b] for q in (1, 2, 3)])
    X = dc.feats(W, D.iloc[a:b])[AGG].to_numpy(float).T                                       # (8, n)
    cov = np.vstack([X, hol[a:b][None]])
    names = ["wind", "solar", "load"] if mode == "joint" else ["wind", "solar", "load"] + NBC if mode == "jointnb" else ["rl"]
    ctx = np.stack([T[k][a:e] for k in names])[None]                                             # (1, V, CTX)
    with torch.no_grad():
        r_ = f.forecast(ctx, prediction_length=hz, known_covariates=cov[None], categorical_known_covariates=wd[a:b][None, None, :])
    med = np.asarray(r_.median)[0]; q = {p: np.asarray(r_.quantile(p))[0] for p in (0.05, 0.1, 0.9, 0.95)}
    np.savez_compressed(f"{OUT}/{tag}.npz", names=np.array(names), median=med[:, LEAD:], **{f"q{int(p*100):02d}": v[:, LEAD:] for p, v in q.items()}, true=np.stack([T[k][int(d0):b] for k in names]), d1=med[:, :LEAD], d1_true=np.stack([T[k][e:int(d0)] for k in names]))
    print(tag, f"{time.time() - t0:.0f}s", flush=True)
print("finished", flush=True)
