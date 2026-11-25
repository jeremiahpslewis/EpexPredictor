"""E3: base load model refit DAILY (app retrains every train()); load only. Compare with the weekly base and the D-1 decay correction."""
import numpy as np, pandas as pd, time
import common as c
A = c.load_all(); idx, aux, fc, W = A["idx"], A["aux"], A["fc"], A["W"]
X = {k: c.feats(W[k], aux).to_numpy() for k in ("d0", "d1", "d2")}; start = idx.get_loc(pd.Timestamp("2025-10-01", tz="UTC"))
y = fc["load"].to_numpy(); out = {"gate": [], "prev": [], "true": [], "prevtrue": []}; t0 = time.time()
for j, d0 in enumerate(A["cuts"].astype(int)):
    rows = np.arange(start, d0 - 144); m = c.fit(X["d0"][rows], y[rows])
    g = np.where((np.arange(c.HZ) < c.GATE_D1)[:, None], X["d1"][d0:d0 + c.HZ], X["d2"][d0:d0 + c.HZ])
    out["gate"].append(m.predict(g)); out["prev"].append(m.predict(X["d1"][d0 - 96:d0])); out["true"].append(y[d0:d0 + c.HZ]); out["prevtrue"].append(y[d0 - 96:d0])
    if j % 10 == 0: print(j, f"{time.time()-t0:.0f}s", flush=True)
R = {k: np.stack(v) for k, v in out.items()}; np.savez_compressed(f"{c.OUT}/load_daily_autumn.npz", **R)
B = np.load(f"{c.OUT}/base_rolling_autumn.npz")
dec = lambda f, p, pt: f + (pt - p)[:, -24:].mean(1, keepdims=True) * np.exp(-(np.arange(c.HZ) + 1) / 96.0)[None]
for n, f, p in (("weekly", B["load_gate"], B["load_prev_d1"]), ("daily", R["gate"], R["prev"])):
    for lab, ff in ((n, f), (n + "+decay", dec(f, p, R["prevtrue"]))):
        e = ff - R["true"]; print(f"{lab:<14} load MAE night {np.abs(e[:, :24]).mean():.2f} day1 {np.abs(e[:, :96]).mean():.2f} h24-38 {np.abs(e[:, 96:]).mean():.2f} bias day1 {e[:, :96].mean():+.2f}")
z = lambda f: -B["wind_gate"] - B["solar_gate"] + f; T = R["true"] - B["wind_true"] - B["solar_true"]
for s in c.SEG: print("RL with daily load vs weekly base", s, [round(v, 2) for v in c.boot(z(R["gate"]), z(B["load_gate"]), T, s)], "| daily+decay vs daily", [round(v, 2) for v in c.boot(z(dec(R["gate"], R["prev"], R["prevtrue"])), z(R["gate"]), T, s)])
