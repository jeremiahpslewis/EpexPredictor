"""Blend weight that depends on the load range (notes 17.4): candidate = w(x) * structural + (1 - w(x)) * Timestral price, w(x) = sigmoid(a + b (x - 30) / 15),
learned walk-forward from all earlier forecasts with known outcome. Writes out/candw_<period>.npz and prints scores against the plain average."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import os
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; LAM, N0, LAGD = 0.01, 15, 2; HZ = 32 if os.environ.get("NIGHT") == "1" else 56; TAG = "candwn" if HZ == 32 else "candw"
CFG = [("winter", "hw2", "win_ts", "median"), ("summer", "hs2", "ssn_ts", "median"), ("autumn", "ha", "tsl_123_8k", "vint")]
sg = lambda u: 1 / (1 + np.exp(-u)); AG, BG = np.meshgrid(np.arange(-2, 2.01, 0.25), np.arange(-3, 3.01, 0.25), indexing="ij"); AG, BG = AG.ravel(), BG.ravel()
R = []
for period, run, tsf, tsk in CFG:
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); tags = [str(c)[:10] for c in I["cutoffs"]]; s = np.load(f"{SP}/out/structfc_{run}_ts_dyn2xxh.npz")["vint"].astype(float); m = np.load(f"{SP}/out/{tsf}.npz")[tsk].astype(float)
    for j in range(len(a)):
        z = np.load(f"{SP}/out/struct_{run}/{tags[j]}.npz"); x = (z["x_ts"] if "x_ts" in z.files else z["x_known"]).astype(float)
        R.append(dict(period=period, j=j, day=np.datetime64(tags[j]), x=x[:153], a=a[j], s=s[j], m=m[j]))
R.sort(key=lambda r: r["day"]); X = np.array([r["x"][:HZ] for r in R]); A = np.array([r["a"][:HZ] for r in R]); S = np.array([r["s"][:HZ] for r in R]); M = np.array([r["m"][:HZ] for r in R]); days = np.array([r["day"] for r in R])
def learn(k):
    """a, b from the forecasts k (indices) by grid search"""
    x, a_, s_, m_ = X[k].ravel(), A[k].ravel(), S[k].ravel(), M[k].ravel(); ok = np.isfinite(x) & np.isfinite(a_) & np.isfinite(s_) & np.isfinite(m_); x, a_, s_, m_ = x[ok], a_[ok], s_[ok], m_[ok]
    W = sg(AG[:, None] + BG[:, None] * (x[None] - 30) / 15); obj = np.abs(W * s_[None] + (1 - W) * m_[None] - a_[None]).mean(1) + LAM * (AG ** 2 + BG ** 2); i = int(np.argmin(obj)); return AG[i], BG[i]
par = np.zeros((len(R), 2)); out = {p: {} for p, *_ in CFG}
for i, r in enumerate(R):
    k = np.flatnonzero(days <= r["day"] - np.timedelta64(LAGD, "D")); ab = learn(k) if len(k) >= N0 else (0.0, 0.0); par[i] = ab
    w = sg(ab[0] + ab[1] * (np.nan_to_num(r["x"], nan=30.0) - 30) / 15); w = np.where(np.arange(len(w)) < HZ, w, 0.5) if HZ == 32 else w; out[r["period"]][r["j"]] = (w * r["s"] + (1 - w) * r["m"], w)
for period, run, tsf, tsk in CFG:
    I = np.load(f"{SP}/out/inputs_{period}.npz"); n = len(I["actual"]); v = np.stack([out[period][j][0] for j in range(n)]); w = np.stack([out[period][j][1] for j in range(n)]); k = [i for i, r in enumerate(R) if r["period"] == period]
    np.savez_compressed(f"{SP}/out/{TAG}_{period}.npz", vint=v.astype(np.float32), w=w.astype(np.float32), cutoffs=I["cutoffs"], a=par[k, 0], b=par[k, 1])
    print(f"{period}: a first / median / last {par[k[0], 0]:+.2f} / {np.median(par[k, 0]):+.2f} / {par[k[-1], 0]:+.2f}, b {par[k[0], 1]:+.2f} / {np.median(par[k, 1]):+.2f} / {par[k[-1], 1]:+.2f}; mean weight on structural {w[:, :HZ].mean():.2f}; at x = 10 / 30 / 50 GW with the last parameters: " + " / ".join(f"{sg(par[k[-1], 0] + par[k[-1], 1] * (xx - 30) / 15):.2f}" for xx in (10, 30, 50)))
ab = learn(np.arange(len(R))); print(f"all 146 forecasts in hindsight ({'night' if HZ == 32 else 'first 14 h'}): a {ab[0]:+.2f}, b {ab[1]:+.2f}; weight at 10 / 30 / 50 GW: " + " / ".join(f"{sg(ab[0] + ab[1] * (xx - 30) / 15):.2f}" for xx in (10, 30, 50)))
