"""Latent index-shift state for the index model: evidence, dynamics, filter.
Evidence per day: for each hour with a known price, the shift of the load index [GW] at which the fitted curve gives that price (bisection, +-15 GW; hits at the
bound are treated as missing); the day's evidence is the median over its hours. Built the way it exists at a forecast origin: published loads, no realised-load term.
State: s_d = phi * s_{d-1} + eta (local level with mean reversion), evidence z_d = s_d + eps_d. phi, var(eta), var(eps) by maximum likelihood on the training window.
Days without evidence are predicted through (the Kalman filter's missing-data step); the forecast uses the curve at the predicted state.
usage: ocx_state.py acf DATE | ocx_state.py fc PERIOD [EVERY]"""
import sys, os, time, numpy as np, pandas as pd
import oc, ocx2
from oc import *
from ocx2 import rows2, index2, pred2, fitx2, pcurve, hx
def implied(A, sl, rl=None):
    """hourly implied index shift for the hours in sl (NaN where no price or where the bound is hit); loads as published (or as given), no realised-load term"""
    n = sl.stop - sl.start; d = rows2(sl, rl=rl, drl=np.zeros(n)); a = d["p"].copy(); d = {k: np.nan_to_num(v) for k, v in d.items()}; x = index2(A, d, np); lo = np.full(n, -15.0); hi = np.full(n, 15.0)
    for _ in range(24):
        mid = 0.5 * (lo + hi); f = pcurve(A, x + mid, d["gas"], d["eua"], np)[0] - a; lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    dx = 0.5 * (lo + hi); bad = ~np.isfinite(a) | (np.abs(dx) > 14.8) | ~np.isfinite(RL[sl]).all(1) | ~np.isfinite(ocx2.SOL[sl]) | ~np.isfinite(oc.GAS[sl]); return np.where(bad, np.nan, dx)
def daily(dx, days):
    s = pd.Series(dx); g = s.groupby(days); z = g.median(); n = g.count(); return z.where(n >= 12), n
def kalman(z, phi, q, r, rmul=None):
    """scalar filter over a daily series with NaN for missing days; returns filtered mean, variance and the log likelihood"""
    m, v = 0.0, q / max(1 - phi * phi, 1e-3); ll = 0.0; ms, vs = [], []
    for i, zi in enumerate(z):
        m, v = phi * m, phi * phi * v + q
        if np.isfinite(zi):
            ri = r * (1.0 if rmul is None else rmul[i]); s_ = v + ri; k = v / s_; ll += -0.5 * (np.log(2 * np.pi * s_) + (zi - m) ** 2 / s_); m, v = m + k * (zi - m), (1 - k) * v
        ms.append(m); vs.append(v)
    return np.array(ms), np.array(vs), ll
def learn(z):
    best = (-1e18, 0.9, 1.0, 1.0)
    for phi in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995):
        for q in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
            for r in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                ll = kalman(z, phi, q, r)[2]
                if ll > best[0]: best = (ll, phi, q, r)
    return best[1:]
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "acf":
        h1 = hx(sys.argv[2]); A, _ = fitx2(h1); sl = slice(max(h1 - oc.TRAIN_H, 24 * 40), h1); dx = implied(A, sl); z, n = daily(dx, oc.DAY[sl]); zz = z.reindex(range(int(z.index.min()), int(z.index.max()) + 1)); v = zz.to_numpy()
        print(f"fit to the 540 days before {sys.argv[2]}: {np.isfinite(v).sum()} days with evidence of {len(v)}; daily implied index shift: sd {np.nanstd(v):.2f} GW; hours at the bound {100 * np.mean(~np.isfinite(dx)):.0f}%")
        ac = [pd.Series(v).autocorr(k) for k in range(1, 15)]; print("  autocorrelation at lags 1-14 days: " + " ".join(f"{a:.2f}" for a in ac)); print("  geometric from lag 1 would be:      " + " ".join(f"{ac[0] ** k:.2f}" for k in range(1, 15)))
        phi, q, r = learn(v); print(f"  maximum likelihood: persistence {phi}, state innovation variance {q}, evidence noise variance {r} -> steady-state weight on a new day's evidence {kalman(np.r_[np.zeros(200)], phi, q, r)[1][-1] / (kalman(np.r_[np.zeros(200)], phi, q, r)[1][-1] + r):.2f}; state sd {np.sqrt(q / (1 - phi * phi)):.2f} GW")
        wk = pd.Series(v).rolling(7, min_periods=4).mean(); print(f"  7-day mean of the evidence: sd {np.nanstd(wk):.2f} GW; its autocorrelation at 7 and 14 days: {wk.autocorr(7):.2f} {wk.autocorr(14):.2f}")
