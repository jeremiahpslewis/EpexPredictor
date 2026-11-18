"""Online layer of the structural model as a structural time series (state space) model on its errors.

    error[day, hour] = Z(x, hour) . state[day] + noise,      state[day+1] = rho * state[day] + innovation

  state, one value per day (the auction clears once a day; the 24 hours are several observations of the same state):
    level                         1
    hour-of-day profile           2K   trigonometric seasonal, K harmonics
    correction of the curve       B    radial basis functions of the effective residual load x, slow (rho 0.99: drift over weeks)
    correction of the curve       B    the same basis, fast (days)
    low-load x time-of-day        16   the four lowest-load basis functions times the first two harmonics
  rho < 1 pulls every state back to zero when it is not observed (mean reversion); innovation variances per group; the
  slow curve states have a fixed rho, all others share an estimated one.
  The filter is the online update: each day's errors revise the state, with a robust (down-weighted outlier) observation
  noise. Hyperparameters (4 innovation scales, rho, noise scale) are maximum likelihood on the error history of each
  forecast (prediction error decomposition), started from the previous forecast's values.
  The newest day (prices known, published residual load not) enters with inflated noise, as its x is an own forecast.
usage: dyn3.py PERIOD RUN [ARMS] [KEY]"""
import sys, os, numpy as np
from multiprocessing import get_context
from scipy.optimize import minimize
from scipy import stats

K_H, CEN, BWX = 4, -5.0 + 10.0 * np.arange(8), 6.0
GROUPS = np.r_[0, np.ones(2 * K_H, int), 2 * np.ones(len(CEN), int), 3 * np.ones(len(CEN), int), 4 * np.ones(16, int)]; P_DIM = len(GROUPS); RHO_SLOW = 0.99
def design(x, hour):
    x = np.asarray(x, float); h = np.asarray(hour, float); cols = [np.ones_like(x)]
    for k in range(1, K_H + 1): cols += [np.cos(2 * np.pi * k * h / 24), np.sin(2 * np.pi * k * h / 24)]
    phi = np.exp(-0.5 * ((x[..., None] - CEN) / BWX) ** 2); phi = phi / np.maximum(phi.sum(-1, keepdims=True), 1e-9)
    cols += [phi[..., b] for b in range(len(CEN))] * 2
    for b in range(4):
        for k in (1, 2): cols += [phi[..., b] * np.cos(2 * np.pi * k * h / 24), phi[..., b] * np.sin(2 * np.pi * k * h / 24)]
    return np.stack(cols, -1)
def kalman(theta, R, Zs, infl, want_state=False):
    q = np.exp(theta[:5])[GROUPS] ** 2; rho = np.where(GROUPS == 2, RHO_SLOW, 1 / (1 + np.exp(-theta[5]))); s2 = np.exp(theta[6]) ** 2
    a = np.zeros(P_DIM); P = np.diag(q / (1 - rho ** 2 + 1e-6)); ll = 0.0; n = len(R)
    for t in range(n):
        ok = np.isfinite(R[t]) & np.isfinite(Zs[t]).all(1)
        if ok.any():
            Z = Zs[t][ok]; v = R[t][ok] - Z @ a; S = Z @ P @ Z.T + s2 * infl[t] * np.eye(ok.sum())
            # robust: observations far out get more noise (two-pass weights from the first innovation)
            w = np.minimum(1.0, 2.5 * np.sqrt(np.diag(S)) / (np.abs(v) + 1e-9)); S = Z @ P @ Z.T + np.diag(s2 * infl[t] / w ** 2)
            L = np.linalg.cholesky(S + 1e-9 * np.eye(len(v))); Kt = np.linalg.solve(L.T, np.linalg.solve(L, Z @ P)).T
            ll += -0.5 * (np.sum(np.linalg.solve(L, v) ** 2) + 2 * np.log(np.diag(L)).sum())
            a = a + Kt @ v; P = P - Kt @ Z @ P; P = 0.5 * (P + P.T)
        a = rho * a; P = rho[:, None] * P * rho[None, :] + np.diag(q)
    return (a, P, rho, q, s2) if want_state else -ll
def work(args):
    period, run, arms, key, js = args
    import live_common as lc
    SP = lc.SP; I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]
    hour_all = lc.load()["DE"].index.tz_convert("Europe/Berlin").hour.to_numpy(); day_all = lc.load()["DE"].index.tz_convert("Europe/Berlin").normalize()
    out = []; theta = np.array([np.log(0.15), np.log(0.05), np.log(0.05), np.log(0.3), np.log(0.2), 0.5, np.log(0.6)])
    for j in js:
        z = np.load(f"{SP}/out/struct_{run}/{tags[j]}.npz"); R = z[key].astype(float) / 10; X = z["XH1" if key == "R1" else "XH1"].astype(float)
        if key == "R": X = X[:len(R)]
        lo, hi = np.nanpercentile(R, [0.5, 99.5]); R = np.clip(R, lo, hi); n = len(R); Zs = design(X, np.broadcast_to(np.arange(24), X.shape))
        infl = np.ones(n); infl[-1] = 2.0 if key == "R1" else 1.0
        res = minimize(kalman, theta, args=(R, Zs, infl), method="Nelder-Mead", options={"maxiter": 300, "xatol": 0.03, "fatol": 0.5}); theta = res.x
        a, P, rho, q, s2 = kalman(theta, R, Zs, infl, want_state=True)      # state predicted for the day after the newest one
        d0 = int(cuts[j]); hh = hour_all[d0:d0 + lc.HZ]; second = (day_all[d0:d0 + lc.HZ] != day_all[d0]); lag0 = 1 if key == "R1" else 2
        steps = np.where(second, lag0, lag0 - 1)                              # further days ahead of the first predicted day
        rec = {"j": j, "theta": theta.copy()}
        for arm in arms:
            x = z[f"x_{arm}"].astype(float); Zf = design(x, hh); mean = np.zeros(lc.HZ); var = np.zeros(lc.HZ)
            for st in np.unique(steps):
                m = steps == st; a_, P_ = a.copy(), P.copy()
                for _ in range(int(st)): a_ = rho * a_; P_ = rho[:, None] * P_ * rho[None, :] + np.diag(q)
                mean[m] = Zf[m] @ a_; var[m] = np.einsum("ij,jk,ik->i", Zf[m], P_, Zf[m]) + s2
            rec[arm] = z[f"{arm}_shape"].astype(float) + mean; rec[arm + "_sd"] = np.sqrt(var)
        out.append(rec)
    return out
if __name__ == "__main__":
    import live_common as lc
    period, run = sys.argv[1], sys.argv[2]; arms = (sys.argv[3] if len(sys.argv) > 3 else "known,ts").split(","); key = sys.argv[4] if len(sys.argv) > 4 else "R1"; SP = lc.SP
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); n = len(a); nsh = 4
    with get_context("spawn").Pool(nsh) as pool: res = pool.map(work, [(period, run, arms, key, list(range(k * n // nsh, (k + 1) * n // nsh))) for k in range(nsh)])
    res = sorted(sum(res, []), key=lambda r: r["j"]); th = np.array([r["theta"] for r in res])
    print(f"{period} {run} [{key}]: median innovation scales per day: level {np.median(np.exp(th[:, 0])):.2f}, hour profile {np.median(np.exp(th[:, 1])):.2f}, curve slow {np.median(np.exp(th[:, 2])):.2f}, curve fast {np.median(np.exp(th[:, 3])):.2f}, low-load x hour {np.median(np.exp(th[:, 4])):.2f} ct; rho of the fast states {np.median(1 / (1 + np.exp(-th[:, 5]))):.3f}; noise {np.median(np.exp(th[:, 6])):.2f} ct")
    def sc(p): e = np.abs(p - a); return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():7.2f}{e[:, 56:96].mean():7.2f}"
    for arm in arms:
        p = np.stack([r[arm] for r in res]); sd = np.stack([r[arm + "_sd"] for r in res]); old = np.load(f"{SP}/out/structfc_{run}_{arm}_dyn2xxh.npz") if os.path.exists(f"{SP}/out/structfc_{run}_{arm}_dyn2xxh.npz") else None
        y = a[:, :56]; cov = lambda zq: 100 * np.mean(np.abs(y - p[:, :56]) <= zq * sd[:, :56])
        print(f"  {arm:<6} night 34-48h 48-58h | state-space layer {sc(p)} | kernel layers {sc(old['vint'].astype(float)) if old is not None else ''} | curve only {sc(old['curve'].astype(float)) if old is not None else ''} | inside 80% / 90% band: {cov(stats.norm.ppf(0.9)):.0f}% / {cov(stats.norm.ppf(0.95)):.0f}%")
        np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_bsts.npz", vint=p.astype(np.float32), sd=sd.astype(np.float32), cutoffs=I["cutoffs"])
