"""Online correction with a shift of the effective residual load.
Where the curve is steep (between the conventional plateau and the surplus regime) a price error is mostly an error of
the position on the curve, a few GW of effective residual load. So the first layer is a shift Delta [GW], learned from
recent errors weighted by the local slope of the curve s = d price / d x (flat parts say nothing about it):
    Delta[hour] = sum(w s r) / (sum(w s^2) + lam),  w: exponential forgetting and a kernel over the hour
and applied through the curve: curve(x + Delta) - curve(x). The layers of dyn2 (slow in x, fast in x and hour) follow on
what is left. Memory and shrinkage of each layer are chosen per forecast on the error history (walk forward).
usage: dyn4.py PERIOD RUN [ARMS] [off]"""
import sys, os, numpy as np
import live_common as lc
SP = lc.SP; period, run = sys.argv[1], sys.argv[2]; ARMS = (sys.argv[3] if len(sys.argv) > 3 else "known,ts").split(","); XS = not (len(sys.argv) > 4 and sys.argv[4] == "off")
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]; a = I["actual"].astype(float)
F = lc.load(); hour_all = F["DE"].index.tz_convert("Europe/Berlin").hour.to_numpy(); gas_all = F["DE"]["gasprice"].to_numpy(float)
Z = [np.load(f"{SP}/out/struct_{run}/{t}.npz") for t in tags]
C = -7.5 + 5.0 * np.arange(17); BW = 4.0; VAL = 120; HH = np.arange(24); DMAX = 8.0; BWH_X = float(os.environ.get("BWH_X", 4.0))
def kern(x): return np.exp(-0.5 * ((x[..., None] - C) / BW) ** 2)
def kh(bw):
    d = np.abs(HH[:, None] - HH[None, :]); d = np.minimum(d, 24 - d); return np.exp(-0.5 * (d / bw) ** 2)
def curve_fns(z):
    g = z["grid"].astype(float); b = z["b_curve"].astype(float); h = z["h_curve"].astype(float); db = np.diff(b) / np.diff(g); dh = np.diff(h) / np.diff(g)
    def val(x, gas):
        xc = np.clip(x, g[0], g[-1]); return (np.interp(xc, g, b) + gas * np.interp(xc, g, h) + np.minimum(x - g[0], 0) * db[0] + np.maximum(x - g[-1], 0) * (db[-1] + gas * dh[-1])) / 10
    def slope(x, gas):
        k = np.clip(np.searchsorted(g, x, side="right") - 1, 0, len(db) - 1); return (db[k] + gas * dh[k]) / 10
    return val, slope
def shift_walk(R, X, G, val, slope, tau, lam):
    """per day: Delta from the days before, the error left after applying it; returns left-over errors and the final Delta"""
    n = len(R); al = 0.5 ** (1 / tau); KH = kh(BWH_X); Sn = np.zeros(24); Sd = np.zeros(24); left = R.copy()
    for v in range(n):
        dl = np.clip(Sn / (Sd + lam), -DMAX, DMAX); ok = np.isfinite(R[v]) & np.isfinite(X[v]); xv = np.where(ok, X[v], 0.0)
        left[v] = R[v] - (val(xv + dl, G[v]) - val(xv, G[v]))
        s = np.where(ok, slope(xv, G[v]), 0.0); r = np.where(ok, R[v], 0.0)
        Sn = al * Sn + KH @ (s * r); Sd = al * Sd + KH @ (s * s)
    return left, np.clip(Sn / (Sd + lam), -DMAX, DMAX)
def walk(R, X, tau1, lam1, tau2, lam2, bwh):
    n = len(R); a1 = 0.5 ** (1 / tau1); a2 = 0.5 ** (1 / tau2); KH = kh(bwh)
    Dn = np.zeros(len(C)); Dd = np.zeros(len(C)); En = np.zeros((len(C), 24)); Ed = np.zeros((len(C), 24)); pred = np.zeros_like(R)
    for v in range(n):
        D, E = Dn / (Dd + lam1), En / (Ed + lam2)
        ok = np.isfinite(R[v]) & np.isfinite(X[v]); K = kern(np.where(ok, X[v], 0.0)); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9); Kn = np.where(ok[:, None], Kn, 0.0)
        p1 = Kn @ D; p2 = np.einsum("hb,bh->h", Kn, E); pred[v] = p1 + p2
        r = np.where(ok, R[v], 0.0); e1 = r - p1; K0 = K * ok[:, None]
        Dn = a1 * Dn + (K0 * r[:, None]).sum(0); Dd = a1 * Dd + K0.sum(0)
        En = a2 * En + np.einsum("hb,hg->bg", K0 * e1[:, None], KH); Ed = a2 * Ed + np.einsum("hb,hg->bg", K0, KH)
    return pred, (Dn / (Dd + lam1), En / (Ed + lam2))
P = {arm: [] for arm in ARMS}; chosen = []; deltas = []
for j, z in enumerate(Z):
    R = z["R1"].astype(float) / 10; X = z["XH1"].astype(float); lo, hi = np.nanpercentile(R, [1, 99]); R = np.clip(R, lo, hi); n = len(R); sl = slice(max(20, n - VAL), n)
    d0 = int(cuts[j]); extra = len(z["R1"]) > len(z["R"]); start = d0 - 96 * (n if extra else n + 1) + 96 * np.arange(n); G = np.repeat(gas_all[np.clip(start, 0, None)][:, None], 24, 1)
    if extra: G[-1] = float(I["gas"][j])
    G = np.where(np.isfinite(G), G, float(I["gas"][j])); val, slope = curve_fns(z); gj = float(I["gas"][j])
    sc = lambda left: np.nanmean(np.abs(left[sl])); bestx = (sc(R), None, R, np.zeros(24))
    if XS:
        for tau in (2, 5, 15):
            for lam in (0.03, 0.1, 0.3):
                left, dl = shift_walk(R, X, G, val, slope, tau, lam)
                if sc(left) < bestx[0]: bestx = (sc(left), (tau, lam), left, dl)
    R2, DL = bestx[2], bestx[3]; best = (sc(R2), None)
    for g in [(t1, 20, t2, l2, bw) for t1 in (20, 60) for t2 in (1.5, 3, 6, 12) for l2 in (1, 2, 8) for bw in (1.0, 1.5, 3.0)]:
        s_ = np.nanmean(np.abs(R2[sl] - walk(R2, X, *g)[0][sl])); best = min(best, (s_, g), key=lambda t: t[0])
    g = best[1]; chosen.append((list(bestx[1]) if bestx[1] else [0, 0]) + (list(g) if g else [0] * 5)); deltas.append(DL)
    D, E = (np.zeros(len(C)), np.zeros((len(C), 24))) if g is None else walk(R2, X, *g)[1]
    hh = hour_all[d0:d0 + lc.HZ]
    for arm in ARMS:
        x = z[f"x_{arm}"].astype(float); K = kern(x); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9)
        P[arm].append(z[f"{arm}_shape"].astype(float) + (val(x + DL[hh], gj) - val(x, gj)) + Kn @ D + np.einsum("qb,bq->q", Kn, E[:, hh]))
ch = np.array(chosen, float); dl = np.array(deltas)
print(f"{period} {run} [shift {'on' if XS else 'off'}]: shift used in {100 * (ch[:, 0] > 0).mean():.0f}% of forecasts, median memory {np.median(ch[ch[:, 0] > 0, 0]) if (ch[:, 0] > 0).any() else 0:.0f} d; shift at night: mean {dl[:, :8].mean():+.1f} GW, mean |.| {np.abs(dl[:, :8]).mean():.1f}, range {dl[:, :8].min():+.1f}..{dl[:, :8].max():+.1f}; fast layer memory {np.median(ch[:, 4]):.0f} d")
def sc(p): e = np.abs(p - a); return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():7.2f}{e[:, 56:96].mean():7.2f}"
for arm in ARMS:
    p = np.stack(P[arm]); old = np.load(f"{SP}/out/structfc_{run}_{arm}_dyn2xxh.npz")
    print(f"  {arm:<6} night 34-48h 48-58h | curve only {sc(old['curve'].astype(float))} | online correction as before {sc(old['vint'].astype(float))} | with shift layer {sc(p)}")
    np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_dyn4{'' if XS else 'off'}.npz", vint=p.astype(np.float32), curve=old["curve"], cutoffs=I["cutoffs"], delta=dl.astype(np.float32))
