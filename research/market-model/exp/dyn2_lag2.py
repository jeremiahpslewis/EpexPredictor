"""Online correction, second design: the correction of the curve depends on the effective residual load x AND the hour
(kernel over both, exponential forgetting, shrunk to zero without recent data). Memory and shrinkage are chosen per
forecast on the error history (walk forward). usage: dyn2.py PERIOD RUN [ARMS] [MODE]   MODE: xh | x+xh"""
import sys, os, numpy as np
import live_common as lc
SP = lc.SP; period, run = sys.argv[1], sys.argv[2]; ARMS = (sys.argv[3] if len(sys.argv) > 3 else "known,ts").split(","); MODE = sys.argv[4] if len(sys.argv) > 4 else "xh"
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]; a = I["actual"].astype(float)
F = lc.load(); hour_all = F["DE"].index.tz_convert("Europe/Berlin").hour.to_numpy()
Z = [np.load(f"{SP}/out/struct_{run}/{t}.npz") for t in tags]
C = -7.5 + 5.0 * np.arange(17); BW = 4.0; LAG = 2; VAL = 120; HH = np.arange(24)
def kern(x): return np.exp(-0.5 * ((x[..., None] - C) / BW) ** 2)
def kh(bw):
    d = np.abs(HH[:, None] - HH[None, :]); d = np.minimum(d, 24 - d); return np.exp(-0.5 * (d / bw) ** 2)        # (24, 24)
def walk(R, X, tau1, lam1, tau2, lam2, bwh, use1=True):
    n = len(R); a1 = 0.5 ** (1 / tau1); a2 = 0.5 ** (1 / tau2); KH = kh(bwh)
    Dn = np.zeros(len(C)); Dd = np.zeros(len(C)); En = np.zeros((len(C), 24)); Ed = np.zeros((len(C), 24)); st = []; pred = np.zeros_like(R)
    for v in range(n):
        st.append((Dn / (Dd + lam1), En / (Ed + lam2)))
        D, E = st[v - LAG + 1] if v - LAG + 1 >= 0 else (np.zeros(len(C)), np.zeros((len(C), 24)))
        K = kern(X[v]); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9); ok = np.isfinite(R[v]) & np.isfinite(X[v]); Kn = np.where(ok[:, None], Kn, 0.0)
        p1 = (Kn @ D) * use1; p2 = np.einsum("hb,bh->h", Kn, E); pred[v] = p1 + p2
        r = np.where(ok, R[v], 0.0); e1 = r - p1; K0 = K * ok[:, None]
        Dn = a1 * Dn + (K0 * r[:, None]).sum(0) * use1; Dd = a1 * Dd + K0.sum(0)
        En = a2 * En + np.einsum("hb,hg->bg", K0 * e1[:, None], KH); Ed = a2 * Ed + np.einsum("hb,hg->bg", K0, KH)
    return pred, (Dn / (Dd + lam1), En / (Ed + lam2))
P = {arm: [] for arm in ARMS}; chosen = []
for j, z in enumerate(Z):
    R = z["R1"].astype(float)[:-1] / 10; X = z["XH1"].astype(float)[:-1]; lo, hi = np.nanpercentile(R, [1, 99]); R = np.clip(R, lo, hi); n = len(R); sl = slice(max(20, n - VAL), n)
    sc = lambda pred: np.nanmean(np.abs(R[sl] - pred[sl])); best = (np.nanmean(np.abs(R[sl])), None)
    grid = [(t1, 20, t2, l2, bw) for t1 in ((20, 60) if MODE == "x+xh" else (5,)) for t2 in (1.5, 3, 6, 12) for l2 in (1, 2, 8) for bw in (1.0, 1.5, 3.0)]
    for g in grid:
        s_ = sc(walk(R, X, *g, use1=(MODE == "x+xh"))[0]); best = min(best, (s_, g), key=lambda t: t[0])
    g = best[1]; chosen.append(list(g) if g else [0] * 5)
    if g is None: D, E = np.zeros(len(C)), np.zeros((len(C), 24))
    else: _, (D, E) = walk(R, X, *g, use1=(MODE == "x+xh"))
    d0 = int(cuts[j]); hh = hour_all[d0:d0 + lc.HZ]
    for arm in ARMS:
        x = z[f"x_{arm}"].astype(float); K = kern(x); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9)
        P[arm].append(z[f"{arm}_shape"].astype(float) + (Kn @ D) * (MODE == "x+xh") + np.einsum("qb,bq->q", Kn, E[:, hh]))
ch = np.array(chosen, float); print(f"{period} {run} [{MODE}]: median memory {np.median(ch[:, 2]):.0f} d, shrinkage {np.median(ch[:, 3]):.0f}, hour width {np.median(ch[:, 4]):.1f} h")
def sc(p): e = np.abs(p - a); return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():7.2f}{e[:, 56:96].mean():7.2f}"
for arm in ARMS:
    p = np.stack(P[arm]); old = np.load(f"{SP}/out/structfc_{run}_{arm}_dyn.npz")
    print(f"  {arm:<6} night 34-48h 48-58h | curve only {sc(old['curve'].astype(float))} | x + hour profile {sc(old['vint'].astype(float))} | this design {sc(p)}")
    np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_lag2{MODE.replace('+', '')}.npz", vint=p.astype(np.float32), curve=old["curve"], cutoffs=I["cutoffs"])
    if arm == "known":
        rl = I["rl_true"].astype(float)[:, :96]; A = a[:, :96]; hr = np.stack([hour_all[int(c):int(c) + 96] for c in cuts])
        print("         first day, MAE (bias): " + "  ".join(f"{lab}: {np.abs(p[:, :96] - A)[m].mean():.2f} ({(p[:, :96] - A)[m].mean():+.2f})" for lab, m in (("<25 GW", rl < 25), ("25-45", (rl >= 25) & (rl < 45)), (">=45", rl >= 45), ("00-06 h", hr < 6), ("06-10 h", (hr >= 6) & (hr < 10)), ("16-20 h", (hr >= 16) & (hr < 20)))))
