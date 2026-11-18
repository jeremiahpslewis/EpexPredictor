"""Online correction with a regime memory.
Slow layer: correction per cell of (effective residual load, hour), kernel-smoothed over both. What a cell knows is
replaced as new observations of that cell arrive (memory N1 kernel-weighted observations), not as the calendar moves;
only a slow calendar decay (half-life CAL1 days) on top. A regime that comes in clusters weeks apart, like windy winter
nights, is therefore still known when it returns. N1, the shrinkage LAM1 and the hour width are fixed for all forecasts,
set from the persistence of the curve's errors across visits (persist_check.py): about half to two thirds of the earlier
error is applied once a cell has been visited a few times.
Fast layer: unchanged from dyn2 (kernel over x and hour, exponential forgetting in days, tuned per forecast).
usage: dyn6.py PERIOD RUN [ARMS] [unit]"""
import sys, os, numpy as np
import live_common as lc
SP = lc.SP; period, run = sys.argv[1], sys.argv[2]; ARMS = (sys.argv[3] if len(sys.argv) > 3 else "known,ts").split(","); UNIT = len(sys.argv) > 4 and sys.argv[4] == "unit"
N1 = float(os.environ.get("N1", 15)); LAM1 = float(os.environ.get("LAM1", 10)); CAL1 = float(os.environ.get("CAL1", 90)); BWH1 = float(os.environ.get("BWH1", 3.0))
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]; a = I["actual"].astype(float)
F = lc.load(); hour_all = F["DE"].index.tz_convert("Europe/Berlin").hour.to_numpy()
C = -7.5 + 5.0 * np.arange(17); BW = 4.0; VAL = 120; HH = np.arange(24)
def kern(x): return np.exp(-0.5 * ((x[..., None] - C) / BW) ** 2)
def kh(bw):
    d = np.abs(HH[:, None] - HH[None, :]); d = np.minimum(d, 24 - d); return np.exp(-0.5 * (d / bw) ** 2)
KH1 = kh(BWH1); ACAL = 0.5 ** (1 / CAL1)
def walk(R, X, tau2, lam2, bwh):
    n = len(R); a2 = 0.5 ** (1 / tau2); KH = kh(bwh); nb = len(C)
    Sn = np.zeros((nb, 24)); Sd = np.zeros((nb, 24)); En = np.zeros((nb, 24)); Ed = np.zeros((nb, 24)); pred = np.zeros_like(R)
    for v in range(n):
        S, E = Sn / (Sd + LAM1), En / (Ed + lam2)
        ok = np.isfinite(R[v]) & np.isfinite(X[v]); K = kern(np.where(ok, X[v], 0.0)); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9); Kn = np.where(ok[:, None], Kn, 0.0)
        p1 = np.einsum("hb,bh->h", Kn, S); pred[v] = p1 + np.einsum("hb,bh->h", Kn, E)
        r = np.where(ok, R[v], 0.0); e1 = r - p1; K0 = K * ok[:, None]
        add = np.einsum("hb,hg->bg", K0, KH1); dec = ACAL * np.exp(-add / N1)
        Sn = dec * Sn + np.einsum("hb,hg->bg", K0 * r[:, None], KH1); Sd = dec * Sd + add
        En = a2 * En + np.einsum("hb,hg->bg", K0 * e1[:, None], KH); Ed = a2 * Ed + np.einsum("hb,hg->bg", K0, KH)
    return pred, (Sn / (Sd + LAM1), En / (Ed + lam2), Sd)
P = {arm: [] for arm in ARMS}; chosen = []
for j, t in enumerate(tags):
    if UNIT and t != "2026-02-02": continue
    z = np.load(f"{SP}/out/struct_{run}/{t}.npz"); R = z["R1"].astype(float) / 10; X = z["XH1"].astype(float); lo, hi = np.nanpercentile(R, [1, 99]); R = np.clip(R, lo, hi); n = len(R); sl = slice(max(20, n - VAL), n)
    best = (np.inf, None)
    for g in [(t2, l2, bw) for t2 in (1.5, 3, 6, 12) for l2 in (1, 2, 8) for bw in (1.0, 1.5, 3.0)]:
        s_ = np.nanmean(np.abs(R[sl] - walk(R, X, *g)[0][sl])); best = min(best, (s_, g), key=lambda q: q[0])
    g = best[1]; chosen.append(list(g)); _, (S, E, Sd) = walk(R, X, *g)
    if UNIT:
        np.set_printoptions(precision=1, suppress=True, linewidth=200); kk = kern(np.array([4.0, 6.0, 8.0, 12.0, 20.0, 30.0])); kk = kk / kk.sum(1, keepdims=True)
        print("slow layer before the 3 February night, correction [ct] at x = 4, 6, 8, 12, 20, 30 GW (rows) and hours 1..5 (columns):"); print((kk @ S)[:, 1:6]); print("retained weight in those cells:"); print((kk @ Sd)[:, 1:6]); sys.exit()
    d0 = int(cuts[j]); hh = hour_all[d0:d0 + lc.HZ]
    for arm in ARMS:
        x = z[f"x_{arm}"].astype(float); K = kern(x); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9)
        P[arm].append(z[f"{arm}_shape"].astype(float) + np.einsum("qb,bq->q", Kn, S[:, hh]) + np.einsum("qb,bq->q", Kn, E[:, hh]))
ch = np.array(chosen, float); print(f"{period} {run} [regime memory {N1:.0f} obs, shrinkage {LAM1:.0f}, calendar half-life {CAL1:.0f} d]: fast layer median memory {np.median(ch[:, 0]):.1f} d, shrinkage {np.median(ch[:, 1]):.0f}, hour width {np.median(ch[:, 2]):.1f} h")
def sc(p): e = np.abs(p - a); return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():7.2f}{e[:, 56:96].mean():7.2f}"
for arm in ARMS:
    p = np.stack(P[arm]); old = np.load(f"{SP}/out/structfc_{run}_{arm}_dyn2xxh.npz")
    print(f"  {arm:<6} night 34-48h 48-58h | curve only {sc(old['curve'].astype(float))} | online correction as before {sc(old['vint'].astype(float))} | with regime memory {sc(p)}")
    np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_dyn6.npz", vint=p.astype(np.float32), curve=old["curve"], cutoffs=I["cutoffs"])
