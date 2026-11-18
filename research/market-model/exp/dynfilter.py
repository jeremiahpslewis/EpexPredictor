"""Online correction of the structural model from its own recent errors (all of them known at issue):
   error(x, hour) ~ L + delta(x) + prof[hour]
   L        level, exponentially weighted mean of the daily mean errors
   delta(x) correction of the curve as a function of the effective residual load x (kernel over x, exponential forgetting,
            shrunk to zero where there is little recent data) - the lower and upper ends of the curve drift, the middle hardly
   prof     hour-of-day profile of what is left
Each part is updated day by day on the error left by the parts before it, as predicted beforehand (so nothing is fitted
twice). Memory lengths are chosen per forecast on the error history itself (walk forward), never on the scored days.
usage: dynfilter.py PERIOD RUN [ARMS]"""
import sys, os, numpy as np
import live_common as lc
SP = lc.SP; period, run = sys.argv[1], sys.argv[2]; ARMS = (sys.argv[3] if len(sys.argv) > 3 else "known,ts").split(",")
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]; a = I["actual"].astype(float)
F = lc.load(); hour_all = F["DE"].index.tz_convert("Europe/Berlin").hour.to_numpy()
Z = [np.load(f"{SP}/out/struct_{run}/{t}.npz") for t in tags]
C = -7.5 + 5.0 * np.arange(17); BW = 4.0; LAG = 1; VAL = 120
def kern(x): return np.exp(-0.5 * ((x[..., None] - C) / BW) ** 2)                      # (..., bins)
def walk(R, X, tauL, tauD, lamD, tauH, lamH, use=(1, 1, 1)):
    """one pass over the days; returns one-step(LAG)-ahead predictions for every day and the final state"""
    n = len(R); aL = 0.5 ** (1 / tauL); aD = 0.5 ** (1 / tauD); aH = 0.5 ** (1 / tauH)
    Ln = Ld = 0.0; Dn = np.zeros(len(C)); Dd = np.zeros(len(C)); Hn = np.zeros(24); Hd = np.zeros(24)
    states = []; pred = np.zeros_like(R)
    for v in range(n):
        states.append((Ln / (Ld + 0.5) if Ld > 0 else 0.0, Dn / (Dd + lamD), Hn / (Hd + lamH)))
        L, D, Hp = states[v - LAG + 1] if v - LAG + 1 >= 0 else (0.0, np.zeros(len(C)), np.zeros(24))  # state after day v-LAG
        K = kern(X[v]); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9)
        pL = L * use[0]; pD = (Kn @ D) * use[1]; pH = Hp * use[2]; pred[v] = pL + pD + pH
        ok = np.isfinite(R[v]) & np.isfinite(X[v])
        if ok.any():
            r = np.where(ok, R[v], 0.0); e1 = r - pL; e2 = e1 - pD
            Ln = aL * Ln + r[ok].mean(); Ld = aL * Ld + 1.0
            Dn = aD * Dn + (K * (e1 * ok)[:, None]).sum(0); Dd = aD * Dd + (K * ok[:, None]).sum(0)
            Hn = aH * Hn + e2 * ok; Hd = aH * Hd + ok
        else: Ln *= aL; Ld *= aL; Dn *= aD; Dd *= aD; Hn *= aH; Hd *= aH
    final = (Ln / (Ld + 0.5) if Ld > 0 else 0.0, Dn / (Dd + lamD), Hn / (Hd + lamH))
    return pred, final
out = {arm: {k: [] for k in ("L", "LD", "LDH")} for arm in ARMS}; chosen = []
for j, z in enumerate(Z):
    R = z["R1"].astype(float) / 10; X = z["XH1"].astype(float); lo, hi = np.nanpercentile(R, [1, 99]); R = np.clip(R, lo, hi); n = len(R); sl = slice(max(20, n - VAL), n)
    sc = lambda pred: np.nanmean(np.abs(R[sl] - pred[sl]))
    base = np.nanmean(np.abs(R[sl])); best = (base, None)
    for tL in (1.5, 3, 6, 12):
        s_ = sc(walk(R, X, tL, 10, 20, 10, 2, (1, 0, 0))[0]); best = min(best, (s_, tL), key=lambda t: t[0])
    tL = best[1]; uL = tL is not None and best[0] < 0.98 * base; tL = tL or 3; b2 = (best[0] if uL else base, None)
    for tD in (5, 10, 20, 40, 80):
        for lD in (10, 40):
            s_ = sc(walk(R, X, tL, tD, lD, 10, 2, (uL, 1, 0))[0]); b2 = min(b2, (s_, (tD, lD)), key=lambda t: t[0])
    uD = b2[1] is not None and b2[0] < 0.99 * (best[0] if uL else base); tD, lD = b2[1] or (20, 40); b3 = (b2[0] if uD else (best[0] if uL else base), None)
    for tH in (5, 10, 20, 40):
        s_ = sc(walk(R, X, tL, tD, lD, tH, 2, (uL, uD, 1))[0]); b3 = min(b3, (s_, tH), key=lambda t: t[0])
    uH = b3[1] is not None and b3[0] < 0.99 * (b2[0] if uD else (best[0] if uL else base)); tH = b3[1] or 10
    _, (L, D, Hp) = walk(R, X, tL, tD, lD, tH, 2, (uL, uD, uH)); chosen.append([tL, uL, tD, lD, uD, tH, uH, base, b3[0]])
    d0 = int(cuts[j]); hh = hour_all[d0:d0 + lc.HZ]
    for arm in ARMS:
        x = z[f"x_{arm}"].astype(float); K = kern(x); Kn = K / np.maximum(K.sum(1, keepdims=True), 1e-9); base_fc = z[f"{arm}_shape"].astype(float)
        out[arm]["L"].append(base_fc + L * uL); out[arm]["LD"].append(base_fc + L * uL + (Kn @ D) * uD); out[arm]["LDH"].append(base_fc + L * uL + (Kn @ D) * uD + Hp[hh] * uH)
ch = np.array(chosen, float); print(f"{period} {run}: level used {100*ch[:,1].mean():.0f}% (memory {np.median(ch[:,0]):.1f} d), curve correction used {100*ch[:,4].mean():.0f}% (memory {np.median(ch[:,2]):.0f} d), hour profile used {100*ch[:,6].mean():.0f}% (memory {np.median(ch[:,5]):.0f} d); history error {ch[:,7].mean():.2f} -> {ch[:,8].mean():.2f} ct")
def sc(p, m=None):
    e = np.abs(p - a); return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():7.2f}{e[:, 56:96].mean():7.2f}"
for arm in ARMS:
    base = np.stack([z[f"{arm}_shape"] for z in Z]).astype(float); P = {k: np.stack(v) for k, v in out[arm].items()}
    old = np.load(f"{SP}/out/structfc_{run}_{arm}_R1.npz")["vint"].astype(float) if os.path.exists(f"{SP}/out/structfc_{run}_{arm}_R1.npz") else None
    print(f"  {arm:<6} night 34-48h 48-58h | curve only {sc(base)} | + level {sc(P['L'])} | + curve correction {sc(P['LD'])} | + hour profile {sc(P['LDH'])}" + (f" | old filter {sc(old)}" if old is not None else ""))
    np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_dyn.npz", vint=P["LDH"].astype(np.float32), ld=P["LD"].astype(np.float32), curve=base.astype(np.float32), cutoffs=I["cutoffs"])
    if arm == "known":
        rl = I["rl_true"].astype(float)[:, :96]; A = a[:, :96]
        print("         by residual load, first day, MAE (bias): " + "  ".join(f"{lab}: curve {np.abs(base[:, :96] - A)[m].mean():.2f} ({(base[:, :96] - A)[m].mean():+.2f}) -> {np.abs(P['LDH'][:, :96] - A)[m].mean():.2f} ({(P['LDH'][:, :96] - A)[m].mean():+.2f})" for lab, m in (("<25 GW", rl < 25), ("25-45", (rl >= 25) & (rl < 45)), (">=45", rl >= 45))))
