"""Level state for the structural model, estimated from its past errors (which end with the issue day D).
For each forecast: L = exponentially weighted mean of the daily mean errors (a steady-state local-level filter); its
smoothing and gain are chosen on the error history itself, by the error two days ahead. Optionally an hour-of-day
profile on top. usage: level_filter.py PERIOD RUN  -> writes keys into out/struct_RUN/level.npz and prints scores"""
import sys, os, glob, numpy as np
import live_common as lc
SP = lc.SP; period, run = sys.argv[1], sys.argv[2]; KEY = os.environ.get("RKEY", "R"); LAG = 1 if KEY == "R1" else 2; ARMS = os.environ.get("ARMS", "known,own,ownbc").split(",")
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; cuts = I["cuts"]; a = I["actual"].astype(float); gas = I["gas"].astype(float)
F = lc.load(); idx = F["DE"].index; local = idx.tz_convert("Europe/Berlin"); hour_all = local.hour.to_numpy()
Z = [np.load(f"{SP}/out/struct_{run}/{t}.npz") for t in tags]; assert len(Z) == len(tags)
LAMS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.93)
def ewma(dm, lam):
    L = np.zeros(len(dm)); l = 0.0; wsum = 0.0
    for i, v in enumerate(dm):
        if np.isfinite(v): l = lam * l + v; wsum = lam * wsum + 1.0
        L[i] = l / wsum if wsum > 0 else 0.0
    return L
out = {k: [] for k in ("lvl", "lvlh", "lvl_info")}
for j, z in enumerate(Z):
    R = z[KEY] / 10.0; lo, hi = np.nanpercentile(R, [2, 98]); Rc = np.clip(R, lo, hi); dm = np.nanmean(Rc, 1); n = len(dm); d0 = int(cuts[j]); hh = hour_all[d0:d0 + lc.HZ]
    best = None
    for lam in LAMS:
        L = ewma(dm, lam); x, y = L[:-LAG], dm[LAG:]; ok = np.isfinite(x) & np.isfinite(y); ok[:20] = False
        g = float(np.clip((x[ok] * y[ok]).sum() / max((x[ok] ** 2).sum(), 1e-9), 0, 1.2)); err = np.abs(y[ok] - g * x[ok]).mean()
        if best is None or err < best[0]: best = (err, lam, g, L[-1], np.abs(y[ok]).mean())
    err, lam, g, Llast, base = best; use = err < 0.98 * base
    lvl = g * Llast if use else 0.0
    # hour-of-day profile on top: exponentially weighted error of each hour relative to the level, with its own gain
    prof = np.zeros(24)
    if use:
        Lfull = ewma(dm, lam); dev = Rc - dm[:, None]; ph = np.stack([ewma(dev[:, h], 0.85) for h in range(24)], 1)
        for h in range(24):
            x, y = ph[:-LAG, h], dev[LAG:, h]; ok = np.isfinite(x) & np.isfinite(y); ok[:20] = False
            gh = float(np.clip((x[ok] * y[ok]).sum() / max((x[ok] ** 2).sum(), 1e-9), 0, 1.0))
            if np.abs(y[ok] - gh * x[ok]).mean() < 0.98 * np.abs(y[ok]).mean(): prof[h] = gh * ph[-1, h]
    out["lvl"].append(np.full(lc.HZ, lvl)); out["lvlh"].append(lvl + prof[hh]); out["lvl_info"].append([lam, g, Llast, float(use)])
R_ = {k: np.array(v) for k, v in out.items()}; np.savez_compressed(f"{SP}/out/struct_{run}/level_{KEY}.npz", **R_)
info = R_["lvl_info"]; print(f"{run}: smoothing chosen {np.round(np.bincount((info[:, 0] * 100).astype(int), minlength=94)[[0, 30, 50, 70, 85, 93]] / len(info), 2)} for {LAMS}; mean gain {info[:, 1].mean():.2f}; used in {100 * info[:, 3].mean():.0f}% of forecasts")
def sc(p, m=slice(None)):
    e = np.abs(p - a)[m]; d = (p - a)[m]; hi = a[m][:, :96] > 30
    return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}   bias {d[:, :96].mean():+.2f}   >30ct {np.abs(d[:, :96][hi]).mean():4.1f} /{d[:, :96][hi].mean():+5.1f}"
S = lambda k: np.stack([z[k] for z in Z]).astype(float)
print(f"{'':<40}{'night':>6}{'34-48h':>8}{'48-58h':>8}")
for arm in ARMS:
    if f"{arm}_shape" not in Z[0].files: continue
    base = S(f"{arm}_shape")
    print(f"{arm + ': curve only':<40}{sc(base)}"); print(f"{arm + ': + level':<40}{sc(base + R_['lvl'])}"); print(f"{arm + ': + level + hour profile':<40}{sc(base + R_['lvlh'])}"); np.savez_compressed(f"{SP}/out/structfc_{run}_{arm}_{KEY}.npz", vint=(base + R_["lvlh"]).astype(np.float32), curve=base.astype(np.float32), cutoffs=I["cutoffs"])
    if arm == "known":
        for lab, m in (("gas < 78", gas < 78), ("78-88", (gas >= 78) & (gas < 88)), (">= 88", gas >= 88)):
            print(f"   {lab:<10} curve only {np.abs(base - a)[m][:, :56].mean():.2f} ({(base - a)[m][:, :56].mean():+.2f})   + level {np.abs(base + R_['lvl'] - a)[m][:, :56].mean():.2f} ({(base + R_['lvl'] - a)[m][:, :56].mean():+.2f})   + profile {np.abs(base + R_['lvlh'] - a)[m][:, :56].mean():.2f} ({(base + R_['lvlh'] - a)[m][:, :56].mean():+.2f})")
