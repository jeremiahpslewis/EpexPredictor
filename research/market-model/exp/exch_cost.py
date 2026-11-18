"""What the exchange / neighbour part costs: German price error with the full equilibrium against the German balance cleared alone with the measured exchanges."""
import sys, numpy as np
import star as S
run, period, tag = sys.argv[1], sys.argv[2], sys.argv[3]; SP = S.lc.SP
z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")}); pe = z["peq"].astype(float); eeq = z["eeq"].astype(float)
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96; lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4
H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1); P = H["P"]; n = len(P)
cap, mc, wd, sl, shift, ct = S.build(q, d, np); fl = S.floors(q, d, np).sum(1) * np.ones(n); hde = q["hz"][0] * d["hdd"][:, 0] if "hz" in q else 0.0
xok = d["xok"].all(1) & d["ok"][:, 0]; xs = np.where(d["xok"], d["x"], 0.0).sum(1)
def clear(exports, ps_meas=None):
    pbar = None; PB = None
    for it in range(4):
        lo_ = np.full(n, S.PLO); hi_ = np.full(n, S.PHI)
        for _ in range(S.NBIS):
            mid = 0.5 * (lo_ + hi_); ps = (0.0 if pbar is None else q["p_ps"] * np.tanh((mid - pbar) / q["k_ps"])) if ps_meas is None else ps_meas
            f = S.supply(mid[:, None], cap[:, :1], mc[:, :1], wd[:1], sl[:1], np, pbar=None if PB is None else PB[:, None], lam=q["lam"][:1])[:, 0] + ps - (d["rl"][:, 0] + shift + exports - fl - hde + q["capf"][0, 5]); lo_ = np.where(f < 0, mid, lo_); hi_ = np.where(f < 0, hi_, mid)
        p0 = 0.5 * (lo_ + hi_); dm = S.day_mean(p0, d["day"]); pbar = dm if pbar is None else 0.5 * (pbar + dm); PB = pbar
    return p0
pa = clear(xs); pb = clear(xs, ps_meas=np.where(d["psok"], d["ps"], 0.0)); k = xok; kb = xok & d["psok"]
mae = lambda p, kk: np.mean(np.abs(P[kk, 0] - p[kk])) / 10
print(f"{run} {tag}: hours with all eight scheduled exchanges {k.sum()} of {n}")
print(f"  German price error on those hours, ct: full equilibrium (model exchanges) {mae(pe[:, 0], k):.2f} | German balance alone with measured exchanges {mae(pa, k):.2f} | also with measured pumped storage {mae(pb, kb):.2f} (full equilibrium on the same hours {mae(pe[:, 0], kb):.2f})")
ex_err = eeq.sum(1) - xs; print(f"  model net exports minus scheduled on those hours: mean {ex_err[k].mean():+.2f} GW, mean abs {np.abs(ex_err[k]).mean():.2f} GW; per border mean abs GW: " + "  ".join(f"{zn} {np.mean(np.abs(eeq[k, i] - d['x'][k, i])):.2f}" for i, zn in enumerate(S.ZN[1:])))
hr = H["hour"]; night = (hr < 8)
print(f"  night hours (0-8h): full equilibrium {mae(pe[:, 0], k & night):.2f} | measured exchanges {mae(pa, k & night):.2f}")
