"""The application's core forecast (notes 21.141): frozen curves + first-year hour-of-day shape + level of the 672 quarter hours before."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; F = lc.load(); D = F["DE"]; idx = D.index; loc = idx.tz_convert("Europe/Berlin")
WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); key = loc.hour.to_numpy() + 24 * WE.astype(int)
Z = np.load(f"{SP}/out/shift15_lite.npz"); U = Z["rl"] - Z["other"] - Z["Sp"]; first = np.asarray(idx < pd.Timestamp("2025-03-01", tz="UTC"))
day = np.asarray((loc.normalize() - loc.normalize()[0]).days); dev = U - pd.Series(U).groupby(day).transform("mean").to_numpy()
shape = pd.Series(dev[first]).groupby(key[first]).mean().reindex(range(48)).to_numpy(); shape = shape - 0.0
print("SHAPE working days (local hour 0..23):", np.round(shape[:24], 3).tolist()); print("SHAPE weekends and holidays:", np.round(shape[24:], 3).tolist())
np.save(f"{SP}/out/core_shape.npy", shape); rng = np.random.default_rng(0); DL = np.load(f"{SP}/out/day_look.npz")
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tag, blk in (("winter", "mid", 3), ("summer", "mid", 3), ("autumn", "v2", 7), ("winterf", "mid", 3), ("summerf", "mid", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); P = DL[f"{per}_P"].astype(float); n = len(cuts); g = I["gas"].astype(float)
    lvl = np.array([np.nanmean(U[c - 672:c]) for c in cuts]); core = np.stack([lvl[j] + shape[key[c:c + 96]] for j, c in enumerate(cuts)]); c0 = np.repeat(lvl[:, None], 96, 1)
    price = lambda eps: np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    pc, p0, pr = price(core), price(c0), price(T["median"].astype(float)[:, :96]); E = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1)
    out = []
    for seg, s0, s1 in (("whole day", 0, 96), ("night", 0, 24)):
        cl = 100 * (E(p0, s0, s1).mean() - E(pc, s0, s1).mean()) / (E(p0, s0, s1).mean() - E(pr, s0, s1).mean()); d = E(pc, s0, s1) - E(pr, s0, s1); b = boot(d, blk)
        out.append(f"{seg}: C0 {E(p0, s0, s1).mean():.3f}  core {E(pc, s0, s1).mean():.3f}  route {E(pr, s0, s1).mean():.3f}  P {E(P, s0, s1).mean():.3f} | core closes {cl:.0f}% | core - route {100 * d.mean() / E(pr, s0, s1).mean():+.0f}% ({b[0]:+.3f} .. {b[1]:+.3f})")
    print(f"{per:8s} {n:3d} days | " + " | ".join(out))
