import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; F = lc.load(); D = F["DE"]; idx = D.index; loc = idx.tz_convert("Europe/Berlin")
WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); key = loc.hour.to_numpy() + 24 * WE.astype(int); doy = loc.dayofyear.to_numpy(); shape = np.load(f"{SP}/out/core_shape.npy")
Z = np.load(f"{SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]; first = np.asarray(idx < pd.Timestamp("2025-03-01", tz="UTC")); DL = np.load(f"{SP}/out/day_look.npz"); rng = np.random.default_rng(0)
day = np.asarray((loc.normalize() - loc.normalize()[0]).days); dev = U - pd.Series(U).groupby(day).transform("mean").to_numpy()
# seasonal shape table: for each day of year (1..366) and key (48): mean first-year deviation over days within +-45 days of year
fd = pd.DataFrame({"dev": dev[first], "key": key[first], "doy": doy[first]}).dropna(); TAB = np.zeros((367, 48))
for d in range(1, 367):
    dist = np.minimum(np.abs(fd.doy - d), 366 - np.abs(fd.doy - d)); sel = fd[dist <= 45]; TAB[d] = sel.groupby("key").dev.mean().reindex(range(48)).fillna(0.0).to_numpy()
np.save(f"{SP}/out/core_shape_seasonal.npy", TAB)
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tag, blk in (("winter", "mid", 3), ("summer", "mid", 3), ("autumn", "v2", 7), ("winterf", "mid", 3), ("summerf", "mid", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); n = len(cuts); g = I["gas"].astype(float)
    lvl = np.array([np.nanmean(U[c - 672:c]) for c in cuts]); ut = np.stack([U[c:c + 96] for c in cuts]); dv = ut - np.nanmean(ut, 1, keepdims=True)
    s1 = np.stack([shape[key[c:c + 96]] for c in cuts]); s2 = np.stack([TAB[doy[c:c + 96], key[c:c + 96]] for c in cuts])
    price = lambda eps: np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    p0, p1, p2, pr = price(np.repeat(lvl[:, None], 96, 1)), price(lvl[:, None] + s1), price(lvl[:, None] + s2), price(T["median"].astype(float)[:, :96]); E = lambda v, s0=0, s1_=96: np.abs(v - a)[:, s0:s1_].mean(1)
    cc = lambda s: np.corrcoef(s.ravel(), np.nan_to_num(dv.ravel()))[0, 1]; out = []
    for seg, s0, s1_ in (("whole day", 0, 96), ("night", 0, 24)):
        d = E(p2, s0, s1_) - E(p0, s0, s1_); b = boot(d, blk); cl = 100 * (E(p0, s0, s1_).mean() - E(p2, s0, s1_).mean()) / (E(p0, s0, s1_).mean() - E(pr, s0, s1_).mean())
        out.append(f"{seg}: level {E(p0, s0, s1_).mean():.3f}  +year shape {E(p1, s0, s1_).mean():.3f}  +seasonal shape {E(p2, s0, s1_).mean():.3f}  route {E(pr, s0, s1_).mean():.3f} | seasonal vs level {100 * d.mean() / E(p0, s0, s1_).mean():+.1f}% ({b[0]:+.3f} .. {b[1]:+.3f}), closes {cl:.0f}% of level -> route")
    print(f"{per:8s} corr with the true deviation: year shape {cc(s1):.2f}, seasonal {cc(s2):.2f}\n     " + "\n     ".join(out))
