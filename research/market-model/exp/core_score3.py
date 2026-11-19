"""J1a (core without the day-type switch) against the whole-shift route (notes 21.130)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); Zs = np.load(f"{SP}/out/shift15_lite.npz"); U = Zs["rl"] - Zs["other"] - Zs["Sp"]
F = lc.load(); D = F["DE"]; WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5))
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, blk in (("winter", 3), ("summer", 3), ("autumn", 7), ("winterf", 3), ("summerf", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); J = np.load(f"{SP}/out/core_joins_{per}.npz"); C = np.load(f"{SP}/out/tscore_k1a_{per}.npz"); a, P = J["a"], J["P"]; n = len(a); g = I["gas"].astype(float); cuts = I["cuts"].astype(int)
    price = lambda eps: np.stack([np.asarray(M.clear(th, C["xh"][j].astype(float), np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    pa = price(C["k1"].astype(float) + C["ts"]); pc = price(C["k1"].astype(float)); c0 = price(np.repeat(np.array([np.nanmean(U[c - 7 * 96:c]) for c in cuts])[:, None], 96, 1)); E = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1); w = J["p_whole_shift"]; sw = np.array([WE[c + 48] != WE[c - 48] for c in cuts]); out = []
    for seg, s0, s1 in (("whole day", 0, 96), ("night", 0, 24)):
        d = E(pa, s0, s1) - E(w, s0, s1); b = boot(d, blk); d2 = E(0.5 * (pa + P), s0, s1) - E(0.5 * (w + P), s0, s1); b2 = boot(d2, blk)
        out.append(f"{seg}: core alone {E(pc, s0, s1).mean():.3f} (closes {100 * (E(c0, s0, s1).mean() - E(pc, s0, s1).mean()) / (E(c0, s0, s1).mean() - E(w, s0, s1).mean()):.0f}%)  J1a {E(pa, s0, s1).mean():.3f}  J1 {E(J['p_J1'], s0, s1).mean():.3f}  whole shift {E(w, s0, s1).mean():.3f} | J1a against whole shift {100 * d.mean() / E(w, s0, s1).mean():+.1f}% ({b[0]:+.3f} .. {b[1]:+.3f}), with price route {100 * d2.mean() / E(0.5 * (w + P), s0, s1).mean():+.1f}% ({b2[0]:+.3f} .. {b2[1]:+.3f})")
    dn = E(pa, 0, 24) - E(w, 0, 24); print(f"{per:8s} left to Timestral {100 * C['thin1'].mean():.0f}% of the variance, sd of its part on the target day {C['ts'].std():.2f} GW\n     " + "\n     ".join(out) + f"\n     night shortfall where the day type changes {dn[sw].mean():+.3f}, elsewhere {dn[~sw].mean():+.3f}")
    np.savez_compressed(f"{SP}/out/core_j1a_{per}.npz", a=a, P=P, p_J1a=pa, p_core=pc, p_C0=c0, p_whole=w)
