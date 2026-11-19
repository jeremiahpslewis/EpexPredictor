"""Sponge test, scoring (notes 21.128): the state core alone, the core with Timestral on its innovations, and Timestral on the whole shift."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; Zs = np.load(f"{SP}/out/shift15_lite.npz"); U = Zs["rl"] - Zs["other"] - Zs["Sp"]; DL = np.load(f"{SP}/out/day_look.npz"); rng = np.random.default_rng(0)
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tag, blk in (("winter", "mid", 3), ("summer", "mid", 3), ("autumn", "v2", 7), ("winterf", "mid", 3), ("summerf", "mid", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); C = np.load(f"{SP}/out/tscore_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); P = DL[f"{per}_P"].astype(float); n = len(cuts); g = I["gas"].astype(float)
    assert np.nanmax(np.abs(C["xh"] - T["xh"][:, :96])) < 0.5, float(np.nanmax(np.abs(C["xh"] - T["xh"][:, :96])))
    S = {"C0": np.repeat(np.array([np.nanmean(U[c - 7 * 96:c]) for c in cuts])[:, None], 96, 1), "K1": C["k1"].astype(float), "K3 (core alone)": C["core"].astype(float), "core + Timestral": (C["core"] + C["ts"]).astype(float), "Timestral, whole shift": T["median"].astype(float)[:, :96]}
    price = lambda eps: np.stack([np.asarray(M.clear(th, C["xh"][j].astype(float), np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    Pz = {k: price(v) for k, v in S.items()}; E = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1); ut = np.stack([U[c:c + 96] for c in cuts]); okd = ~np.isnan(ut).any(1); sh = lambda v: np.sqrt(((v - ut)[okd] ** 2).mean())
    print(f"{per:8s} {n:3d} days | left to Timestral: innovations of K1 {100 * C['thin1'].mean():.0f}%, of K3 {100 * C['thin'].mean():.0f}% of the shift's variance | sd of what Timestral adds on the target day {C['ts'].std():.2f} GW against {S['Timestral, whole shift'].std():.2f} for its whole-shift forecast")
    print("         shift RMSE GW: " + "  ".join(f"{k} {sh(v):.2f}" for k, v in S.items()))
    for seg, s0, s1 in (("whole day", 0, 96), ("night", 0, 24)):
        t = E(Pz["Timestral, whole shift"], s0, s1); c0 = E(Pz["C0"], s0, s1).mean(); d = E(Pz["core + Timestral"], s0, s1) - t; b = boot(d, blk); m = 0.5 * (Pz["core + Timestral"] + P); m0 = 0.5 * (Pz["Timestral, whole shift"] + P); d2 = E(m, s0, s1) - E(m0, s0, s1); b2 = boot(d2, blk)
        print(f"         price MAE {seg:9s} ct: " + "  ".join(f"{k} {E(v, s0, s1).mean():.3f}" for k, v in Pz.items()) + f" | core alone closes {100 * (c0 - E(Pz['K3 (core alone)'], s0, s1).mean()) / (c0 - t.mean()):.0f}% of C0 -> Timestral | core + Timestral against Timestral on the whole shift {100 * d.mean() / t.mean():+.1f}% ({b[0]:+.3f} .. {b[1]:+.3f}) | averaged with the price route: {100 * d2.mean() / E(m0, s0, s1).mean():+.1f}% ({b2[0]:+.3f} .. {b2[1]:+.3f})")
    np.savez_compressed(f"{SP}/out/core_score_{per}.npz", a=a, P=P, **{("p_" + k.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("+", "plus")): v for k, v in Pz.items()})
