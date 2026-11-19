"""The three joins of core and Timestral against Timestral on the whole shift (notes 21.129)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; DL = np.load(f"{SP}/out/day_look.npz"); rng = np.random.default_rng(0)
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tag, blk in (("winter", "mid", 3), ("summer", "mid", 3), ("autumn", "v2", 7), ("winterf", "mid", 3), ("summerf", "mid", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); C = np.load(f"{SP}/out/tscore_{per}.npz"); C1 = np.load(f"{SP}/out/tscore_k1_{per}.npz"); C2 = np.load(f"{SP}/out/tscore_cov_{per}.npz")
    a = DL[f"{per}_a"].astype(float); P = DL[f"{per}_P"].astype(float); n = len(a); g = I["gas"].astype(float); core = C["core"].astype(float)
    S = {"core alone": core, "J0 innovations": core + C["ts"], "J1 less K1": C1["k1"].astype(float) + C1["ts"], "J2 core as input": C2["ts"].astype(float), "whole shift": T["median"].astype(float)[:, :96]}
    price = lambda eps: np.stack([np.asarray(M.clear(th, C["xh"][j].astype(float), np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)]); Pz = {k: price(v) for k, v in S.items()}; E = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1)
    print(f"{per:8s} {n:3d} days | distance from the core (sd, GW): " + "  ".join(f"{k} {(S[k] - core).std():.2f}" for k in list(S)[1:]))
    for seg, s0, s1 in (("whole day", 0, 96), ("night", 0, 24)):
        t = E(Pz["whole shift"], s0, s1); row = "  ".join(f"{k} {E(v, s0, s1).mean():.3f}" for k, v in Pz.items()); cmp_ = []
        for k in ("J0 innovations", "J1 less K1", "J2 core as input"):
            d = E(Pz[k], s0, s1) - t; b = boot(d, blk); d2 = E(0.5 * (Pz[k] + P), s0, s1) - E(0.5 * (Pz["whole shift"] + P), s0, s1); b2 = boot(d2, blk); cmp_.append(f"{k.split()[0]} {100 * d.mean() / t.mean():+.1f}% ({b[0]:+.3f} .. {b[1]:+.3f}), with price route {100 * d2.mean() / E(0.5 * (Pz['whole shift'] + P), s0, s1).mean():+.1f}% ({b2[0]:+.3f} .. {b2[1]:+.3f})")
        print(f"         {seg:9s} MAE ct: {row}\n              against the whole-shift route: " + " | ".join(cmp_))
    np.savez_compressed(f"{SP}/out/core_joins_{per}.npz", a=a, P=P, **{"p_" + k.split()[0].replace("core", "core_alone").replace("whole", "whole_shift"): v for k, v in Pz.items()})
