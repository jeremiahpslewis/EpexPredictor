"""Ledger of the structural core (notes 21.126): price error as structure is added to the shift, and how thin the layer left to Timestral is."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc, core_fit as cf
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; Zs = np.load(f"{SP}/out/shift15_lite.npz"); X = Zs["rl"] - Zs["other"]; U = X - Zs["Sp"]; DL = np.load(f"{SP}/out/day_look.npz")
F = lc.load(); D = F["DE"]; WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); assert len(WE) == len(U); ND = 85; res = {}
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2"), ("winterf", "mid"), ("summerf", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); n = len(cuts)
    S = {k: np.zeros((n, 96)) for k in ("C0", "C1", "C2", "C3", "C4", "T")}; thin = {k: [] for k in ("C1", "C2", "C3")}; co = []
    for j, c in enumerate(cuts):
        u, x, we = U[c - ND * 96:c], X[c - ND * 96:c], WE[c - ND * 96:c]; xf = T["xh"][j].astype(float)[:96]; wf = WE[c:c + 96]
        S["C0"][j] = np.nanmean(U[c - 7 * 96:c]); S["T"][j] = T["median"][j].astype(float)[:96]
        for lev in (1, 2, 3):
            m = cf.fit(u, x, we, lev); S[f"C{lev}"][j] = cf.apply(m, xf, wf); r = u - cf.apply(m, x, we); thin[f"C{lev}"].append(np.nanvar(r) / np.nanvar(u))
        co.append(m["co"][48:]); rl = np.nanmean(r.reshape(ND, 96), 1); ok = np.isfinite(rl[1:]) & np.isfinite(rl[:-1]); rho = float((rl[1:][ok] * rl[:-1][ok]).sum() / (rl[:-1][ok] ** 2).sum()); S["C4"][j] = S["C3"][j] + rho * rl[-1]
    g = I["gas"].astype(float); price = lambda eps: np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    Pz = {k: price(v) for k, v in S.items()}; e = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1).mean(); ut = np.stack([U[c:c + 96] for c in cuts]); okd = ~np.isnan(ut).any(1)
    sh = lambda v: np.sqrt(((v - ut)[okd] ** 2).mean()); gap = lambda k, s0=0, s1=96: 100 * (e(Pz["C0"], s0, s1) - e(Pz[k], s0, s1)) / (e(Pz["C0"], s0, s1) - e(Pz["T"], s0, s1))
    res[per] = Pz; np.savez_compressed(f"{SP}/out/core_ledger_{per}.npz", a=a, **{f"p_{k}": v for k, v in Pz.items()}, **{f"s_{k}": v for k, v in S.items()})
    print(f"{per:8s} {n:3d} days | variance left to Timestral in the context: after C1 {100 * np.mean(thin['C1']):.0f}%  C2 {100 * np.mean(thin['C2']):.0f}%  C3 {100 * np.mean(thin['C3']):.0f}% | coefficients (median) within hour {np.median([c_[0] for c_ in co]):+.2f}, daily swing {np.median([c_[1] for c_ in co]):+.2f}, level {np.median([c_[2] for c_ in co]):+.2f}")
    print("         shift RMSE GW:   " + "  ".join(f"{k} {sh(S[k]):.2f}" for k in S))
    print("         price MAE whole day ct: " + "  ".join(f"{k} {e(Pz[k]):.3f}" for k in Pz) + f" | gap C0 -> Timestral closed by the core: C1 {gap('C1'):.0f}%  C2 {gap('C2'):.0f}%  C3 {gap('C3'):.0f}%  C4 {gap('C4'):.0f}%")
    print("         price MAE night ct:     " + "  ".join(f"{k} {e(Pz[k], 0, 24):.3f}" for k in Pz) + f" | closed: C3 {gap('C3', 0, 24):.0f}%  C4 {gap('C4', 0, 24):.0f}%")
