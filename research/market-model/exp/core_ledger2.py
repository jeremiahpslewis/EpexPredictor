"""Ledger of the state core (notes 21.127)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc, core_state as cs
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; Zs = np.load(f"{SP}/out/shift15_lite.npz"); X = Zs["rl"] - Zs["other"]; U = X - Zs["Sp"]; DL = np.load(f"{SP}/out/day_look.npz")
F = lc.load(); D = F["DE"]; WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); ND, WARM = 85, 40
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2"), ("winterf", "mid"), ("summerf", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); n = len(cuts)
    S = {k: np.zeros((n, 96)) for k in ("C0", "K0", "K1", "K2", "T")}; thin, lam, rho = [], [], []
    for j, c in enumerate(cuts):
        nd = ND + WARM; Ud = np.r_[U[c - nd * 96:c], np.full(96, np.nan)].reshape(nd + 1, 96); Xd = X[c - nd * 96:c + 96].reshape(nd + 1, 96); WEd = WE[c - nd * 96:c + 96].reshape(nd + 1, 96)[:, 48]
        R = cs.run(Ud, Xd, WEd, WARM, T["xh"][j].astype(float)[:96]); S["C0"][j] = np.nanmean(U[c - 7 * 96:c]); S["T"][j] = T["median"][j].astype(float)[:96]
        for k in ("K0", "K1", "K2"): S[k][j] = R[k][-1]
        inn = Ud[WARM:-1] - R["K2"][:-1]; thin.append(np.nanvar(inn) / np.nanvar(Ud[WARM:-1])); lam.append(R["lam"]); rho.append(R["rho"])
    g = I["gas"].astype(float); price = lambda eps: np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * g[j] + 0.37 * 75.0), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    Pz = {k: price(v) for k, v in S.items()}; e = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1).mean(); ut = np.stack([U[c:c + 96] for c in cuts]); okd = ~np.isnan(ut).any(1); sh = lambda v: np.sqrt(((v - ut)[okd] ** 2).mean())
    gap = lambda k, s0=0, s1=96: 100 * (e(Pz["C0"], s0, s1) - e(Pz[k], s0, s1)) / (e(Pz["C0"], s0, s1) - e(Pz["T"], s0, s1))
    np.savez_compressed(f"{SP}/out/core_state_{per}.npz", a=a, **{f"p_{k}": v for k, v in Pz.items()}, **{f"s_{k}": v for k, v in S.items()})
    print(f"{per:8s} {n:3d} days | day-ahead innovations of K2: {100 * np.mean(thin):.0f}% of the shift's variance ({100 * np.sqrt(np.mean(thin)):.0f}% in GW) | lambda {np.median(lam):+.2f}  rho {np.median(rho):.2f}")
    print("         shift RMSE GW:   " + "  ".join(f"{k} {sh(S[k]):.2f}" for k in S))
    print("         price MAE whole day ct: " + "  ".join(f"{k} {e(Pz[k]):.3f}" for k in Pz) + f" | gap C0 -> Timestral closed: K0 {gap('K0'):.0f}%  K1 {gap('K1'):.0f}%  K2 {gap('K2'):.0f}%")
    print("         price MAE night ct:     " + "  ".join(f"{k} {e(Pz[k], 0, 24):.3f}" for k in Pz) + f" | closed: K0 {gap('K0', 0, 24):.0f}%  K1 {gap('K1', 0, 24):.0f}%  K2 {gap('K2', 0, 24):.0f}%")
