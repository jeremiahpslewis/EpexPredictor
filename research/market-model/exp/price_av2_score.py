"""Price effect of the averaged load forecast against the Timestral load forecast alone, winter and summer (notes 21.119)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/day_look.npz")
def boot(d, blk=3):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tf in (("winter", "win_trees"), ("summer", "ssn_trees")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = Z[f"{per}_a"].astype(float); n = len(a); tr = np.load(f"{SP}/out/{tf}.npz")["vint"].astype(float)[:, :96]; ok = ~np.isnan(tr).any(1)
    def slim(tag):
        T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); return np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(I["gas"][j]) + 0.37 * 75.0), np.full(96, 75.0), eps=T["median"][j].astype(float)[:96])) / 10 for j in range(n)])
    A = {"AV": slim("avgA"), "TS": slim("mid")}; P = {"AV": np.load(f"{SP}/out/tsm_rlavgA_{per}.npz")["median"].astype(float)[:, :96], "TS": np.load(f"{SP}/out/tsm_rl_{per}.npz")["median"].astype(float)[:, :96]}
    e = lambda v, s0=0, s1=96: np.abs(v - a)[ok][:, s0:s1].mean(1)
    print(per, f"({int(ok.sum())} of {n} days)")
    for nm, D in (("slim route", A), ("price route", P), ("mix (trees + price + slim) / 3", {k: (tr + P[k] + A[k]) / 3 for k in A})):
        for seg, s0, s1 in (("night", 0, 24), ("whole day", 0, 96)):
            d = e(D["AV"], s0, s1) - e(D["TS"], s0, s1); b = boot(d); print(f"   {nm:32s} {seg:10s} MAE ct: AV {e(D['AV'], s0, s1).mean():.3f}  TS {e(D['TS'], s0, s1).mean():.3f} | AV - TS {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f}) = {100 * d.mean() / e(D['TS'], s0, s1).mean():+.1f}%")
