"""Price effect of the lead-weighted load forecast (LW) against the plain average (AV) and Timestral load alone (TS). Notes 21.120."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/day_look.npz")
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
for per, tf, tk, blk, ts_s, ts_p in (("winter", "win_trees", "vint", 3, "mid", "rl"), ("summer", "ssn_trees", "vint", 3, "mid", "rl"), ("autumn", "rl_E_pes_T", "vint", 7, "v2", "rlv2")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = Z[f"{per}_a"].astype(float); n = len(a); tr = np.load(f"{SP}/out/{tf}.npz")[tk].astype(float)[:, :96]; ok = ~np.isnan(tr).any(1)
    def slim(tag):
        T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); return np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(I["gas"][j]) + 0.37 * 75.0), np.full(96, 75.0), eps=T["median"][j].astype(float)[:96])) / 10 for j in range(n)])
    pr = lambda tag: np.load(f"{SP}/out/tsm_{tag}_{per}.npz")["median"].astype(float)[:, :96]
    A = {"LW": slim("lwA"), "AV": slim("avgA"), "TS": slim(ts_s)}; P = {"LW": pr("rllwA"), "AV": pr("rlavgA"), "TS": pr(ts_p)}
    if per == "autumn": A["T"] = slim("appO"); P["T"] = pr("rlappO")
    e = lambda v, s0=0, s1=96: np.abs(v - a)[ok][:, s0:s1].mean(1)
    print(per, f"({int(ok.sum())} days)")
    for nm, D in (("slim route", A), ("price route", P), ("mix (trees + price + slim) / 3", {k: (tr + P[k] + A[k]) / 3 for k in A})):
        for seg, s0, s1 in (("night", 0, 24), ("whole day", 0, 96)):
            row = "  ".join(f"{k} {e(D[k], s0, s1).mean():.3f}" for k in D); d = e(D["LW"], s0, s1) - e(D["AV"], s0, s1); b = boot(d, blk); d2 = e(D["LW"], s0, s1) - e(D["TS"], s0, s1); b2 = boot(d2, blk)
            extra = ""
            if "T" in D: d3 = e(D["LW"], s0, s1) - e(D["T"], s0, s1); b3 = boot(d3, blk); extra = f" | LW - T {100 * d3.mean() / e(D['T'], s0, s1).mean():+.1f}% ({b3[0]:+.3f} .. {b3[1]:+.3f})"
            print(f"   {nm:31s} {seg:10s} {row} | LW - AV {100 * d.mean() / e(D['AV'], s0, s1).mean():+.1f}% ({b[0]:+.3f} .. {b[1]:+.3f}) | LW - TS {100 * d2.mean() / e(D['TS'], s0, s1).mean():+.1f}% ({b2[0]:+.3f} .. {b2[1]:+.3f}){extra}")
