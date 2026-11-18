"""Price effect of the averaged load forecast (notes 21.114): slim route and price route with tree load (T), averaged load (AV), Timestral load (TS)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/day_look.npz"); I = np.load(f"{SP}/out/inputs_autumn.npz")
a = Z["autumn_a"].astype(float); tr = np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"].astype(float)[:, :96]; n = len(a)
def boot(d, blk=7):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
def slim(tag):
    T = np.load(f"{SP}/out/tsshift_{tag}_lite_autumn.npz"); return np.stack([np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(I["gas"][j]) + 0.37 * 75.0), np.full(96, 75.0), eps=T["median"][j].astype(float)[:96])) / 10 for j in range(n)])
def price(tag):
    z = np.load(f"{SP}/out/tsm_rl{tag}_autumn.npz"); assert (z["cutoffs"] == I["cutoffs"]).all(); return z["median"].astype(float)[:, :96]
A = {k: slim(t) for k, t in (("T", "appO"), ("AV", "avgA"), ("TS", "v2"))}; P = {k: price(t) for k, t in (("T", "appO"), ("AV", "avgA"), ("TS", "v2"))}
e = lambda v, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1)
V = {"slim route": A, "price route": P, "mix (trees + price + slim) / 3": {k: (tr + P[k] + A[k]) / 3 for k in A}, "average(price, slim)": {k: 0.5 * (P[k] + A[k]) for k in A}}
print("trees alone: night", round(e(tr, 0, 24).mean(), 3), "whole day", round(e(tr).mean(), 3))
for nm, D in V.items():
    print(nm)
    for seg, s0, s1 in (("night", 0, 24), ("whole day", 0, 96), ("evening 17-21h", 68, 84)):
        row = "  ".join(f"{k} {e(D[k], s0, s1).mean():.3f}" for k in ("T", "AV", "TS")); d = e(D["AV"], s0, s1) - e(D["T"], s0, s1); b = boot(d); d2 = e(D["AV"], s0, s1) - e(D["TS"], s0, s1); b2 = boot(d2)
        print(f"   {seg:15s} MAE ct: {row} | AV - T {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f}) = {100 * d.mean() / e(D['T'], s0, s1).mean():+.1f}%, better on {int((d < 0).sum())} of {n} | AV - TS {d2.mean():+.3f} ({b2[0]:+.3f} .. {b2[1]:+.3f})")
np.savez_compressed(f"{SP}/out/price_av.npz", a=a, trees=tr, **{f"slim_{k}": v for k, v in A.items()}, **{f"price_{k}": v for k, v in P.items()})
