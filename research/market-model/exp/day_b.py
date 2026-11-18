"""M1b (bounded imports) against M1n (A) and Timestral on the price (P): night, blocks, whole day, all five sets. Notes 21.83."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1b as Mb
SP = Mb.SP; Q = np.load(f"{SP}/out/market1b_post.npz"); th = {k: float(np.median(Q[k])) for k in Mb.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/day_look.npz")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
BL = (("night 00-06", 0, 24), ("morning 06-10", 24, 40), ("midday 10-16", 40, 64), ("evening 16-22", 64, 88), ("whole day", 0, 96)); out = {}
for per, tag, blk in (("winter", "mid_b", 3), ("summer", "mid_b", 3), ("autumn", "v2_b", 7), ("winterf", "mid_b", 3), ("summerf", "mid_b", 3)):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_{per}.npz"); a, A, P = (Z[f"{per}_{k}"].astype(float) for k in ("a", "A", "P")); rows = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(np.asarray(Mb.clear(th, T["xh"][j].astype(float)[:96], np.full(96, Lj), np.full(96, cj), eps=T["median"][j].astype(float)[:96] + T["nbh"][j].astype(float)[:96])) / 10)
    B = np.stack(rows); out[per] = B; print(f"{per} ({len(a)} days)")
    for lab, s0, s1 in BL:
        e = lambda v: np.abs(v - a)[:, s0:s1].mean(1); d = e(B) - e(A); b = boot(d, blk); d2 = e(0.5 * (B + P)) - e(P); b2 = boot(d2, blk); d3 = e(0.5 * (B + P)) - e(0.5 * (A + P)); b3 = boot(d3, blk)
        print(f"   {lab:14s} M1b {e(B).mean():.3f}  A {e(A).mean():.3f}  P {e(P).mean():.3f} | M1b-A {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f}) | avg(M1b,P)-P {d2.mean():+.3f} ({b2[0]:+.3f} .. {b2[1]:+.3f}) | avg(M1b,P)-avg(A,P) {d3.mean():+.3f} ({b3[0]:+.3f} .. {b3[1]:+.3f}) | bias M1b {(B - a)[:, s0:s1].mean():+.2f} A {(A - a)[:, s0:s1].mean():+.2f}")
    neg = a < -1.0; hi = a > np.quantile(a, 0.98)
    if neg.sum() > 20: print(f"   slots with actual price below -1 ct ({int(neg.sum())}, {neg.any(1).sum()} days): MAE M1b {np.abs(B - a)[neg].mean():.2f}  A {np.abs(A - a)[neg].mean():.2f}  P {np.abs(P - a)[neg].mean():.2f}; mean forecast M1b {B[neg].mean():+.1f} A {A[neg].mean():+.1f} P {P[neg].mean():+.1f} actual {a[neg].mean():+.1f}")
    print(f"   top 2% of prices ({int(hi.sum())} slots, above {np.quantile(a, 0.98):.1f} ct): MAE M1b {np.abs(B - a)[hi].mean():.2f}  A {np.abs(A - a)[hi].mean():.2f}  P {np.abs(P - a)[hi].mean():.2f}; bias M1b {(B - a)[hi].mean():+.2f} A {(A - a)[hi].mean():+.2f} P {(P - a)[hi].mean():+.2f}")
np.savez_compressed(f"{SP}/out/day_b.npz", **{k: v.astype(np.float32) for k, v in out.items()})
