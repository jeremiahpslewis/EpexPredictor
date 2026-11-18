"""Scores versions of the shift forecast (same model A curves) over the day by block, against A and P. usage: day_ver.py TAG [TAG ...]   (notes 21.87)"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/day_look.npz")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
BL = (("night 00-06", 0, 24), ("morning 06-10", 24, 40), ("midday 10-16", 40, 64), ("evening 16-22", 64, 88), ("whole day", 0, 96)); out = {}
for tag in sys.argv[1:]:
    print(f"===== version {tag.upper()} =====")
    for per, base, blk in (("winter", "mid", 3), ("summer", "mid", 3), ("autumn", "v2", 7), ("winterf", "mid", 3), ("summerf", "mid", 3)):
        I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{base}_{tag}_{per}.npz"); a, A, P = (Z[f"{per}_{k}"].astype(float) for k in ("a", "A", "P")); rows = []
        for j, c in enumerate(I["cutoffs"]):
            t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, Lj), np.full(96, cj), eps=T["median"][j].astype(float)[:96] + T["nbh"][j].astype(float)[:96])) / 10)
        B = np.stack(rows); out[f"{tag}_{per}"] = B; line = f"{per:8s}"
        for lab, s0, s1 in BL:
            e = lambda v: np.abs(v - a)[:, s0:s1].mean(1); d = e(B) - e(A); b = boot(d, blk); line += f" | {lab.split()[0]} {e(B).mean():.2f} vs A {e(A).mean():.2f}: {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f})"
        e = lambda v: np.abs(v - a).mean(1); d2 = e(B) - e(P); b2 = boot(d2, blk); d3 = e(0.5 * (B + P)) - e(P); b3 = boot(d3, blk); print(line); print(f"         whole day vs P ({e(P).mean():.2f}): {d2.mean():+.3f} ({b2[0]:+.3f} .. {b2[1]:+.3f}); average with P minus P {d3.mean():+.3f} ({b3[0]:+.3f} .. {b3[1]:+.3f}); evening bias {(B - a)[:, 64:88].mean():+.2f} (A {(A - a)[:, 64:88].mean():+.2f}); top 2% of prices bias {(B - a)[a > np.quantile(a, 0.98)].mean():+.2f} (A {(A - a)[a > np.quantile(a, 0.98)].mean():+.2f})")
np.savez_compressed(f"{SP}/out/day_ver_{'_'.join(sys.argv[1:])}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
