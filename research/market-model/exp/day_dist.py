"""Whole-day predictive distributions at five quantile levels: A (shift only), A+ (shift and load uncertainty), P (price route), V (quantile average of A+ and P). Notes 21.88."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; TAU = np.array([0.05, 0.10, 0.50, 0.90, 0.95]); QK = ("q05", "q10", "median", "q90", "q95")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
SETS = (("winter", "mid", "tsgen_mid_winter", "tsm_rl_winter"), ("summer", "mid", "tsgen_mid_summer", "tsm_rl_summer"), ("autumn", "v2", "tsgen_mid_v2_autumn", "tsm_rlv2_autumn"), ("winterf", "mid", "tsgen_mid_winterf", "tsm_rl_winterf_fx"), ("summerf", "mid", "tsgen_mid_summerf", "tsm_rl_summerf_fx")); BL = (("night", 0, 24), ("morning", 24, 40), ("midday", 40, 64), ("evening", 64, 88), ("whole day", 0, 96)); out = {}
def pin(q, a): d = a[None] - q; return np.maximum(TAU[:, None, None] * d, (TAU[:, None, None] - 1) * d)       # (5, n, 96)
for per, base, gdir, pf in SETS:
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :96]; T = np.load(f"{SP}/out/tsshift_{base}_{per}.npz"); Pz = np.load(f"{SP}/out/{pf}.npz"); n = len(a); QA = np.zeros((5, n, 96)); QB = QA.copy()
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; Lv, cv = np.full(96, Lj), np.full(96, cj); xh = T["xh"][j].astype(float)[:96]; nbh = T["nbh"][j].astype(float)[:96]; um = T["median"][j].astype(float)[:96]; g = np.load(f"{SP}/out/{gdir}/{str(c)[:10]}.npz"); gm = g["median"].astype(float)[:, :96]
        for i, (tau, k) in enumerate(zip(TAU, QK)):
            ku = QK[4 - i]; du = np.abs(T[ku][j].astype(float)[:96] - um) if k != "median" else 0.0      # a HIGH price quantile needs a LOW shift quantile
            dx = np.sqrt(((g[k].astype(float)[:, :96] - gm) ** 2).sum(0)) if k != "median" else 0.0; sg = np.sign(tau - 0.5)
            QA[i, j] = np.asarray(M.clear(th, xh + sg * du, Lv, cv, eps=um + nbh)) / 10; QB[i, j] = np.asarray(M.clear(th, xh + sg * np.sqrt(du ** 2 + dx ** 2), Lv, cv, eps=um + nbh)) / 10
    QP = np.stack([Pz[k].astype(float)[:, :96] for k in QK]); QA, QB, QP = (np.sort(q, 0) for q in (QA, QB, QP)); QV = 0.5 * (QB + QP); out[per] = dict(a=a, A=QA, Ap=QB, P=QP, V=QV); hi = a > np.quantile(a, 0.98); print(f"{per} ({n} days)")
    for nm, Q in (("A  shift only", QA), ("A+ shift + load", QB), ("P  price route", QP), ("V  average of A+ and P", QV)):
        pl = pin(Q, a); line = f"   {nm:24s}"
        for lab, s0, s1 in BL: line += f" | {lab} pinball {pl[:, :, s0:s1].mean():.3f} cov90 {((a >= Q[0]) & (a <= Q[4]))[:, s0:s1].mean():.2f}"
        print(line + f" || whole day: cov80 {((a >= Q[1]) & (a <= Q[3])).mean():.2f}, width90 {(Q[4] - Q[0]).mean():.2f} ct, pinball at 95% {pl[4].mean():.3f}, at 5% {pl[0].mean():.3f} | top 2% of prices: at or below q95 {(a <= Q[4])[hi].mean():.2f}, pinball at 95% {pl[4][hi].mean():.2f}, mean q95 {Q[4][hi].mean():.1f} vs actual {a[hi].mean():.1f} | evening: above q95 {(a > Q[4])[:, 64:88].mean():.3f} (nominal 0.05)")
np.savez_compressed(f"{SP}/out/day_dist.npz", **{f"{p}_{k}": v.astype(np.float32) for p, d in out.items() for k, v in d.items()})
