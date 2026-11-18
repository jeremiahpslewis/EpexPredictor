"""Whole target day by block of hours: market model with Timestral on the shift (A) against Timestral on the price (P). Notes 21.76."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0)
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def clearA(per, tag):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_{per}.npz"); rows = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(np.asarray(M.clear(th, T["xh"][j].astype(float)[:96], np.full(96, Lj), np.full(96, cj), eps=T["median"][j].astype(float)[:96] + T["nbh"][j].astype(float)[:96])) / 10)
    return np.stack(rows)
BL = (("night 00-06", 0, 24), ("morning 06-10", 24, 40), ("midday 10-16", 40, 64), ("evening 16-22", 64, 88), ("late 22-24", 88, 96), ("whole day", 0, 96))
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
out = {}
for grp, sets in (("USED nights", (("winter", "mid", "tsm_rl_winter", 3), ("summer", "mid", "tsm_rl_summer", 3), ("autumn", "v2", "tsm_rlv2_autumn", 7))), ("FRESH nights (post hoc)", (("winterf", "mid", "tsm_rl_winterf_fx", 3), ("summerf", "mid", "tsm_rl_summerf_fx", 3)))):
    print(grp)
    for per, tag, pf, blk in sets:
        a = np.load(f"{SP}/out/inputs_{per}.npz")["actual"].astype(float)[:, :96]; A = clearA(per, tag); Pp = np.load(f"{SP}/out/{pf}.npz")["median"].astype(float)[:, :96]; out[per] = (a, A, Pp); print(f"  {per} ({len(a)} days; share of slots at or below zero price {np.mean(a <= 0):.2f})")
        for lab, s0, s1 in BL:
            eA = np.abs(A - a)[:, s0:s1].mean(1); eP = np.abs(Pp - a)[:, s0:s1].mean(1); eB = np.abs(0.5 * (A + Pp) - a)[:, s0:s1].mean(1); d = eA - eP; b = boot(d, blk); db = eB - eP; bb = boot(db, blk); cr = np.corrcoef((A - a)[:, s0:s1].mean(1), (Pp - a)[:, s0:s1].mean(1))[0, 1]
            print(f"    {lab:14s} A {eA.mean():.2f}  P {eP.mean():.2f}  A-P {d.mean():+.2f} ({b[0]:+.2f} .. {b[1]:+.2f})  blend-P {db.mean():+.2f} ({bb[0]:+.2f} .. {bb[1]:+.2f})  bias A {(A - a)[:, s0:s1].mean():+.2f} P {(Pp - a)[:, s0:s1].mean():+.2f}  corr of block errors {cr:.2f}")
np.savez_compressed(f"{SP}/out/day_look.npz", **{f"{k}_{n}": v.astype(np.float32) for k, t in out.items() for n, v in zip(("a", "A", "P"), t)})
