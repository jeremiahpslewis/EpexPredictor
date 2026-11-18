"""Scores the market cleared with the Timestral forecast of the supply shift (notes 21.51) on the key metric, with the control. usage: shift_score.py [GEN=mid]"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, os
NONB = os.environ.get("NONB", "0") == "1"; SFX = "_m1nonb" if NONB else ""
SP = M.SP; GEN = sys.argv[1] if len(sys.argv) > 1 else "mid"; P = np.load(f"{SP}/out/market1{'' if NONB else 'n'}_post.npz"); th = {k: float(np.median(P[k])) for k in M.PN}; rng = np.random.default_rng(0)
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
def parts(e):
    lev = e.mean(1, keepdims=True); hr = np.repeat(e.reshape(len(e), 6, 4).mean(2), 4, 1); return np.abs(lev).mean(), np.abs(hr - lev).mean(), np.abs(e - hr).mean()
for per in os.environ.get("PERS", "winter,summer,autumn").split(","):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float); Z = np.load(f"{SP}/out/tsshift_{GEN}{SFX}_{per}.npz"); n = len(a); pr = {}
    for k in ("median", "q05", "q10", "q25", "q75", "q90", "q95"):
        rows = []
        for j, c in enumerate(I["cutoffs"]):
            t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(np.asarray(M.clear(th, Z["xh"][j].astype(float), np.full(153, Lj), np.full(153, cj), eps=Z[k][j].astype(float) + Z["nbh"][j].astype(float))) / 10)
        pr[k] = np.stack(rows)
    np.savez_compressed(f"{SP}/out/shiftfc_{GEN}{SFX}_{per}.npz", **{k: v.astype(np.float32) for k, v in pr.items()})
    t = np.load(f"{SP}/out/tsm_rl_{per}.npz")["median"].astype(float); ker = np.load(f"{SP}/out/structfc_m1nf2_{per}_price.npz")["vint_ts"].astype(float); s = pr["median"]; m = lambda v: np.abs(v - a)[:, :24].mean(1); blk = 7 if per == "autumn" else 3
    try: t8 = np.load(f"{SP}/out/tsm_rl8_{per}.npz")["median"].astype(float)
    except Exception: t8 = None
    lo, hi = np.minimum(pr["q05"], pr["q95"]), np.maximum(pr["q05"], pr["q95"]); cov = ((a >= lo) & (a <= hi))[:, :24].mean(); lo2, hi2 = np.minimum(pr["q25"], pr["q75"]), np.maximum(pr["q25"], pr["q75"]); cov2 = ((a >= lo2) & (a <= hi2))[:, :24].mean()
    print(f"{per}: kernel layer {m(ker).mean():.2f} | shift by Timestral {m(s).mean():.2f} (level, shape, within-hour {tuple(round(float(x), 2) for x in parts((s - a)[:, :24]))}; bias {(s - a)[:, :24].mean():+.2f}) | Timestral price {m(t).mean():.2f}" + (f" | Timestral price + neighbours {m(t8).mean():.2f}" if t8 is not None else "") + f" | blends: shift+price {m(0.5 * (s + t)).mean():.2f}, shift+kernel {m(0.5 * (s + ker)).mean():.2f}, all three {m((s + ker + t) / 3).mean():.2f}")
    for lab, d in (("shift - kernel", m(s) - m(ker)), ("shift - Timestral price", m(s) - m(t)), ("shift+price blend - Timestral price", m(0.5 * (s + t)) - m(t))) + ((("price with neighbours - price", m(t8) - m(t)),) if t8 is not None else ()): print(f"     {lab}: {d.mean():+.3f} ({boot(d, blk)[0]:+.3f} .. {boot(d, blk)[1]:+.3f})")
    print(f"     bands from the shift quantiles, night slots: 90% band covers {cov:.2f} (mean width {(hi - lo)[:, :24].mean():.2f} ct), 50% band covers {cov2:.2f}; Timestral price 90% band covers {((a >= np.load(f'{SP}/out/tsm_rl_{per}.npz')['q05']) & (a <= np.load(f'{SP}/out/tsm_rl_{per}.npz')['q95']))[:, :24].mean():.2f} (width {(np.load(f'{SP}/out/tsm_rl_{per}.npz')['q95'] - np.load(f'{SP}/out/tsm_rl_{per}.npz')['q05'])[:, :24].mean():.2f})")
