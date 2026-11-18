"""Second day ahead, 00:00-14:00 (slots 96-152), all seven sets: A, P, their average, and on the sealed sets the control. Notes 21.95."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0)
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95])
SETS = (("winter", "mid", "tsm_rl_winter", 3, None), ("summer", "mid", "tsm_rl_summer", 3, None), ("autumn", "v2", "tsm_rlv2_autumn", 7, None), ("winterf", "mid", "tsm_rl_winterf_fx", 3, None), ("summerf", "mid", "tsm_rl_summerf_fx", 3, None), ("winters", "mid", "tsm_rl_winters_fx", 3, "tsm_rl8_winters_fx"), ("summers", "mid", "tsm_rl_summers_fx", 3, "tsm_rl8_summers_fx")); out = {}
for per, base, pf, blk, p8 in SETS:
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float); T = np.load(f"{SP}/out/tsshift_{base}_{per}.npz"); P = np.load(f"{SP}/out/{pf}.npz")["median"].astype(float); G = np.load(f"{SP}/out/gen_{base}_{per}.npz"); rows = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(np.asarray(M.clear(th, T["xh"][j].astype(float), np.full(153, Lj), np.full(153, cj), eps=T["median"][j].astype(float) + T["nbh"][j].astype(float))) / 10)
    A = np.stack(rows); ok = ~np.isnan(a[:, 96:153]).any(1); a, A, P = a[ok], A[ok], P[ok]; out[per] = (a, A, P); e = lambda v, s0, s1: np.abs(v - a)[:, s0:s1].mean(1); B = 0.5 * (A + P); le = np.abs(G["rl_pes"].astype(float)[ok] - G["rl_true"].astype(float)[ok])
    d = e(B, 96, 153) - e(P, 96, 153); b = boot(d, blk); d2 = e(A, 96, 153) - e(P, 96, 153); b2 = boot(d2, blk); d1 = e(B, 0, 56) - e(P, 0, 56)
    line = f"{per:8s} ({len(a)} days) second day 00-14h: A {e(A, 96, 153).mean():.3f}  P {e(P, 96, 153).mean():.3f}  average {e(B, 96, 153).mean():.3f} | A-P {d2.mean():+.3f} ({b2[0]:+.3f} .. {b2[1]:+.3f}) | average-P {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f}) = {100 * d.mean() / e(P, 96, 153).mean():+.1f}% | same hours on the first day: A {e(A, 0, 56).mean():.3f} P {e(P, 0, 56).mean():.3f}, average-P {100 * d1.mean() / e(P, 0, 56).mean():+.1f}% | error correlation A,P: second day {np.corrcoef((A - a)[:, 96:153].mean(1), (P - a)[:, 96:153].mean(1))[0, 1]:.2f}, first day {np.corrcoef((A - a)[:, :56].mean(1), (P - a)[:, :56].mean(1))[0, 1]:.2f} | load-forecast error GW first / second day {le[:, :56].mean():.2f} / {le[:, 96:153].mean():.2f}"
    if p8:
        P8 = np.load(f"{SP}/out/{p8}.npz")["median"].astype(float)[ok]; C = 0.5 * (P + P8); d3 = e(B, 96, 153) - e(C, 96, 153); b3 = boot(d3, blk); d4 = e(C, 96, 153) - e(P, 96, 153); b4 = boot(d4, blk); line += f"\n         control: average of two price runs minus P {d4.mean():+.3f} ({b4[0]:+.3f} .. {b4[1]:+.3f}); average(A,P) minus that {d3.mean():+.3f} ({b3[0]:+.3f} .. {b3[1]:+.3f})"
    print(line)
np.savez_compressed(f"{SP}/out/day2_look.npz", **{f"{k}_{n}": v.astype(np.float32) for k, t in out.items() for n, v in zip(("a", "A", "P"), t)})
