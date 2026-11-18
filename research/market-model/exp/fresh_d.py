"""Version D on the fresh nights (notes 21.70): M1d curves, Timestral on its own shift. Paired against A and P from out/fresh_scores.npz."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1d as Dm
SP = Dm.SP; Q = np.load(f"{SP}/out/market1d_post.npz"); th = {k: float(np.median(Q[k])) for k in Dm.PN}; rng = np.random.default_rng(0); S = np.load(f"{SP}/out/fresh_scores.npz")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk=3):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(6000)], [5, 95])
mae, sg, cov = [], [], []
for per in ("winterf", "summerf"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :24]; T = np.load(f"{SP}/out/tsshift_mid_d_{per}.npz"); pr = {}
    for k in ("median", "q05", "q95"):
        rows = []
        for j, c in enumerate(I["cutoffs"]):
            t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; rows.append(Dm.clear(th, T["xh"][j].astype(float)[:24], np.full(24, Lj), np.full(24, cj), T[k][j].astype(float)[:24] + T["nbh"][j].astype(float)[:24]) / 10)
        pr[k] = np.stack(rows)
    mae.append(np.abs(pr["median"] - a).mean(1)); sg.append((pr["median"] - a).mean(1)); lo, hi = np.minimum(pr["q05"], pr["q95"]), np.maximum(pr["q05"], pr["q95"]); cov.append(((a >= lo) & (a <= hi)).ravel()); np.savez_compressed(f"{SP}/out/shiftfc_d_{per}.npz", **{k: v.astype(np.float32) for k, v in pr.items()})
D = np.concatenate(mae); n1 = int(S["n_winter"]); print(f"D on the 80 fresh nights: {D.mean():.3f} ct (winter {D[:n1].mean():.3f}, summer {D[n1:].mean():.3f}) | A {S['mae_A'].mean():.3f} | P {S['mae_P'].mean():.3f}")
for lab, o in (("D - A", S["mae_A"]), ("D - P", S["mae_P"])): d = D - o; b = boot(d); print(f"   {lab}: {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f})   winter {d[:n1].mean():+.3f} ({boot(d[:n1])[0]:+.3f} .. {boot(d[:n1])[1]:+.3f})   summer {d[n1:].mean():+.3f} ({boot(d[n1:])[0]:+.3f} .. {boot(d[n1:])[1]:+.3f})")
print(f"   raw 90% band slot coverage {np.concatenate(cov).mean():.2f}; night-level error correlation with A {np.corrcoef(np.concatenate(sg), S['signed_A'])[0, 1]:.2f}, with P {np.corrcoef(np.concatenate(sg), S['signed_P'])[0, 1]:.2f}; bias {np.concatenate(sg).mean():+.3f}")
