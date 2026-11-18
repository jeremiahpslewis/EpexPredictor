"""Scores the sealed nights exactly as pre-registered in notes 21.92. usage: sealed_score.py"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; rng = np.random.default_rng(0); TAU = np.array([0.05, 0.10, 0.50, 0.90, 0.95]); QK = ("q05", "q10", "median", "q90", "q95")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill()
def boot(d, blk=3):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(6000)], [5, 95])
def pin(Q, a): d = a[None] - Q; return np.maximum(TAU[:, None, None] * d, (TAU[:, None, None] - 1) * d)
S = {}
for per in ("winters", "summers"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :96]; T = np.load(f"{SP}/out/tsshift_mid_{per}.npz"); Pz = np.load(f"{SP}/out/tsm_rl_{per}_fx.npz"); P8 = np.load(f"{SP}/out/tsm_rl8_{per}_fx.npz")["median"].astype(float)[:, :96]; n = len(a); QB = np.zeros((5, n, 96))
    used = set(int(c) for b in (per[:-1], per[:-1] + "f") for c in np.load(f"{SP}/out/inputs_{b}.npz")["cuts"]); assert not (used & set(int(c) for c in I["cuts"])), "a sealed night equals a used or fresh one"
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; Lv, cv = np.full(96, Lj), np.full(96, cj); xh = T["xh"][j].astype(float)[:96]; nbh = T["nbh"][j].astype(float)[:96]; um = T["median"][j].astype(float)[:96]; g = np.load(f"{SP}/out/tsgen_mid_{per}/{str(c)[:10]}.npz"); gm = g["median"].astype(float)[:, :96]
        for i, (tau, k) in enumerate(zip(TAU, QK)):
            du = np.abs(T[QK[4 - i]][j].astype(float)[:96] - um) if k != "median" else 0.0; dx = np.sqrt(((g[k].astype(float)[:, :96] - gm) ** 2).sum(0)) if k != "median" else 0.0; QB[i, j] = np.asarray(M.clear(th, xh + np.sign(tau - 0.5) * np.sqrt(du ** 2 + dx ** 2), Lv, cv, eps=um + nbh)) / 10
    QP = np.sort(np.stack([Pz[k].astype(float)[:, :96] for k in QK]), 0); A = QB[2].copy(); QB = np.sort(QB, 0); S[per] = dict(a=a, A=A, P=QP[2], P8=P8, QA=QB, QP=QP, QV=0.5 * (QB + QP))
cat = lambda f: np.concatenate([f(S[p]) for p in S]); n1 = len(S["winters"]["a"]); mae = lambda v, a, s0=0, s1=96: np.abs(v - a)[:, s0:s1].mean(1)
E_ = {"A": cat(lambda s: mae(s["A"], s["a"])), "P": cat(lambda s: mae(s["P"], s["a"])), "B": cat(lambda s: mae(0.5 * (s["A"] + s["P"]), s["a"])), "C": cat(lambda s: mae(0.5 * (s["P"] + s["P8"]), s["a"])), "P8": cat(lambda s: mae(s["P8"], s["a"]))}
print(f"SEALED NIGHTS, {len(E_['A'])} days. Whole-day MAE ct: A {E_['A'].mean():.3f} | P {E_['P'].mean():.3f} | average of A and P {E_['B'].mean():.3f} | second price run {E_['P8'].mean():.3f} | average of P and the second price run {E_['C'].mean():.3f}")
def line(lab, d): b = boot(d); return f"   {lab}: {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f})   winter {d[:n1].mean():+.3f} ({boot(d[:n1])[0]:+.3f} .. {boot(d[:n1])[1]:+.3f})   summer {d[n1:].mean():+.3f} ({boot(d[n1:])[0]:+.3f} .. {boot(d[n1:])[1]:+.3f})"
print(line("PRIMARY  average(A, P) minus P, whole day", E_["B"] - E_["P"])); print(line("control  average(P, second price run) minus P", E_["C"] - E_["P"])); print(line("         average(A, P) minus average(P, second price run)", E_["B"] - E_["C"])); print(line("         A minus P, whole day", E_["A"] - E_["P"]))
nA = cat(lambda s: mae(s["A"], s["a"], 0, 24)); nP = cat(lambda s: mae(s["P"], s["a"], 0, 24)); print(f"night 00-06: A {nA.mean():.3f}, P {nP.mean():.3f}"); print(line("         A minus P, night", nA - nP))
pl = {k: cat(lambda s, k=k: pin(s[k], s["a"]).mean((0, 2))) for k in ("QA", "QP", "QV")}; print(f"mean pinball loss, whole day: market model {pl['QA'].mean():.4f} | price route {pl['QP'].mean():.4f} | quantile average {pl['QV'].mean():.4f}"); print(line("SECONDARY quantile average minus price route (pinball)", pl["QV"] - pl["QP"]) + f"   = {100 * (pl['QV'] - pl['QP']).mean() / pl['QP'].mean():+.1f}%")
for k, nm in (("QA", "market model"), ("QP", "price route"), ("QV", "quantile average")):
    c90 = np.mean(np.concatenate([((s["a"] >= s[k][0]) & (s["a"] <= s[k][4])).ravel() for s in S.values()])); c80 = np.mean(np.concatenate([((s["a"] >= s[k][1]) & (s["a"] <= s[k][3])).ravel() for s in S.values()])); pk = [(s[k][2] >= np.quantile(s[k][2], 0.9)) for s in S.values()]; ex = np.mean(np.concatenate([(s["a"] > s[k][4])[m] for s, m in zip(S.values(), pk)])); ex9 = np.mean(np.concatenate([(s["a"] > s[k][3])[m] for s, m in zip(S.values(), pk)])); print(f"   {nm}: 90% interval covers {c90:.3f}, 80% interval {c80:.3f}; in the tenth of slots with the highest forecast: above q95 {ex:.3f}, above q90 {ex9:.3f}")
np.savez_compressed(f"{SP}/out/sealed_scores.npz", **{f"mae_{k}": v for k, v in E_.items()}, night_A=nA, night_P=nP, **{f"pin_{k}": v for k, v in pl.items()}, n_winter=n1)
