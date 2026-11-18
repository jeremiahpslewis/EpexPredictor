"""Scores the frozen versions on the fresh nights exactly as pre-registered in notes 21.67. usage: fresh_score.py"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; P = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P[k])) for k in M.PN}; rng = np.random.default_rng(0); Z = np.load(f"{SP}/out/shift15.npz")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index.tz_convert("UTC")).ffill(); QS = ("median", "q05", "q25", "q75", "q95")
def cleared(per, tag):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_{per}.npz"); out = {k: [] for k in QS}
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj
        for k in QS: out[k].append(np.asarray(M.clear(th, T["xh"][j].astype(float)[:24], np.full(24, Lj), np.full(24, cj), eps=T[k][j].astype(float)[:24] + T["nbh"][j].astype(float)[:24])) / 10)
    o = {k: np.stack(v) for k, v in out.items()}; o["lo"] = np.minimum(o["q05"], o["q95"]); o["hi"] = np.maximum(o["q05"], o["q95"]); return o
def boot(d, blk=3):
    nb = len(d) // blk; x = d[:nb * blk].reshape(nb, blk).mean(1); return np.percentile([x[rng.integers(0, nb, nb)].mean() for _ in range(6000)], [5, 95])
# band scaling fitted on the USED nights only
U = []
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2")): a = np.load(f"{SP}/out/inputs_{per}.npz")["actual"].astype(float)[:, :24]; f = np.load(f"{SP}/out/shiftfc_{tag}_{per}.npz"); m = f["median"].astype(float)[:, :24]; U.append((a, m, np.minimum(f["q05"], f["q95"]).astype(float)[:, :24], np.maximum(f["q05"], f["q95"]).astype(float)[:, :24]))
def cover(sets, k): return float(np.mean(np.concatenate([((a >= m + k * (lo - m)) & (a <= m + k * (hi - m))).ravel() for a, m, lo, hi in sets])))
ks = np.linspace(1.0, 3.0, 201); kstar = float(ks[np.argmin([abs(cover(U, k) - 0.90) for k in ks])]); print(f"band scaling fitted on the 146 used nights: factor {kstar:.2f} (slot coverage there {cover(U, 1.0):.2f} -> {cover(U, kstar):.2f})")
ALL = {}
for per in ("winterf", "summerf"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :24]; A = cleared(per, "mid"); C = cleared(per, "mid_c"); Pf = np.load(f"{SP}/out/tsm_rl_{per}_fx.npz"); P0 = np.load(f"{SP}/out/tsm_rl_{per}.npz"); T = np.load(f"{SP}/out/tsshift_mid_{per}.npz")
    dx = np.array([T["xh"][j].astype(float)[:24].mean() - (Z["rl"][int(d0) - 96:int(d0) - 72] - Z["other"][int(d0) - 96:int(d0) - 72]).mean() for j, d0 in enumerate(I["cuts"])])
    ALL[per] = dict(a=a, A=A, C=C, P=Pf["median"].astype(float)[:, :24], P0=P0["median"].astype(float)[:, :24], Plo=Pf["q05"].astype(float)[:, :24], Phi=Pf["q95"].astype(float)[:, :24], dx=dx, used=set(int(c) for c in np.load(f"{SP}/out/inputs_{per[:-1]}.npz")["cuts"]) & set(int(c) for c in I["cuts"]))
    assert not ALL[per]["used"], "a fresh night equals a used one"
cat = lambda f: np.concatenate([f(ALL[p]) for p in ALL]); mae = lambda v, a: np.abs(v - a).mean(1)
E_ = {k: cat(lambda S, k=k: mae(S[k]["median"] if k in ("A", "C") else S[k], S["a"])) for k in ("A", "C", "P", "P0")}; S_ = {k: cat(lambda S, k=k: ((S[k]["median"] if k in ("A", "C") else S[k]) - S["a"]).mean(1)) for k in ("A", "C", "P", "P0")}; n1 = len(ALL["winterf"]["a"])
print(f"PRIMARY, 80 fresh nights pooled, night MAE 00:00-06:00 ct: A {E_['A'].mean():.3f} | C {E_['C'].mean():.3f} | P (price, look-ahead fixed) {E_['P'].mean():.3f} | P0 (price as before) {E_['P0'].mean():.3f}")
for lab, x, y in (("A - P", "A", "P"), ("C - A", "C", "A"), ("P - P0", "P", "P0")): d = E_[x] - E_[y]; b = boot(d); print(f"   {lab}: {d.mean():+.3f} ({b[0]:+.3f} .. {b[1]:+.3f})   winter {d[:n1].mean():+.3f} ({boot(d[:n1])[0]:+.3f} .. {boot(d[:n1])[1]:+.3f})   summer {d[n1:].mean():+.3f} ({boot(d[n1:])[0]:+.3f} .. {boot(d[n1:])[1]:+.3f})")
print("   by period (winter / summer):", {k: (round(float(v[:n1].mean()), 3), round(float(v[n1:].mean()), 3)) for k, v in E_.items()}, "| blend A+P:", round(float(cat(lambda S: mae(0.5 * (S['A']['median'] + S['P']), S['a'])).mean()), 3))
dx = cat(lambda S: S["dx"]); big = dx < -15; print(f"large-fall group (forecast fall of demand > 15 GW): {int(big.sum())} nights; signed night error ct: " + " | ".join(f"{k} {S_[k][big].mean():+.2f} ±{S_[k][big].std() / np.sqrt(max(big.sum(), 1)):.2f} (abs {np.abs(S_[k][big]).mean():.2f})" for k in ("A", "C", "P")) + f" || other nights: " + " | ".join(f"{k} {S_[k][~big].mean():+.2f} (abs {np.abs(S_[k][~big]).mean():.2f})" for k in ("A", "C", "P")))
for k, lab in (("A", "A"), ("C", "C")):
    sets = [(S["a"], S[k]["median"], S[k]["lo"], S[k]["hi"]) for S in ALL.values()]; nm = lambda kk: float(np.mean(np.concatenate([((a.mean(1) >= (m + kk * (lo - m)).mean(1)) & (a.mean(1) <= (m + kk * (hi - m)).mean(1))) for a, m, lo, hi in sets]))); w = float(np.mean(np.concatenate([(hi - lo).ravel() for a, m, lo, hi in sets])))
    print(f"bands {lab}: 90% band slot coverage raw {cover(sets, 1.0):.2f}, scaled {cover(sets, kstar):.2f}; night-mean inside the mean of the slot bounds raw {nm(1.0):.2f}, scaled {nm(kstar):.2f}; mean width raw {w:.2f} ct, scaled {w * kstar:.2f}")
sp = [(S["a"], S["P"], S["Plo"], S["Phi"]) for S in ALL.values()]; print(f"bands P: slot coverage {cover(sp, 1.0):.2f}; mean width {float(np.mean(np.concatenate([(hi - lo).ravel() for a, m, lo, hi in sp]))):.2f} ct")
print("night-level error correlation A with P:", round(float(np.corrcoef(S_["A"], S_["P"])[0, 1]), 2), "| C with A:", round(float(np.corrcoef(S_["C"], S_["A"])[0, 1]), 2))
np.savez_compressed(f"{SP}/out/fresh_scores.npz", **{f"mae_{k}": v for k, v in E_.items()}, **{f"signed_{k}": v for k, v in S_.items()}, dx=dx, n_winter=n1, kstar=kstar)
