"""Version B, second form (notes 21.66): grouped instrument. Hours are grouped by demand decile; the correction is the piecewise-linear curve through
(mean price / cost, mean misfit) of the groups, flat outside. Same carry-forward test."""
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10; P = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P[k])) for k in M.PN}; R = pd.read_parquet(f"{SP}/out/da_rows.parquet"); R["Dp"] = R.rl - R.other; R["r"] = R.p / R.L; R["nb"] = R.X - R.eX - M.other_curves(th, R.p.values, R.L.values)[0]
F = R[R.tr].dropna(subset=["sh_da", "eX_da", "Dp", "r"]).copy(); F["g"] = pd.qcut(F.Dp, NB, labels=False); G = F.groupby("g").agg(r=("r", "mean"), Dp=("Dp", "mean"), d=("sh_da", "mean"), x=("eX_da", "mean")); G["c"] = G.d + G.x; print(G.round(2).T.to_string()); assert (np.diff(G.r) > 0).all()
rk, ck = G.r.values, G.c.values; corr = lambda p, L, s=1.0: s * np.interp(np.asarray(p, float) / np.asarray(L, float), rk, ck); np.savez(f"{SP}/out/vb2_coef.npz", rk=rk, ck=ck)
def clear(D, L, co2, eps, s):
    def ex(p, wd): return M.supply(th, p, L, co2) + corr(p, L, s) + eps - wd - D
    up = ex(np.full(D.shape, M.EPS), 0.0) < 0; dn = ex(np.full(D.shape, -M.EPS), th["K_s"]) > 0; lo = np.where(up, M.EPS, -3000.0); hi = np.where(up, 20000.0, -M.EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
pg = np.linspace(-80, 300, 381)
for L_, c_ in ((60.0, 60.0), (90.0, 70.0), (130.0, 80.0)): sA = M.supply(th, pg, np.full_like(pg, L_), np.full_like(pg, c_)); sB = sA + corr(pg, L_); print(f"cost index {L_:.0f}: smallest slope GW per EUR, A {np.diff(sA).min():.3f}, B {np.diff(sB).min():.3f}; slope at r = 0.1 / 0.3 / 0.6 / 1.0 / 1.4: A", [round(float(np.interp(x * L_, pg[1:], np.diff(sA))), 2) for x in (0.1, 0.3, 0.6, 1.0, 1.4)], "B", [round(float(np.interp(x * L_, pg[1:], np.diff(sB))), 2) for x in (0.1, 0.3, 0.6, 1.0, 1.4)])
H = R[R.tr].dropna(subset=["p", "Dp", "L", "co2", "nb"]); full = pd.date_range(H.index.min().floor("D"), H.index.max().ceil("D"), freq="1h", tz=H.index.tz); H = H.reindex(full); wd = th["K_s"] * (H.p < -M.EPS) + 0.5 * th["K_s"] * (H.p.abs() <= M.EPS); S0 = M.supply(th, H.p.fillna(50).values, H.L.fillna(90).values, H.co2.fillna(70).values) - wd
uA = H.Dp - S0 - H.nb; uB = uA - corr(H.p.fillna(50).values, H.L.fillna(90).values); loc = H.index.tz_convert("Europe/Berlin"); night = np.asarray(loc.hour < 6); ok = (uA.notna() & uA.shift(24).notna() & H.L.shift(48).notna()).values & night; Lg, cg = H.L.shift(48).values, H.co2.shift(48).values
def run(u, s): p = np.full(len(H), np.nan); p[ok] = clear(H.Dp.values[ok], Lg[ok], cg[ok], (u.shift(24) + H.nb).values[ok], s); return p
pA, pB = run(uA, 0.0), run(uB, 1.0); T = pd.DataFrame(dict(day=np.asarray(loc.normalize().tz_localize(None))[ok], eA=(pA - H.p.values)[ok] / 10, eB=(pB - H.p.values)[ok] / 10, dD=(H.Dp - H.Dp.shift(24)).values[ok])); N = T.groupby("day").mean(); N["b"] = pd.cut(N.dD, [-99, -15, -10, -5, 5, 10, 15, 99]); rng = np.random.default_rng(0)
d = N.eB.abs() - N.eA.abs(); x = d.values[: len(d) // 7 * 7].reshape(-1, 7).mean(1); print(f"carry the previous night's shift, {len(N)} fitting nights: night-level MAE ct A {N.eA.abs().mean():.3f}, B {N.eB.abs().mean():.3f}; B minus A {d.mean():+.3f}", np.round(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)], [5, 95]), 3))
print(N.groupby("b").agg(n=("eA", "size"), A_abs=("eA", lambda x: x.abs().mean()), B_abs=("eB", lambda x: x.abs().mean()), A_signed=("eA", "mean"), B_signed=("eB", "mean")).round(2).T.to_string())
print("shift sd GW at night: A", round(float(uA[night].std()), 2), "B", round(float(uB[night].std()), 2), "| night-to-night change sd: A", round(float((uA - uA.shift(24))[night].std()), 2), "B", round(float((uB - uB.shift(24))[night].std()), 2), "| corr of the shift with demand at night: A", round(float(uA[night].corr(H.Dp[night])), 2), "B", round(float(uB[night].corr(H.Dp[night])), 2))
