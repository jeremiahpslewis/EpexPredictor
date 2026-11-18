"""Generative market model, level M0h (notes 21.24): M0e with a share (1 - a) of thermal capacity answering the DAY's mean price / cost, as block
orders do; the rest answers the hour. Hours of a day are coupled; slope term = determinant of diagonal + rank one per day; a day clears by fixed point.
usage: market0h.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market0e as B
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
SP = B.SP; NU = B.NU; EPS = B.EPS; LSC = B.LSC; PN = B.PN + ("a",)
def grid(H):
    G = H.copy(); G.index = G.index.tz_convert("Etc/GMT-1"); full = pd.date_range(G.index.min().floor("D"), G.index.max().floor("D") + pd.Timedelta(hours=23), freq="1h", tz="Etc/GMT-1"); G = G.reindex(full); v = G.p.notna().values
    A = {c: np.where(v, G[c].values, d).reshape(-1, 24).astype(np.float64) for c, d in (("D", 30.0), ("L", 90.0), ("p", 80.0), ("T", 15.0), ("X", 2.0), ("S", 0.0))}; A["v"] = v.reshape(-1, 24); A["index"] = full; return A
def model(D, L, v, p=None, T=None, X=None, S=None):
    F_T = numpyro.sample("F_T", dist.LogNormal(np.log(4.0), 0.4)); K_T = numpyro.sample("K_T", dist.LogNormal(np.log(35.0), 0.3)); m_T = numpyro.sample("m_T", dist.Normal(1.0, 0.3)); s_T = numpyro.sample("s_T", dist.LogNormal(np.log(0.4), 0.5))
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6)); K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5))
    B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6)); sd_T = numpyro.sample("sd_T", dist.HalfNormal(5.0)); sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_S = numpyro.sample("sd_S", dist.HalfNormal(3.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(3.0))
    K_z = numpyro.sample("K_z", dist.LogNormal(np.log(2.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6)); a = numpyro.sample("a", dist.Beta(2.0, 2.0))
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); phi = lambda z: jnp.exp(Nd.log_prob(z)); r = p / L; nv = jnp.maximum(v.sum(1, keepdims=True), 1); rbar = jnp.where(v, r, 0.0).sum(1, keepdims=True) / nv
    zT = (r - m_T) / s_T; zB = (rbar - m_T) / s_T; zS = (r - m_st) / s_st; zz = (p - c_z) / s_z
    gT = F_T + K_T * (a * Nd.cdf(zT) + (1 - a) * Nd.cdf(zB)) + K_z * Nd.cdf(zz); gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)); gS = K_st * (2.0 * Nd.cdf(zS) - 1.0)
    s_h = a * K_T * phi(zT) / (s_T * L) + x1 / L + 2.0 * K_st * phi(zS) / (s_st * L) + (K_z + K_x) * phi(zz) / s_z + 1e-12
    sh = jnp.abs(p) <= EPS; be = p < -EPS; free = v & ~sh; cday = (1 - a) * K_T * phi(zB) / s_T
    logdet = jnp.where(free, jnp.log(s_h), 0.0).sum() + jnp.log1p(cday[:, 0] / nv[:, 0] * jnp.where(free, 1.0 / (L * s_h), 0.0).sum(1)).sum()
    u = D - T - X - S - B_g; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc
    ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc))); ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be)
    ll = dist.StudentT(NU, 0.0, sd_T).log_prob(T - gT) + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + dist.StudentT(NU, 0.0, sd_S).log_prob(S - gS) + jnp.where(sh, ld_shelf, ld_cont)
    numpyro.factor("ll", jnp.where(v, ll, 0.0).sum() + logdet)
def block(th, rbar): return th["K_T"] * (1 - th["a"]) * norm.cdf((rbar - th["m_T"]) / th["s_T"])
def clear_day(th, D, L, v, eps=0.0, eT=0.0, eX=0.0, eS=0.0, iters=40):
    """D, L, v: (days, 24). The day's mean price / cost is found by damped iteration; the hours clear with the hourly share of thermal capacity."""
    thh = dict(th, K_T=th["a"] * th["K_T"]); sh = D.shape; z = lambda x: np.broadcast_to(x, sh).reshape(-1); nv = np.maximum(v.sum(1, keepdims=True), 1)
    p = B.clear(th, D.reshape(-1), L.reshape(-1), z(eps), z(eT), z(eX), z(eS))[0].reshape(sh); rbar = np.where(v, p / L, 0.0).sum(1, keepdims=True) / nv
    for _ in range(iters):
        p = B.clear(thh, D.reshape(-1), L.reshape(-1), z(eps), z(eT) + z(block(th, rbar)), z(eX), z(eS))[0].reshape(sh); rbar = 0.5 * rbar + 0.5 * np.where(v, p / L, 0.0).sum(1, keepdims=True) / nv
    gT, gX, gS = B.curves(thh, p, L); return p, gT + block(th, rbar) + eT, gX + eX, gS + eS, rbar
def simulate(th, D, L, v, rng):
    n = D.shape; eps = rng.logistic(0.0, th["sd_d"] * LSC, n); eT, eX, eS = (tdist.rvs(NU, size=n, random_state=rng) * th[k] for k in ("sd_T", "sd_X", "sd_S")); return clear_day(th, D, L, v, eps, eT, eX, eS)[:4]
def fit(A, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(A["D"]), jnp.asarray(A["L"]), jnp.asarray(A["v"]), p=jnp.asarray(A["p"]), T=jnp.asarray(A["T"]), X=jnp.asarray(A["X"]), S=jnp.asarray(A["S"]), extra_fields=("diverging",))
    s = {k: np.asarray(x) for k, x in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(x["r_hat"])) for x in summ.values()); ess = min(float(np.nanmin(x["n_eff"])) for x in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
if __name__ == "__main__":
    mode = sys.argv[1]; H = B.H; A = grid(H[np.asarray(H.index < "2025-12-01")]); rng = np.random.default_rng(0); pri = Predictive(model, num_samples=2000)(jax.random.PRNGKey(7), jnp.asarray(A["D"][:2]), jnp.asarray(A["L"][:2]), jnp.asarray(A["v"][:2])); pri = {k: np.asarray(x) for k, x in pri.items()}; q3 = lambda x: " ".join(f"{y:.3g}" for y in np.percentile(x, [5, 50, 95]))
    if mode == "prior":
        d0, nd = 120, 150; D, L, v = A["D"][d0:d0 + nd], A["L"][d0:d0 + nd], A["v"][d0:d0 + nd]; P = np.array([simulate({k: float(pri[k][i]) for k in PN}, D, L, v, rng)[0] for i in range(80)]); po = np.where(v, A["p"][d0:d0 + nd], np.nan); rr = P / L[None]; ro = po / L
        print(f"prior predictive, 80 markets on 150 real days (5-50-95%; observed): median price / cost {q3(np.median(rr.reshape(80, -1), 1))} ({np.nanmedian(ro):.2f}); at or below zero {q3((P <= 0).mean((1, 2)))} ({np.nanmean(po <= 0):.3f}); on the shelf {q3((P == 0).mean((1, 2)))} ({np.nanmean(np.abs(po) <= EPS):.3f}); sd of hour minus the day's mean, price / cost {q3((rr - rr.mean(2, keepdims=True)).std((1, 2)))} ({np.nanstd(ro - np.nanmean(ro, 1, keepdims=True)):.2f}); highest price ct {q3(P.max((1, 2)) / 10)} ({np.nanmax(po) / 10:.0f})")
    if mode == "recover":
        TRUE = dict(F_T=3.0, K_T=30.0, m_T=1.1, s_T=0.35, x0=3.0, x1=8.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_T=2.0, sd_X=4.0, sd_S=1.5, sd_d=2.0, K_z=2.0, K_x=4.0, c_z=8.0, s_z=7.0, a=0.4); nd = 300; D, L = A["D"][100:400], A["L"][100:400]; v = A["v"][100:400].copy()
        p, T, X, S = simulate(TRUE, D, L, v, rng); v &= rng.random(v.shape) > 0.03; s, rh, ess, div = fit(dict(D=D, L=L, v=v, p=p, T=T, X=X, S=S), warm=500, draws=400); nin = 0
        print(f"simulate and recover: {int(v.sum())} hours of 300 days; worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}")
        for k in PN:
            q = np.percentile(s[k], [5, 50, 95]); ok = q[0] <= TRUE[k] <= q[2]; nin += ok
            if not ok or k == "a": print(f"   {k:5s} true {TRUE[k]:6.2f}  recovered {q[1]:6.2f}  ({q[0]:.2f} .. {q[2]:.2f})  {'ok' if ok else 'OUTSIDE the 90% band'}")
        print(f"   {nin} of {len(PN)} inside their 90% bands")
    if mode == "fit":
        s, rh, ess, div = fit(A); th = {k: float(np.median(s[k])) for k in PN}; old = {k: float(np.median(x)) for k, x in np.load(f"{SP}/out/market0e_post.npz").items() if np.ndim(x) == 1}
        print(f"M0h fitted on {int(A['v'].sum())} hours: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['a'])}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); pc = float((pri[k] < q[1]).mean()); print(f"   {k:5s} {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})" + (f"   M0e {old[k]:7.2f}" if k in old else " " * 14) + f"   prior percentile {100 * pc:5.1f}{'   <-- conflict' if pc < 0.005 or pc > 0.995 else ''}")
        np.savez_compressed(f"{SP}/out/market0h_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div)
        F = grid(H); v = F["v"]; p, L, D = F["p"], F["L"], F["D"]; nv = np.maximum(v.sum(1, keepdims=True), 1); rbar = np.where(v, p / L, 0.0).sum(1, keepdims=True) / nv; thh = dict(th, K_T=th["a"] * th["K_T"]); gT, gX, gS = B.curves(thh, p, L); gT = gT + block(th, rbar)
        R = pd.DataFrame(index=F["index"]); nanv = lambda x: np.where(v, x, np.nan).reshape(-1)
        for c in ("D", "L", "p", "T", "X", "S"): R[c] = nanv(F[c])
        R["eT"] = nanv(F["T"] - gT); R["eX"] = nanv(F["X"] - gX); R["eS"] = nanv(F["S"] - gS); R["eps"] = nanv(D - F["T"] - F["X"] - F["S"] - th["B_g"] + th["K_s"] * (p < -EPS) + 0.5 * th["K_s"] * (np.abs(p) <= EPS)); R["rbar"] = nanv(np.broadcast_to(rbar, p.shape)); R["p0"] = nanv(clear_day(th, D, L, v)[0])
        R["tr"] = np.asarray(R.index < pd.Timestamp("2025-12-01", tz="Europe/Berlin")); R = R[R.p.notna()]; R.to_parquet(f"{SP}/out/market0h_rows.parquet"); R["dev"] = R.p / R.L - R.rbar
        for lab, g in (("fitting period", R[R.tr]), ("held-out year", R[~R.tr])):
            e = np.r_[-1.2, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.2]; md = [g[(g.dev >= lo) & (g.dev < hi)].eT.median() for lo, hi in zip(e[:-1], e[1:])]
            print(f"   {lab}: bare price MAE {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} ct; shock sd thermal {g.eT.std():.2f}, imports {g.eX.std():.2f}, storage {g.eS.std():.2f}, unmeasured {g.eps.std():.2f} GW; thermal shock against the hour's price minus the day's mean (binned medians): " + " ".join(f"{x:+.1f}" for x in md) + f" (range {np.nanmax(md) - np.nanmin(md):.1f} GW)")
