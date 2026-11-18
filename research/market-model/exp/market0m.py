"""Generative market model, level M0m (notes 21.11): M0d with memory in its four shocks. Per block: a level per hour of day that reverts (rho) and learns
once a day (kap), plus dependence on the hour before (phi). Sequential likelihood: each hour given the past; slope term and shelf as M0d.
usage: market0m.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market0d as B
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
NOPHI = os.environ.get("NOPHI", "0") == "1"; TAG = "market0m0" if NOPHI else "market0m"   # NOPHI=1: day-to-day memory only, no dependence on the hour before
SP = B.SP; NU = B.NU; EPS = B.EPS; LSC = B.LSC; BL = ("T", "X", "S", "g"); PN = B.PN + tuple(f"{a}_{k}" for k in BL for a in (("rho", "kap") if NOPHI else ("rho", "kap", "phi"))); SDK = dict(T="sd_T", X="sd_X", S="sd_S", g="sd_d")
def grid(H):
    """hourly frame -> dict of (days, 24) arrays on UTC days, with a mask of valid hours"""
    G = H.copy(); G.index = G.index.tz_convert("UTC"); full = pd.date_range(G.index.min().floor("D"), G.index.max().floor("D") + pd.Timedelta(hours=23), freq="1h", tz="UTC"); G = G.reindex(full); v = G.p.notna().values
    A = {c: np.where(v, G[c].values, d).reshape(-1, 24).astype(np.float64) for c, d in (("D", 30.0), ("L", 90.0), ("p", 80.0), ("T", 15.0), ("X", 2.0), ("S", 0.0))}; A["v"] = v.reshape(-1, 24); A["index"] = full; return A
def model(D, L, v, p=None, T=None, X=None, S=None):
    F_T = numpyro.sample("F_T", dist.LogNormal(np.log(4.0), 0.4)); K_T = numpyro.sample("K_T", dist.LogNormal(np.log(35.0), 0.3)); m_T = numpyro.sample("m_T", dist.Normal(1.0, 0.3)); s_T = numpyro.sample("s_T", dist.LogNormal(np.log(0.4), 0.5))
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6))
    K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5))
    B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd = {k: numpyro.sample(SDK[k], dist.HalfNormal(5.0 if NOPHI else 3.0)) for k in BL}; rho = {k: numpyro.sample(f"rho_{k}", dist.Beta(8.0, 2.0)) for k in BL}; kap = {k: numpyro.sample(f"kap_{k}", dist.Beta(2.0, 6.0)) for k in BL}; phi = {k: (0.0 if NOPHI else numpyro.sample(f"phi_{k}", dist.Beta(6.0, 2.0))) for k in BL}
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); r = p / L; zT = (r - m_T) / s_T; zS = (r - m_st) / s_st
    gT = F_T + K_T * Nd.cdf(zT); gX = x0 + x1 * (r - 1.0); gS = K_st * (2.0 * Nd.cdf(zS) - 1.0)
    slope = K_T * jnp.exp(Nd.log_prob(zT)) / (s_T * L) + x1 / L + 2.0 * K_st * jnp.exp(Nd.log_prob(zS)) / (s_st * L)
    sh = jnp.abs(p) <= EPS; be = p < -EPS; u = D - T - X - S - B_g; E = dict(T=T - gT, X=X - gX, S=S - gS, g=u + K_s * be + 0.5 * K_s * sh); vf = v.reshape(-1); pv = jnp.concatenate([jnp.zeros(1, bool), vf[:-1]]); ll = jnp.where(sh | ~v, 0.0, jnp.log(slope + 1e-12)).sum()
    for k in BL:
        def step(m, x, rk=rho[k], kk=kap[k]):
            e_d, v_d = x; pred = rk * m; nu = e_d - pred; return jnp.where(v_d, pred + kk * nu, pred), nu
        _, nu = jax.lax.scan(step, jnp.zeros(24), (E[k], v)); f = jnp.where(vf, nu.reshape(-1), 0.0); om = f - jnp.where(pv, phi[k] * jnp.concatenate([jnp.zeros(1), f[:-1]]), 0.0); sc = sd[k] * jnp.where(pv, 1.0, 1.0 / jnp.sqrt(1.0 - phi[k] ** 2))
        if k != "g": ll = ll + jnp.where(vf, dist.StudentT(NU, 0.0, sc).log_prob(om), 0.0).sum()
        else:
            s_ = sc * LSC; a_ = (om - 0.5 * K_s) / s_; b_ = (om + 0.5 * K_s) / s_; shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / s_)))
            ll = ll + jnp.where(vf, jnp.where(sh.reshape(-1), shelf, dist.Logistic(0.0, s_).log_prob(om)), 0.0).sum()
    numpyro.factor("ll", ll)
def filt(th, E, v):
    """numpy version of the model's memory for one block: returns the level's prediction for the same day (rho m), what remains (omega), and the filtered level after each day"""
    out = {}
    for k in BL:
        r_, k_, p_ = th[f"rho_{k}"], th[f"kap_{k}"], th.get(f"phi_{k}", 0.0); m = np.zeros(24); pred = np.zeros_like(E[k]); lev = np.zeros_like(E[k])
        for d in range(E[k].shape[0]): pred[d] = r_ * m; nu = E[k][d] - pred[d]; m = np.where(v[d], pred[d] + k_ * nu, pred[d]); lev[d] = m
        f = np.where(v, E[k] - pred, 0.0).reshape(-1); vf = v.reshape(-1); pv = np.r_[False, vf[:-1]]; om = f - np.where(pv, p_ * np.r_[0.0, f[:-1]], 0.0); out[k] = (pred, np.where(vf, om, np.nan).reshape(E[k].shape), lev)
    return out
def shocks(th, D, L, p, T, X, S):
    gT, gX, gS = B.curves(th, p, L); sh = np.abs(p) <= EPS; be = p < -EPS; return dict(T=T - gT, X=X - gX, S=S - gS, g=D - T - X - S - th["B_g"] + th["K_s"] * be + 0.5 * th["K_s"] * sh)
def sim_shocks(th, nd, rng):
    out = {}
    for k in BL:
        r_, k_, p_ = th[f"rho_{k}"], th[f"kap_{k}"], th.get(f"phi_{k}", 0.0); s_ = th[SDK[k]]; w = (rng.logistic(0.0, s_ * LSC, (nd, 24)) if k == "g" else tdist.rvs(NU, size=(nd, 24), random_state=rng) * s_); e = np.zeros((nd, 24)); m = np.zeros(24); prev = 0.0
        for d in range(nd):
            nu = np.zeros(24)
            for h in range(24): prev = p_ * prev + w[d, h]; nu[h] = prev
            pred = r_ * m; e[d] = pred + nu; m = pred + k_ * nu
        out[k] = e
    return out
def simulate(th, D, L, rng):
    e = sim_shocks(th, D.shape[0], rng); p, T, X, S = B.clear(th, D.reshape(-1), L.reshape(-1), eps=e["g"].reshape(-1), eT=e["T"].reshape(-1), eX=e["X"].reshape(-1), eS=e["S"].reshape(-1)); return tuple(a.reshape(D.shape) for a in (p, T, X, S))
def fit(A, warm=600, draws=500, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(A["D"]), jnp.asarray(A["L"]), jnp.asarray(A["v"]), p=jnp.asarray(A["p"]), T=jnp.asarray(A["T"]), X=jnp.asarray(A["X"]), S=jnp.asarray(A["S"]), extra_fields=("diverging",))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
if __name__ == "__main__":
    mode = sys.argv[1]; H = B.H; A = grid(H[B.TR]); rng = np.random.default_rng(0); pri = Predictive(model, num_samples=2000)(jax.random.PRNGKey(7), jnp.asarray(A["D"][:2]), jnp.asarray(A["L"][:2]), jnp.asarray(A["v"][:2])); pri = {k: np.asarray(v) for k, v in pri.items()}
    if mode == "prior":
        nd = 182; d0 = 120; D, L = A["D"][d0:d0 + nd], A["L"][d0:d0 + nd]; P = []; q = lambda v: " ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95]))
        for i in range(100): th = {k: float(pri[k][i]) for k in PN}; P.append(simulate(th, D, L, rng)[0].astype(np.float32))
        P = np.array(P); po = A["p"][d0:d0 + nd]; vo = A["v"][d0:d0 + nd]; rr = P / L[None]; ro = np.where(vo, po / L, np.nan); dm = rr.mean(2); dmo = np.nanmean(ro, 1); a1 = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
        np.savez_compressed(f"{SP}/out/{TAG}_prior.npz", P=P, p_obs=np.where(vo, po, np.nan), L=L, D=D, **{k: pri[k][:100] for k in PN})
        print(f"prior predictive with memory, 100 markets on 182 real days (5-50-95% over markets; observed): median price / cost {q(np.median(rr.reshape(100, -1), 1))} ({np.nanmedian(ro):.2f}); at or below zero {q((P <= 0).mean((1, 2)))} ({np.nanmean(np.where(vo, po <= 0, np.nan)):.3f}); sd of the day's mean price / cost {q(dm.std(1))} ({np.nanstd(dmo):.2f}); day-to-day correlation of the day's mean {q([a1(x) for x in dm])} ({a1(dmo[~np.isnan(dmo)]):.2f}); sd of hour-to-hour change in price / cost {q(np.diff(rr.reshape(100, -1), axis=1).std(1))} ({np.nanstd(np.diff(ro.reshape(-1))):.3f}); highest price ct {q(P.max((1, 2)) / 10)} ({np.nanmax(np.where(vo, po, np.nan)) / 10:.0f})")
    if mode == "recover":
        TRUE = dict(F_T=5.0, K_T=30.0, m_T=1.1, s_T=0.35, x0=1.0, x1=10.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_T=(3.3 if NOPHI else 1.7), sd_X=(4.1 if NOPHI else 2.1), sd_S=(1.5 if NOPHI else 0.8), sd_d=(1.4 if NOPHI else 0.7), rho_T=0.97, kap_T=0.25, phi_T=0.85, rho_X=0.9, kap_X=0.2, phi_X=0.85, rho_S=0.9, kap_S=0.15, phi_S=0.85, rho_g=0.9, kap_g=0.15, phi_g=0.87)
        TRUE = {k: v for k, v in TRUE.items() if k in PN}
        nd = 300; S_ = {c: A[c][100:100 + nd].copy() for c in ("D", "L", "v")}; p, T, X, S = simulate(TRUE, S_["D"], S_["L"], rng); v = S_["v"].copy(); v &= rng.random(v.shape) > 0.03
        for g0 in rng.integers(5, nd - 8, 5): v[g0:g0 + rng.integers(1, 5)] = False
        s, rh, ess, div = fit(dict(D=S_["D"], L=S_["L"], v=v, p=p, T=T, X=X, S=S), warm=500, draws=400); nin = 0
        print(f"simulate and recover with memory: {int(v.sum())} hours of 300 days (3% missing at random, five gaps of 1-4 days); worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); ok = q[0] <= TRUE[k] <= q[2]; nin += ok; print(f"   {k:6s} true {TRUE[k]:6.2f}  recovered {q[1]:6.2f}  ({q[0]:.2f} .. {q[2]:.2f})  {'ok' if ok else 'OUTSIDE the 90% band'}")
        print(f"   {nin} of {len(PN)} inside their 90% bands"); np.savez_compressed(f"{SP}/out/{TAG}_recover.npz", **{f"true_{k}": TRUE[k] for k in PN}, **{k: s[k] for k in PN}, rhat=rh, ess=ess)
    if mode == "fit":
        s, rh, ess, div = fit(A); th = {k: float(np.median(s[k])) for k in PN}; old = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market0d_post.npz").items() if np.ndim(v) == 1}
        print(f"M0m fitted on {int(A['v'].sum())} hours (Mar 2024 - Nov 2025): worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['B_g'])}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); pc = float((pri[k] < q[1]).mean()); print(f"   {k:6s} {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})" + (f"   without memory {old[k]:7.2f}" if k in old else " " * 25) + f"   prior percentile {100 * pc:5.1f}{'   <-- prior and data in conflict' if pc < 0.005 or pc > 0.995 else ''}")
        np.savez_compressed(f"{SP}/out/{TAG}_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div)
        F = grid(H); E = shocks(th, *(F[c] for c in ("D", "L", "p", "T", "X", "S"))); fl = filt(th, E, F["v"]); R = pd.DataFrame(index=F["index"]); nd = F["v"].shape[0]
        for c in ("D", "L", "p", "T", "X", "S"): R[c] = np.where(F["v"], F[c], np.nan).reshape(-1)
        for k in BL:
            R[f"e_{k}"] = np.where(F["v"], E[k], np.nan).reshape(-1); R[f"om_{k}"] = fl[k][1].reshape(-1); lev = fl[k][2]
            for h in (1, 2, 4, 7):   # forecast of the shock for day d from an origin on day d - h (level filtered through day d - h - 1)
                f = np.zeros_like(lev); f[h + 1:] = th[f"rho_{k}"] ** (h + 1) * lev[: nd - h - 1]; R[f"f{h}_{k}"] = f.reshape(-1)
        ok = R.p.notna().values; R["p0"] = np.nan; R.loc[ok, "p0"] = B.clear(th, R.D.values[ok], R.L.values[ok])[0]
        for h in (1, 2, 4, 7): R[f"p{h}"] = np.nan; R.loc[ok, f"p{h}"] = B.clear(th, R.D.values[ok], R.L.values[ok], eps=R[f"f{h}_g"].values[ok], eT=R[f"f{h}_T"].values[ok], eX=R[f"f{h}_X"].values[ok], eS=R[f"f{h}_S"].values[ok])[0]
        R["tr"] = np.asarray(R.index < pd.Timestamp("2025-12-01", tz="Europe/Berlin")); R = R[ok]; R.to_parquet(f"{SP}/out/{TAG}_rows.parquet"); ac = lambda x, k: float(pd.Series(x).autocorr(k))
        for k in BL: x = fl[k][1].reshape(-1); print(f"   block {k}: what remains has sd {np.nanstd(x):.2f} GW (shock sd {np.nanstd(R[f'e_{k}']):.2f}); autocorrelation at 1 h {ac(x, 1):+.2f}, 24 h {ac(x, 24):+.2f}, 168 h {ac(x, 168):+.2f}")
        hod = R.index.tz_convert("Europe/Berlin").hour
        for lab, m in (("fitting period", R.tr.values), ("held-out year", ~R.tr.values)):
            g = R[m]; ng = g[np.asarray(hod[m] < 8)]; print(f"   {lab}: hourly price MAE ct, all hours / night: no state {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} / {np.mean(np.abs(ng.p0 - ng.p)) / 10:.2f}; " + "; ".join(f"+{h}d {np.mean(np.abs(g[f'p{h}'] - g.p)) / 10:.2f} / {np.mean(np.abs(ng[f'p{h}'] - ng.p)) / 10:.2f}" for h in (1, 2, 4, 7)))
