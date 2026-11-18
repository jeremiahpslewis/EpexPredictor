"""Generative market model, level M1h (notes 21.30): the merit order by fuel with block orders. Lignite and hard coal answer
al_f * hour's price + (1 - al_f) * day's mean price; gas answers the hour. Likelihood slope term: diagonal + rank one per day; a day clears by fixed point.
usage: market1h.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
from scipy.special import expit, logit
SP = M.SP; NU = M.NU; EPS = M.EPS; LSC = M.LSC; FU = M.FU; CAP = M.CAP; QC = M.QC; PN = M.PN + ("al_l", "al_c"); COLS = (("D", 30.0), ("L", 90.0), ("co2", 70.0), ("p", 80.0), ("X", 2.0), ("S", 0.0), ("lig", 8.0), ("coal", 3.0), ("gasg", 5.0))
def grid(H):
    G = H.copy(); G.index = G.index.tz_convert("Etc/GMT-1"); full = pd.date_range(G.index.min().floor("D"), G.index.max().floor("D") + pd.Timedelta(hours=23), freq="1h", tz="Etc/GMT-1"); G = G.reindex(full); v = G.p.notna().values
    A = {c: np.where(v, G[c].values, d).reshape(-1, 24).astype(np.float64) for c, d in COLS}; A["v"] = v.reshape(-1, 24); A["index"] = full; return A
def pack(A, q=None):
    q = q or {f: A[QC[f]] for f in FU}; return dict(p=jnp.asarray(A["p"]), X=jnp.asarray(A["X"]), S=jnp.asarray(A["S"]), lq=jnp.asarray(np.stack([logit(q[f] / CAP[f]) for f in FU])), wq=jnp.asarray(np.stack([q[f] * (1 - q[f] / CAP[f]) for f in FU])), qsum=jnp.asarray(sum(q.values())))
def model(D, L, co2, v, p=None, X=None, S=None, lq=None, wq=None, qsum=None):
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6))
    K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5)); B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_S = numpyro.sample("sd_S", dist.HalfNormal(3.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(3.0)); a_l = numpyro.sample("a_l", dist.Normal(15.0, 8.0)); a_c = numpyro.sample("a_c", dist.Normal(35.0, 12.0)); m_g = numpyro.sample("m_g", dist.LogNormal(np.log(1.3), 0.3))
    P = {f: (numpyro.sample(f"phi_{f}", dist.Beta(2.0, 12.0)), numpyro.sample(f"kap_{f}", dist.Beta(4.0, 2.0)), numpyro.sample(f"s_{f}", dist.LogNormal(np.log(0.3), 0.5)), numpyro.sample(f"sg_{f}", dist.HalfNormal(1.0))) for f in FU}; al = dict(l=numpyro.sample("al_l", dist.Beta(2.0, 2.0)), c=numpyro.sample("al_c", dist.Beta(2.0, 2.0)), g=1.0)
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); phi_ = lambda z: jnp.exp(Nd.log_prob(z)); r = p / L; zS = (r - m_st) / s_st; zz = (p - c_z) / s_z; cf = dict(l=a_l + co2, c=a_c + 0.85 * co2, g=m_g * L); nv = jnp.maximum(v.sum(1, keepdims=True), 1); pbar = jnp.where(v, p, 0.0).sum(1, keepdims=True) / nv
    gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)); gS = K_st * (2.0 * Nd.cdf(zS) - 1.0); own = x1 / L + 2.0 * K_st * phi_(zS) / (s_st * L) + K_x * phi_(zz) / s_z + 1e-12; cross = 0.0; ll = 0.0
    for j, f in enumerate(FU):
        ph, ka, s_, sg = P[f]; pe = al[f] * p + (1 - al[f]) * pbar; z = (pe / cf[f] - 1.0) / s_; ut = ph + (1 - ph) * ka * Nd.cdf(z); ll = ll + dist.StudentT(NU, 0.0, sg).log_prob(lq[j] - (jnp.log(ut) - jnp.log1p(-ut)))
        b = CAP[f] * (1 - ph) * ka * phi_(z) / (s_ * cf[f]) * wq[j] / (CAP[f] * ut * (1 - ut)); own = own + al[f] * b; cross = cross + (1 - al[f]) * b
    sh = jnp.abs(p) <= EPS; be = p < -EPS; free = v & ~sh; logdet = jnp.where(free, jnp.log(own), 0.0).sum() + jnp.log1p(jnp.where(free, cross / own, 0.0).sum(1) / nv[:, 0]).sum()
    u = D - qsum - X - S - B_g; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc; ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc))); ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be)
    numpyro.factor("ll", jnp.where(v, ll + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + dist.StudentT(NU, 0.0, sd_S).log_prob(S - gS) + jnp.where(sh, ld_shelf, ld_cont), 0.0).sum() + logdet)
AL = lambda th, f: 1.0 if f == "g" else th[f"al_{f}"]
def fuels(th, p, L, co2, pbar, e=None):
    cf = M.cost(th, L, co2); return {f: CAP[f] * expit(logit(M.util(th, f, AL(th, f) * p + (1 - AL(th, f)) * pbar, cf[f])) + (0.0 if e is None else e[f])) for f in FU}
def supply(th, p, L, co2, pbar, e=None): gX, gS = M.other_curves(th, p, L); return sum(fuels(th, p, L, co2, pbar, e).values()) + gX + gS + th["B_g"]
def clear_h(th, D, L, co2, pbar, eps=0.0, e=None, eX=0.0, eS=0.0):
    def ex(p, wd): return supply(th, p, L, co2, pbar, e) + eX + eS + eps - wd - D
    up = ex(np.full(np.shape(D), EPS), 0.0) < 0; dn = ex(np.full(np.shape(D), -EPS), th["K_s"]) > 0; lo = np.where(up, EPS, -3000.0); hi = np.where(up, 20000.0, -EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
def clear_day(th, D, L, co2, v, eps=0.0, e=None, eX=0.0, eS=0.0, iters=30):
    """arrays (groups, n): each row is one day (or part of one) that shares a mean price"""
    nv = np.maximum(v.sum(1, keepdims=True), 1); p = clear_h(th, D, L, co2, 80.0, eps, e, eX, eS); pbar = np.where(v, p, 0.0).sum(1, keepdims=True) / nv
    for _ in range(iters): p = clear_h(th, D, L, co2, pbar, eps, e, eX, eS); pbar = 0.5 * pbar + 0.5 * np.where(v, p, 0.0).sum(1, keepdims=True) / nv
    return p, pbar
def simulate(th, D, L, co2, v, rng):
    n = D.shape; e = {f: tdist.rvs(NU, size=n, random_state=rng) * th[f"sg_{f}"] for f in FU}; eX, eS = (tdist.rvs(NU, size=n, random_state=rng) * th[k] for k in ("sd_X", "sd_S")); p, pbar = clear_day(th, D, L, co2, v, rng.logistic(0.0, th["sd_d"] * LSC, n), e, eX, eS); gX, gS = M.other_curves(th, p, L); return p, fuels(th, p, L, co2, pbar, e), gX + eX, gS + eS
def fit(A, dd, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(A["D"]), jnp.asarray(A["L"]), jnp.asarray(A["co2"]), jnp.asarray(A["v"]), extra_fields=("diverging", "potential_energy"), **dd)
    s = {k: np.asarray(x) for k, x in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(x["r_hat"])) for x in summ.values()); ess = min(float(np.nanmin(x["n_eff"])) for x in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"]).mean(1)
if __name__ == "__main__":
    mode = sys.argv[1]; H = M.H; A = grid(H[np.asarray(H.index < "2025-12-01")]); rng = np.random.default_rng(0); pri = Predictive(model, num_samples=1000)(jax.random.PRNGKey(7), jnp.asarray(A["D"][:2]), jnp.asarray(A["L"][:2]), jnp.asarray(A["co2"][:2]), jnp.asarray(A["v"][:2])); pri = {k: np.asarray(x) for k, x in pri.items()}; q3 = lambda x: " ".join(f"{y:.3g}" for y in np.percentile(x, [5, 50, 95]))
    if mode == "prior":
        d0, nd = 120, 120; sl = slice(d0, d0 + nd); P = np.array([simulate({k: float(pri[k][i]) for k in PN}, A["D"][sl], A["L"][sl], A["co2"][sl], A["v"][sl], rng)[0] for i in range(60)]); po = np.where(A["v"][sl], A["p"][sl], np.nan); L = A["L"][sl]
        print(f"prior predictive, 60 markets on 120 real days (5-50-95%; observed): median price / cost {q3(np.median((P / L[None]).reshape(60, -1), 1))} ({np.nanmedian(po / L):.2f}); at or below zero {q3((P <= 0).mean((1, 2)))} ({np.nanmean(po <= 0):.3f}); sd of hour minus day's mean, price / cost {q3(((P - P.mean(2, keepdims=True)) / L[None]).std((1, 2)))} ({np.nanstd((po - np.nanmean(po, 1, keepdims=True)) / L):.2f}); highest price ct {q3(P.max((1, 2)) / 10)} ({np.nanmax(po) / 10:.0f})")
    if mode == "recover":
        TRUE = dict(x0=3.0, x1=8.0, K_x=4.0, c_z=5.0, s_z=7.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_X=4.0, sd_S=1.5, sd_d=2.0, a_l=12.0, a_c=40.0, m_g=1.4, phi_l=0.17, kap_l=0.6, s_l=0.3, sg_l=0.5, phi_c=0.05, kap_c=0.6, s_c=0.3, sg_c=0.9, phi_g=0.04, kap_g=0.6, s_g=0.4, sg_g=0.7, al_l=0.3, al_c=0.6)
        sl = slice(100, 400); B_ = {k: A[k][sl] for k in ("D", "L", "co2")}; v = A["v"][sl].copy(); p, q, X, S = simulate(TRUE, B_["D"], B_["L"], B_["co2"], v, rng); v &= rng.random(v.shape) > 0.03; s, rh, ess, div, pe = fit(dict(B_, v=v), pack(dict(p=p, X=X, S=S), q), warm=500, draws=400); nin = 0
        print(f"simulate and recover: {int(v.sum())} hours; worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in PN:
            qq = np.percentile(s[k], [5, 50, 95]); okk = qq[0] <= TRUE[k] <= qq[2]; nin += okk
            if not okk or k.startswith("al_"): print(f"   {k:6s} true {TRUE[k]:6.2f}  recovered {qq[1]:6.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})  {'ok' if okk else 'OUTSIDE the 90% band'}")
        print(f"   {nin} of {len(PN)} inside their 90% bands")
    if mode == "fit":
        s, rh, ess, div, pe = fit(A, pack(A), seed=int(os.environ.get("SEED", 0))); th = {k: float(np.median(s[k])) for k in PN}; old = {k: float(np.median(x)) for k, x in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(x) == 1}
        print(f"M1h fitted on {int(A['v'].sum())} hours: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['x0'])}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in PN: qq = np.percentile(s[k], [5, 50, 95]); pc = float((pri[k] < qq[1]).mean()); print(f"   {k:6s} {qq[1]:7.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})" + (f"   M1 {old[k]:7.2f}" if k in old else " " * 13) + f"   prior percentile {100 * pc:5.1f}{'   <-- conflict' if pc < 0.005 or pc > 0.995 else ''}")
        np.savez_compressed(f"{SP}/out/market1h_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div)
