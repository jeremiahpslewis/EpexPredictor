"""Generative market model, level M0 (notes 21.2): two supply curves (domestic thermal, net imports) meet demand (residual load, with absorption
near zero price); three shocks in GW; one likelihood (shock densities times the slope of excess supply at the observed price).
usage: market0.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; numpyro.set_host_device_count(4); NU = 4.0
H = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(H) * 4 - len(E)), np.nan)]; H["co2"] = np.nanmean(E[: len(H) * 4].reshape(-1, 4), 1)
H["L"] = 1.85 * H.gas + 0.37 * H.co2; H["X"] = H.trade; H.index = H.index.tz_convert("Europe/Berlin"); ok = H[["p", "rlreal", "T", "X", "L"]].notna().all(1) & (H.L > 20); H = H[ok]
TR = np.asarray(H.index < "2025-12-01"); HO = ~TR
PN = ("F_T", "K_T", "m_T", "s_T", "x0", "x1", "K_0", "c_0", "s_0", "K_s", "B", "sd_d", "sd_T", "sd_X"); EPS = 0.1; LSC = np.sqrt(3.0) / np.pi   # logistic scale per unit sd
def model(rl, L, p=None, T=None, X=None):
    F_T = numpyro.sample("F_T", dist.LogNormal(np.log(4.0), 0.4)); K_T = numpyro.sample("K_T", dist.LogNormal(np.log(35.0), 0.3)); m_T = numpyro.sample("m_T", dist.Normal(1.0, 0.3)); s_T = numpyro.sample("s_T", dist.LogNormal(np.log(0.4), 0.5))
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6))
    K_0 = numpyro.sample("K_0", dist.LogNormal(np.log(8.0), 0.5)); c_0 = numpyro.sample("c_0", dist.Normal(0.0, 10.0)); s_0 = numpyro.sample("s_0", dist.LogNormal(np.log(15.0), 0.5)); B = numpyro.sample("B", dist.Normal(11.0, 3.0))
    K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_d = numpyro.sample("sd_d", dist.HalfNormal(5.0)); sd_T = numpyro.sample("sd_T", dist.HalfNormal(5.0)); sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0))
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); r = p / L; zT = (r - m_T) / s_T; z0 = (c_0 - p) / s_0
    gT = F_T + K_T * Nd.cdf(zT); gX = x0 + x1 * (r - 1.0); A = K_0 * Nd.cdf(z0)
    slope = K_T * jnp.exp(Nd.log_prob(zT)) / (s_T * L) + x1 / L + K_0 * jnp.exp(Nd.log_prob(z0)) / s_0
    sh = jnp.abs(p) <= EPS; be = p < -EPS; u = T + X - (rl - B) - A; sc = sd_d * LSC; a_ = u / sc; b_ = (u - K_s) / sc
    ld_shelf = jax.nn.log_sigmoid(a_) + jax.nn.log_sigmoid(-b_) + jnp.log1p(-jnp.exp(-(K_s / sc)))          # log P(demand shock in [u - K_s, u]): cleared on the shelf
    ld_cont = dist.Logistic(0.0, sc).log_prob(u - K_s * be) + jnp.log(slope + 1e-12)
    ll = dist.StudentT(NU, 0.0, sd_T).log_prob(T - gT) + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + jnp.where(sh, ld_shelf, ld_cont)
    numpyro.factor("ll", ll.sum())
def curves(th, p, L):
    r = p / L; gT = th["F_T"] + th["K_T"] * norm.cdf((r - th["m_T"]) / th["s_T"]); gX = th["x0"] + th["x1"] * (r - 1.0); A = th["K_0"] * norm.cdf((th["c_0"] - p) / th["s_0"]); return gT, gX, A
def clear(th, rl, L, d=0.0, eT=0.0, eX=0.0):
    """market-clearing price for given shocks. Excess supply rises in the price and jumps by K_s at zero (the shelf): above zero nothing is withdrawn,
    below zero all of K_s; if excess supply changes sign across the jump the market clears on the shelf at a price of zero."""
    def ex(p, wd): gT, gX, A = curves(th, p, L); return gT + eT + gX + eX - (rl - th["B"] + A + wd + d)
    up = ex(np.full(np.shape(rl), EPS), 0.0) < 0; dn = ex(np.full(np.shape(rl), -EPS), th["K_s"]) > 0; lo = np.where(up, EPS, -3000.0); hi = np.where(up, 20000.0, -EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    p = np.where(up | dn, 0.5 * (lo + hi), 0.0); gT, gX, A = curves(th, p, L); return p, gT + eT, gX + eX
def simulate(th, rl, L, rng):
    n = len(rl); d = rng.logistic(0.0, th["sd_d"] * LSC, n); eT = tdist.rvs(NU, size=n, random_state=rng) * th["sd_T"]; eX = tdist.rvs(NU, size=n, random_state=rng) * th["sd_X"]; return clear(th, rl, L, d, eT, eX)
def fit(rl, L, p, T, X, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(rl), jnp.asarray(L), p=jnp.asarray(p), T=jnp.asarray(T), X=jnp.asarray(X), extra_fields=("diverging",))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
if __name__ == "__main__":
    mode = sys.argv[1]; tr = H[TR]; rng = np.random.default_rng(0)
    if mode == "prior":
        pr = Predictive(model, num_samples=200)(jax.random.PRNGKey(1), jnp.asarray(tr.rlreal.values[:5]), jnp.asarray(tr.L.values[:5])); pr = {k: np.asarray(v) for k, v in pr.items()}; sub = tr.iloc[rng.choice(len(tr), 3000, replace=False)]; sims = []
        for i in range(200):
            th = {k: float(pr[k][i]) for k in PN}; p, T, X = simulate(th, sub.rlreal.values, sub.L.values, rng); sims.append(dict(th=th, p=p.astype(np.float32), T=T.astype(np.float32), X=X.astype(np.float32)))
        np.savez_compressed(f"{SP}/out/market0_prior.npz", rl=sub.rlreal.values, L=sub.L.values, p_obs=sub.p.values, T_obs=sub["T"].values, X_obs=sub.X.values, P=np.array([s_["p"] for s_ in sims]), T=np.array([s_["T"] for s_ in sims]), X=np.array([s_["X"] for s_ in sims]), **{k: pr[k] for k in PN})
        P = np.array([s_["p"] for s_ in sims]); print("prior predictive, 200 simulated markets on 3000 real hours of load and cost: median price / cost by market, 5-50-95%: " + " ".join(f"{v:.2f}" for v in np.percentile(np.median(P / sub.L.values[None], 1), [5, 50, 95])) + f" (observed {np.median(sub.p / sub.L):.2f}); share of hours at or below zero: " + " ".join(f"{v:.2f}" for v in np.percentile((P <= 0).mean(1), [5, 50, 95])) + f" (observed {(sub.p <= 0).mean():.2f}); share exactly on the shelf: " + " ".join(f"{v:.3f}" for v in np.percentile((P == 0).mean(1), [5, 50, 95])) + f" (observed {(np.abs(sub.p) <= EPS).mean():.3f}); mean thermal output GW: " + " ".join(f"{v:.0f}" for v in np.percentile(np.array([s_['T'] for s_ in sims]).mean(1), [5, 50, 95])) + f" (observed {sub['T'].mean():.0f}); highest price ct: " + " ".join(f"{v:.0f}" for v in np.percentile(P.max(1) / 10, [5, 50, 95])))
    if mode == "recover":
        TRUE = dict(F_T=5.0, K_T=30.0, m_T=1.1, s_T=0.35, x0=1.0, x1=10.0, K_0=6.0, c_0=10.0, s_0=12.0, K_s=3.0, B=10.0, sd_d=2.5, sd_T=2.0, sd_X=4.0); sub = tr.iloc[::2].copy(); miss = rng.random(len(sub)) < 0.05
        for g0 in rng.integers(0, len(sub) - 200, 8): miss[g0:g0 + rng.integers(24, 120)] = True
        sub = sub[~miss]; p, T, X = simulate(TRUE, sub.rlreal.values, sub.L.values, rng); s, rh, ess, div = fit(sub.rlreal.values, sub.L.values, p, T, X)
        print(f"simulate and recover: {len(sub)} hours (every second hour, 5% missing at random and eight gaps of 1-5 days); worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); print(f"   {k:5s} true {TRUE[k]:7.2f}  recovered {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})  {'ok' if q[0] <= TRUE[k] <= q[2] else 'OUTSIDE the 90% band'}")
        np.savez_compressed(f"{SP}/out/market0_recover.npz", **{f"true_{k}": TRUE[k] for k in PN}, **{k: s[k] for k in PN}, rhat=rh, ess=ess)
    if mode == "fit":
        s, rh, ess, div = fit(tr.rlreal.values, tr.L.values, tr.p.values, tr["T"].values, tr.X.values); th = {k: float(np.median(s[k])) for k in PN}
        print(f"M0 fitted on {len(tr)} hours (Mar 2024 - Nov 2025): worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['B'])}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); print(f"   {k:5s} {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})")
        np.savez_compressed(f"{SP}/out/market0_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div)
        gT, gX, A = curves(th, H.p.values, H.L.values); H["eT"] = H["T"] - gT; H["eX"] = H.X - gX; H["d"] = H["T"] + H.X - (H.rlreal - th["B"]) - A - th["K_s"] * (H.p.values < -EPS) - 0.5 * th["K_s"] * (np.abs(H.p.values) <= EPS); pz, Tz, Xz = clear(th, H.rlreal.values, H.L.values); H["p0"] = pz; H["T0"] = Tz; H["X0"] = Xz; H["tr"] = TR; H.to_parquet(f"{SP}/out/market0_rows.parquet")
        for lab, m in (("fitting period", TR), ("held-out year", HO)):
            g = H[m]; print(f"   {lab}: price with all shocks at zero against observed: MAE {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} ct, median |price / cost error| {np.median(np.abs(g.p0 - g.p) / g.L):.3f}; thermal output MAE {np.mean(np.abs(g.T0 - g['T'])):.2f} GW; net imports MAE {np.mean(np.abs(g.X0 - g.X)):.2f} GW; shock sizes (sd) d {g.d.std():.2f}, e_T {g.eT.std():.2f}, e_X {g.eX.std():.2f} GW")
