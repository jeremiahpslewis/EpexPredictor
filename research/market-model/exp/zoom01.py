"""Zoom levels 0 and 1 (notes section 20): price = c * L, then price = L * k(residual load). Full sampling, pictures, residuals against what is left out."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import oc
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
SP = oc.SP; numpyro.set_host_device_count(4)
ix = pd.DatetimeIndex(oc.S.idx[: oc.n15 // 4 * 4: 4]).tz_convert("Europe/Berlin"); P = oc.P; RL = oc.RL[:, 0]; L = 1.85 * oc.GAS + 0.37 * oc.EUA; NBS = np.nansum(oc.RL[:, 1:] - oc.RLZM[None], 1); TE = oc.TEFF
ok = np.isfinite(P) & np.isfinite(RL) & np.isfinite(L) & (L > 20); y = P / L; tr = ok & (ix < "2025-12-01"); ho = ok & (ix >= "2025-12-01")
K = np.arange(-10.0, 80.1, 10.0)
def model(rl, y=None):
    k0 = numpyro.sample("k0", dist.Normal(0.0, 0.15)); inc = numpyro.sample("inc", dist.HalfNormal(0.35).expand([len(K) - 1])); kn = numpyro.deterministic("kn", k0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(inc)]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3)); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4)); numpyro.sample("y", dist.StudentT(nu, jnp.interp(rl, K, kn), sigma), obs=y)
if __name__ == "__main__":
    c0 = float(np.median(y[tr])); print(f"level 0: c = median price / gas-fired cost in the fitting period = {c0:.3f} (held-out year: {np.median(y[ho]):.3f}); hours fitting {tr.sum()}, held out {ho.sum()}")
    prior = Predictive(model, num_samples=300)(jax.random.PRNGKey(1), jnp.asarray(RL[tr][:10]))
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=600, num_samples=600, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(0), jnp.asarray(RL[tr]), y=jnp.asarray(y[tr]))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum()) if "diverging" in mc.get_extra_fields() else -1
    print(f"level 1 sampling: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['k0'])} draws; k at the knots (median): " + " ".join(f"{v:.2f}" for v in np.median(s["kn"], 0)) + f"; noise scale {np.median(s['sigma']):.3f}, nu {np.median(s['nu']):.1f}")
    np.savez_compressed(f"{SP}/out/zoom1_post.npz", K=K, kn=s["kn"], sigma=s["sigma"], nu=s["nu"], k0=s["k0"], inc=s["inc"], prior_kn=np.asarray(prior["kn"]), c0=c0, rhat=rh, ess=ess)
    km = np.median(s["kn"], 0); fit = np.interp(RL, K, km); res = y - fit
    for lab, m in (("fitting period", tr), ("held-out year", ho)):
        print(f"  {lab}: median |price/L - k| {np.median(np.abs(res[m])):.3f}; mean residual {np.mean(res[m]):+.3f}; in ct at that period's cost: MAE level 0 {np.mean(np.abs(P[m] - c0 * L[m])) / 10:.2f}, level 1 {np.mean(np.abs(P[m] - fit[m] * L[m])) / 10:.2f}")
    pd.DataFrame({"y": y, "fit": fit, "res": res, "rl": RL, "L": L, "gas": oc.GAS, "p": P, "hour": oc.HOUR, "dt": oc.DT, "nbs": NBS, "teff": TE, "tr": tr, "ho": ho, "day": oc.DAY}, index=ix).to_parquet(f"{SP}/out/zoom1_rows.parquet")
