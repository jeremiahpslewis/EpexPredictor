"""Zoom level 2 (notes 20.4): price = L * k(residual load + shift[hour] + shift[day type]). Full sampling; saves draws and rows for the pictures."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import oc
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from zoom01 import ix, P, RL, L, NBS, TE, ok, y, tr, ho, K
SP = oc.SP; numpyro.set_host_device_count(4); HOUR = np.asarray(oc.HOUR, int); DT = np.asarray(oc.DT, int)
def model(rl, hour, dt, y=None):
    k0 = numpyro.sample("k0", dist.Normal(0.0, 0.15)); inc = numpyro.sample("inc", dist.HalfNormal(0.35).expand([len(K) - 1])); kn = numpyro.deterministic("kn", k0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(inc)]))
    ah_ = numpyro.sample("ah_", dist.Normal(0.0, 4.0).expand([23])); ah = numpyro.deterministic("ah", jnp.concatenate([ah_, -ah_.sum(keepdims=True)])); ad = numpyro.sample("ad", dist.Normal(0.0, 4.0).expand([2]))
    x = rl + ah[hour] + jnp.concatenate([jnp.zeros(1), ad])[dt]; sigma = numpyro.sample("sigma", dist.HalfNormal(0.3)); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4)); numpyro.sample("y", dist.StudentT(nu, jnp.interp(x, K, kn), sigma), obs=y)
if __name__ == "__main__":
    prior = Predictive(model, num_samples=300)(jax.random.PRNGKey(1), jnp.asarray(RL[tr][:10]), jnp.asarray(HOUR[tr][:10]), jnp.asarray(DT[tr][:10]))
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=700, num_samples=600, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(0), jnp.asarray(RL[tr]), jnp.asarray(HOUR[tr]), jnp.asarray(DT[tr]), y=jnp.asarray(y[tr]))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values())
    km = np.median(s["kn"], 0); ah = np.median(s["ah"], 0); ad = np.r_[0.0, np.median(s["ad"], 0)]; x = RL + ah[HOUR] + ad[DT]; fit = np.interp(x, K, km); res = y - fit
    print(f"level 2 sampling: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['k0'])}; k at the knots: " + " ".join(f"{v:.2f}" for v in km) + f"; noise {np.median(s['sigma']):.3f}, nu {np.median(s['nu']):.1f}")
    print("  hour shifts, GW: " + " ".join(f"{v:+.1f}" for v in ah) + f" | Saturday {ad[1]:+.1f}, Sunday / holiday {ad[2]:+.1f}")
    Z1 = np.load(f"{SP}/out/zoom1_post.npz"); r1 = y - np.interp(RL, K, np.median(Z1["kn"], 0))
    for lab, m in (("fitting period", tr), ("held-out year", ho)): print(f"  {lab}: median |residual| level 1 {np.median(np.abs(r1[m])):.3f} -> level 2 {np.median(np.abs(res[m])):.3f}; mean residual {np.mean(res[m]):+.3f}; MAE in ct {np.mean(np.abs(P[m] - np.interp(RL[m], K, np.median(Z1['kn'], 0)) * L[m])) / 10:.2f} -> {np.mean(np.abs(P[m] - fit[m] * L[m])) / 10:.2f}")
    np.savez_compressed(f"{SP}/out/zoom2_post.npz", K=K, kn=s["kn"], ah=s["ah"], ad=s["ad"], sigma=s["sigma"], nu=s["nu"], prior_ah=np.asarray(prior["ah"]), prior_ad=np.asarray(prior["ad"]), rhat=rh, ess=ess)
    pd.DataFrame({"y": y, "fit": fit, "res": res, "res1": r1, "x": x, "rl": RL, "L": L, "gas": oc.GAS, "p": P, "hour": HOUR, "dt": DT, "nbs": NBS, "teff": TE, "tr": tr, "ho": ho, "day": oc.DAY}, index=ix).to_parquet(f"{SP}/out/zoom2_rows.parquet")
