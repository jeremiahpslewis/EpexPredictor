"""Level 1 with the CO2 weight free (notes 20.7): price = (1.85 gas + w carbon) * k(residual load). Full sampling on the fitting period; pictures."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import oc
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from zoom01 import ix, P, RL, L, ok, y, tr, ho, K
SP = oc.SP; numpyro.set_host_device_count(4); GASC = 1.85 * oc.GAS; CO2 = np.asarray(oc.EUA, float)
def model(rl, gasc, co2, L0, y=None):
    k0 = numpyro.sample("k0", dist.Normal(0.0, 0.15)); inc = numpyro.sample("inc", dist.HalfNormal(0.35).expand([len(K) - 1])); kn = numpyro.deterministic("kn", k0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(inc)]))
    w = numpyro.sample("w", dist.LogNormal(np.log(0.37), 0.7)); sigma = numpyro.sample("sigma", dist.HalfNormal(0.3)); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4))
    numpyro.sample("y", dist.StudentT(nu, jnp.interp(rl, K, kn) * (gasc + w * co2) / L0, sigma), obs=y)
if __name__ == "__main__":
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=700, num_samples=600, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(0), jnp.asarray(RL[tr]), jnp.asarray(GASC[tr]), jnp.asarray(CO2[tr]), jnp.asarray(L[tr]), y=jnp.asarray(y[tr]))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values())
    w = float(np.median(s["w"])); km = np.median(s["kn"], 0); Z1 = np.load(f"{SP}/out/zoom1_post.npz"); k1 = np.median(Z1["kn"], 0)
    print(f"sampling: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}; CO2 weight {w:.2f} (5-95% {np.percentile(s['w'], 5):.2f}..{np.percentile(s['w'], 95):.2f}, independent hours, so too narrow); k at the knots " + " ".join(f"{v:.2f}" for v in km) + f" (level 1: " + " ".join(f"{v:.2f}" for v in k1) + ")")
    print(f"  carbon price in the fitting period {np.nanpercentile(CO2[tr], 5):.0f}..{np.nanpercentile(CO2[tr], 95):.0f}, held-out year {np.nanpercentile(CO2[ho], 5):.0f}..{np.nanpercentile(CO2[ho], 95):.0f}; correlation of carbon with time in the fitting period {np.corrcoef(CO2[tr], np.arange(tr.sum()))[0, 1]:+.2f}, with gas {np.corrcoef(CO2[tr], oc.GAS[tr])[0, 1]:+.2f}")
    fitw = np.interp(RL, K, km) * (GASC + w * CO2) / L; r1 = y - np.interp(RL, K, k1); rw = y - fitw
    for lab, m in (("fitting period", tr), ("held-out year", ho)): print(f"  {lab}: mean residual weight 0.37: {np.mean(r1[m]):+.3f} -> fitted weight: {np.mean(rw[m]):+.3f}; median |residual| {np.median(np.abs(r1[m])):.3f} -> {np.median(np.abs(rw[m])):.3f}")
    np.savez_compressed(f"{SP}/out/zoom1c_post.npz", K=K, kn=s["kn"], w=s["w"], sigma=s["sigma"], nu=s["nu"], rhat=rh, ess=ess)
    pd.DataFrame({"y": y, "r1": r1, "rw": rw, "rl": RL, "L": L, "gas": oc.GAS, "co2": CO2, "p": P, "tr": tr, "ho": ho, "day": oc.DAY}, index=ix).to_parquet(f"{SP}/out/zoom1c_rows.parquet")
