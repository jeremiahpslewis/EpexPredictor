"""Plain curve (level 1) refitted on the walk-forward schedule (notes 21.19). usage: l1_refit.py END TAG"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import zoom01 as Z, jax, jax.numpy as jnp, numpyro
from numpyro.infer import MCMC, NUTS
END, TAG = sys.argv[1], sys.argv[2]; tr = Z.ok & (Z.ix < END); mc = MCMC(NUTS(Z.model, target_accept_prob=0.9), num_warmup=300, num_samples=300, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(0), jnp.asarray(Z.RL[tr]), y=jnp.asarray(Z.y[tr]))
s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values())
np.savez_compressed(f"{Z.SP}/out/zoom1{TAG}_post.npz", K=Z.K, kn=s["kn"], rhat=rh); print(f"plain curve refit {TAG}: data to {END}, {int(tr.sum())} hours, r-hat {rh:.3f}")
