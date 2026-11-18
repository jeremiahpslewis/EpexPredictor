"""Which mode does the data prefer? Short chains of M0h from several seeds; per chain: medians of a, c_z, K_z, K_T, s_T and the mean log density."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import market0h as Mh, jax, jax.numpy as jnp, numpyro
from numpyro.infer import MCMC, NUTS
A = Mh.grid(Mh.B.H[np.asarray(Mh.B.H.index < "2025-12-01")])
for seed in (int(x) for x in sys.argv[1:]):
    mc = MCMC(NUTS(Mh.model, target_accept_prob=0.9), num_warmup=400, num_samples=150, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(A["D"]), jnp.asarray(A["L"]), jnp.asarray(A["v"]), p=jnp.asarray(A["p"]), T=jnp.asarray(A["T"]), X=jnp.asarray(A["X"]), S=jnp.asarray(A["S"]), extra_fields=("potential_energy",))
    s = mc.get_samples(group_by_chain=True); pe = np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"])
    for c in range(4): print(f"seed {seed} chain {c}: log density {-pe[c].mean():.0f} | " + " ".join(f"{k} {float(np.median(np.asarray(s[k])[c])):.2f}" for k in ("a", "c_z", "s_z", "K_z", "K_x", "K_T", "s_T", "F_T", "sd_T")))
    np.savez_compressed(f"{Mh.SP}/out/market0h_modes_{seed}.npz", pe=pe, **{k: np.asarray(v) for k, v in s.items()})
