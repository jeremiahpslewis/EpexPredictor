"""Two-zone fit started in a chosen basin. Usage: fit_modes.py PANEL_DIR POST.npz OUT.npz INIT WARM SAMPLES CHAINS
INIT = mode0 | mode1 | hybrid  (medians of chain 0 / chain 1 of POST; hybrid = lignite and coal of mode 0, gas of mode 1)."""
import sys, json, time, numpy as np, pandas as pd, jax, numpyro
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import model as Mo
PANEL, POST, OUT, INIT = sys.argv[1:5]; WARM, NS, NC = map(int, sys.argv[5:8]); numpyro.set_host_device_count(NC)
import os
Pn = pd.read_parquet(f"{PANEL}/panel.parquet")
if os.environ.get("TRAIN_END"): Pn = Pn[Pn.index < pd.Timestamp(os.environ["TRAIN_END"], tz="UTC")]   # prereg/HOLDOUT.md
print("hours:", len(Pn), Pn.index.min(), Pn.index.max(), flush=True)
x, obs, brk = Mo.to_inputs(Pn)
S = dict(np.load(POST)); n = len(S["h_G"]) // 2
med = lambda c: {k: np.median(S[k][c * n:(c + 1) * n], axis=0) for k in S if k not in ("divergences", "max_rhat", "variant", "mode", "init")}
m0, m1 = med(0), med(1)
if INIT == "mode0": v = m0
elif INIT == "mode1": v = m1
else:
    v = dict(m0); [v.__setitem__(k, m1[k]) for k in ("alpha_G", "h_G", "sigma_G", "phi0", "phi1")]
v = {k: np.asarray(a) for k, a in v.items() if k in Mo.BASE_PRIOR or k in ("mu", "rho", "sd_u", "nu", "sd_q", "beta_q", "nu_q")}   # Lcorr starts at its prior median (LKJ init bug)
t0 = time.time()
# warmup, then sampling in blocks of BLOCK draws per chain, saving the draws so far to OUT after each block
BLOCK = int(os.environ.get("BLOCK", str(NS)))
save = lambda G: np.savez(OUT, **{k: np.asarray(a).reshape((-1,) + np.asarray(a).shape[2:]) for k, a in G.items()}, init=INIT)
mc = MCMC(NUTS(Mo.model, init_strategy=init_to_value(values=v), target_accept_prob=0.85, max_tree_depth=8),
          num_warmup=WARM, num_samples=min(BLOCK, NS), num_chains=NC, chain_method="parallel", progress_bar=False)
mc.run(jax.random.PRNGKey(hash(INIT) % 2**31), x, obs, brk); blocks = [mc.get_samples(group_by_chain=True)]
print(f"warmup + block 1 at {time.time() - t0:.0f}s", flush=True)
while sum(np.asarray(b["h_G"]).shape[1] for b in blocks) < NS:
    save({k: np.concatenate([np.asarray(b[k]) for b in blocks], 1) for k in blocks[0]})
    mc.post_warmup_state = mc.last_state; mc.run(mc.post_warmup_state.rng_key, x, obs, brk); blocks.append(mc.get_samples(group_by_chain=True))
    print(f"block {len(blocks)} at {time.time() - t0:.0f}s", flush=True)
G = {k: np.concatenate([np.asarray(b[k]) for b in blocks], 1) for k in blocks[0]}; summ = numpyro.diagnostics.summary(G)
rh = {k: float(np.max(s["r_hat"])) for k, s in summ.items()}
pe = np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"]) if "potential_energy" in mc.get_extra_fields() else None
print(f"{INIT}: {time.time()-t0:.0f}s, divergences {int(np.asarray(mc.get_extra_fields()['diverging']).sum())}, max r-hat {max(rh.values()):.3f} ({max(rh, key=rh.get)})")
for c in range(NC): print(f"  chain {c}:", {k: round(float(np.median(np.asarray(G[k][c]))), 3) for k in ["F_L", "F_C", "phi0", "sigma_L", "sigma_C", "sigma_G", "alpha_L", "alpha_C", "alpha_G", "h_G"]})
save(G)
