"""
Pooled eleven-zone fit. Usage: fit11.py RECS.pkl BASIN_POST.npz OUT.npz WARM SAMPLES CHAINS
BASIN_POST: a two-zone basin fit (fit_mode0.npz or fit_mode1.npz); its medians start the German block.

Likelihood per MTU (joint change of variables on the regime's support, no regime conditioning):
    u_z = S_z(p_z) - (D_z + X_z + n_z) / 1000          (GW)
    log f = log MVT_11(u - mu; nu, Sigma) + log det(Q' diag(S'(p)) Q)
German supply: the fuel block of model.py (price-taking, logistic heat rates, smoothed floors). Neighbours:
    S_z(p) = sbar_z [alpha_z + beta_z G((p - a_z - b_z Lc) / (sigma_z Lc))]
with sbar_z the zone's mean implied supply in the fit sample, and (alpha, log beta, a, b, log sigma) partially pooled
across the ten neighbours. German fuel outputs: Student-t measurement equation, log-likelihood tempered by 1/KQ.
MTUs with a missing link schedule are excluded.
"""
import sys, pickle, time
import numpy as np, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import model as Mo, net11 as N
jax.config.update("jax_enable_x64", True)
RECS, BASIN, OUT = sys.argv[1:4]; WARM, NS, NC = map(int, sys.argv[4:7]); numpyro.set_host_device_count(NC)
KQ = 4.0
d = pickle.load(open(RECS, "rb"))["recs"]
d = [r for r in d if r["kind"] == "fit" and not any(v["missing"] for v in r["links"].values())]
T = len(d); print("fit MTUs:", T, " augmented:", sum(r["aug"] is not None for r in d), " generic:", sum("generic" in (r["aug"] or {}).get("added", []) for r in d))
arr = lambda k: jnp.asarray(np.array([r[k] for r in d], dtype=float))
p, n, X, D = arr("p"), arr("n"), arr("X"), arr("D")
Qp = np.zeros((T, N.NZ, N.NZ)); pad = np.ones((T, N.NZ))
for i, r in enumerate(d): k = r["Q"].shape[1]; Qp[i, :, :k] = r["Q"]; pad[i, :k] = 0.0
Qp, pad = jnp.asarray(Qp), jnp.asarray(pad)
de = {k: jnp.asarray(np.array([r["de"][k] for r in d])) for k in d[0]["de"]}
xde = dict(Pg=de["Pg"], tau=de["tau"], hdd=de["hdd"], K_L=de["K_L"], K_C=de["K_C"], K_G=de["K_G"])
Lc = arr("Lc")
need = (D + X + n) / 1000.0                                                   # GW each zone must supply
sbar = jnp.maximum(need.mean(0), 1.0)
DE_KEYS = ["alpha_L", "alpha_C", "alpha_G", "h_L", "a_L", "h_C", "a_C", "h_G", "sigma_L", "sigma_C", "sigma_G", "F_L", "F_C", "phi0", "phi1"]
NBz = N.NZ - 1

def model():
    th = {k: Mo.draw(k, Mo.BASE_PRIOR[k]) for k in DE_KEYS}
    S, dS = Mo.fuel_supply(th, xde, p[:, 0])
    S_de = S["L"] + S["C"] + S["G"]; dS_de = dS["L"] + dS["C"] + dS["G"]
    # pooled neighbours
    m_al = numpyro.sample("m_al", dist.Normal(0.4, 0.3)); t_al = numpyro.sample("t_al", dist.HalfNormal(0.3))
    m_lb = numpyro.sample("m_lb", dist.Normal(0.0, 0.5)); t_lb = numpyro.sample("t_lb", dist.HalfNormal(0.3))
    m_a = numpyro.sample("m_a", dist.Normal(0.0, 40.0)); t_a = numpyro.sample("t_a", dist.HalfNormal(20.0))
    m_b = numpyro.sample("m_b", dist.Normal(1.0, 0.4)); t_b = numpyro.sample("t_b", dist.HalfNormal(0.3))
    m_ls = numpyro.sample("m_ls", dist.Normal(np.log(0.4), 0.5)); t_ls = numpyro.sample("t_ls", dist.HalfNormal(0.5))
    zz = numpyro.sample("zz", dist.Normal(0, 1).expand([5, NBz]))
    al = numpyro.deterministic("al", m_al + t_al * zz[0]); be = numpyro.deterministic("be", jnp.exp(m_lb + t_lb * zz[1]))
    aa = numpyro.deterministic("aa", m_a + t_a * zz[2]); bb = numpyro.deterministic("bb", m_b + t_b * zz[3])
    sg = numpyro.deterministic("sg", jnp.exp(m_ls + t_ls * zz[4]))
    sc = sg[None, :] * Lc[:, None]; zarg = (p[:, 1:] - aa[None, :] - bb[None, :] * Lc[:, None]) / sc
    g = jax.nn.sigmoid(zarg); S_nb = sbar[1:] * (al + be * g); dS_nb = sbar[1:] * be * g * (1 - g) / sc
    S_all = jnp.concatenate([S_de[:, None], S_nb], 1); dS_all = jnp.concatenate([dS_de[:, None], dS_nb], 1)
    u = S_all - need
    mu_de = numpyro.sample("mu_de", dist.Normal(-2.0, 5.0)); mu = jnp.zeros(N.NZ).at[0].set(mu_de)
    sd = numpyro.sample("sd_u", dist.HalfNormal(5.0).expand([N.NZ])); Lk = numpyro.sample("Lcorr", dist.LKJCholesky(N.NZ, 4.0))
    nu = numpyro.sample("nu", dist.Gamma(4.0, 0.8))
    Mt = jnp.einsum("tzk,tz,tzl->tkl", Qp, dS_all, Qp) + jax.vmap(jnp.diag)(pad)
    numpyro.factor("lik", dist.MultivariateStudentT(nu, jnp.zeros(N.NZ), sd[:, None] * Lk).log_prob(u - mu).sum() + jnp.linalg.slogdet(Mt)[1].sum())
    sq = numpyro.sample("sd_q", dist.HalfNormal(2.0).expand([3])); bq = numpyro.sample("beta_q", dist.Normal(0, 0.3).expand([3]))
    nuq = numpyro.sample("nu_q", dist.Gamma(4.0, 0.8))
    for i, (f, col) in enumerate((("L", "lig"), ("C", "coal"), ("G", "gasg"))):
        numpyro.factor(f"q_{f}", dist.StudentT(nuq, 0.0, sq[i]).log_prob(de[col] - S[f] - bq[i] * de["gap"]).sum() / KQ)

B = dict(np.load(BASIN)); init = {k: np.median(B[k]) for k in DE_KEYS}
init.update(m_al=0.4, t_al=0.1, m_lb=0.0, t_lb=0.1, m_a=0.0, t_a=10.0, m_b=1.0, t_b=0.1, m_ls=np.log(0.4), t_ls=0.2,
            zz=np.zeros((5, NBz)), mu_de=float(np.median(B["mu"][:, 0])) if "mu" in B else -2.0, sd_u=np.full(N.NZ, 1.5), nu=5.0,
            sd_q=np.full(3, 1.5), beta_q=np.zeros(3), nu_q=5.0)
t0 = time.time()
mc = MCMC(NUTS(model, init_strategy=init_to_value(values={k: jnp.asarray(v) for k, v in init.items()}), target_accept_prob=0.85, max_tree_depth=8),
          num_warmup=WARM, num_samples=NS, num_chains=NC, chain_method="parallel", progress_bar=False)
mc.run(jax.random.PRNGKey(3))
G = mc.get_samples(group_by_chain=True); summ = numpyro.diagnostics.summary(G)
rh = {k: float(np.nanmax(s["r_hat"])) for k, s in summ.items()}
print(f"{time.time()-t0:.0f}s divergences {int(np.asarray(mc.get_extra_fields()['diverging']).sum())} max r-hat {max(rh.values()):.3f} ({max(rh, key=rh.get)})")
for c in range(NC): print(f"  chain {c}:", {k: round(float(np.median(np.asarray(G[k][c]))), 3) for k in ["F_L", "F_C", "phi0", "sigma_L", "sigma_C", "sigma_G", "alpha_G", "h_G", "mu_de", "m_a", "m_ls"]})
Sm = mc.get_samples()
print("neighbours (median): al", np.round(np.median(np.asarray(Sm["al"]), 0), 2).tolist(), "\n  be", np.round(np.median(np.asarray(Sm["be"]), 0), 2).tolist(),
      "\n  aa", np.round(np.median(np.asarray(Sm["aa"]), 0), 1).tolist(), "\n  bb", np.round(np.median(np.asarray(Sm["bb"]), 0), 2).tolist(), "\n  sg", np.round(np.median(np.asarray(Sm["sg"]), 0), 2).tolist())
np.savez(OUT, **{k: np.asarray(v) for k, v in Sm.items()}, sbar=np.asarray(sbar), de_keys=np.array(DE_KEYS))
