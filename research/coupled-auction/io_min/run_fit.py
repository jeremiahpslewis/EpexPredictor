"""
Fit io_min by NUTS. Usage: run_fit.py PANEL_DIR OUT.npz MODE [PRIOR] [WARM SAMPLES CHAINS]
MODE = recover (synthetic data from THETA_TRUE on the real exogenous series) | real.
PRIOR = base | tight_h | loose_h | alpha_lo | alpha_hi | (see VARIANTS).
"""
import sys, json, time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as Mo

PANEL, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
VAR = sys.argv[4] if len(sys.argv) > 4 else "base"
WARM, NS, NC = (int(a) for a in (sys.argv[5:8] if len(sys.argv) > 7 else (500, 500, 4)))
numpyro.set_host_device_count(4)

THETA_TRUE = dict(alpha_L=0.90, alpha_C=0.75, alpha_G=0.82, h_L=2.6, a_L=12.0, h_C=2.45, a_C=30.0, h_G=2.05,
                  sigma_L=0.10, sigma_C=0.15, sigma_G=0.18, F_L=4.0, F_C=1.0, phi0=1.2, phi1=0.3,
                  A_N=50.0, B_N=80.0, a_N=0.0, b_N=1.0, sigma_N=0.3, I=16.0, E=21.0, delta=5.0, s_m=15.0)
SHOCK_TRUE = dict(mu=np.array([-2.6, 0.0, 0.0]), rho=np.array([0.9, 0.9, 0.8]), sd=np.array([1.5, 3.0, 1.5]), corr=0.3, nu=5.0,
                  rho_q=0.95, sd_q=0.3, beta_q=0.1, nu_q=5.0)

# Prior variants for the confounded pair (decided after the base posterior; see EXPECTATIONS.md E3)
P = dict(Mo.BASE_PRIOR)
VARIANTS = {
    "base": {},
    "tight_h": dict(h_G=("normal", 2.0, 0.05), sigma_G=("lognormal", np.log(0.15), 0.2)),
    "loose_h": dict(h_G=("normal", 2.0, 0.6), sigma_G=("lognormal", np.log(0.15), 1.2)),
    "alpha_lo": dict(alpha_G=("beta", 30.0, 20.0)),          # mean 0.60, sd 0.07
    "alpha_hi": dict(alpha_G=("beta", 57.0, 3.0)),           # mean 0.95, sd 0.03
    "sm_tight": dict(s_m=("lognormal", np.log(8.0), 0.15)),
    "sm_wide": dict(s_m=("lognormal", np.log(30.0), 0.15)),
}
P.update(VARIANTS[VAR])

Pn = pd.read_parquet(f"{PANEL}/panel.parquet")
x, obs, brk = Mo.to_inputs(Pn)
n = len(Pn)

if MODE == "recover":
    # theta_true = MAP on the real data (map_fit.py), so recovery is checked where the real posterior sits
    mp = json.load(open(f"{PANEL}/map.json"))
    THETA_TRUE = {k: float(mp[k]) for k in Mo.THETA_KEYS}
    Lm = np.array(mp["Lcorr"]); C = Lm @ Lm.T
    SHOCK_TRUE.update(mu=np.array(mp["mu"]), rho=np.array(mp["rho"]), sd=np.array(mp["sd_u"]), nu=float(mp["nu"]),
                      sd_q=np.array(mp["sd_q"]), beta_q=np.array(mp["beta_q"]), nu_q=float(mp["nu_q"]))
    rng = np.random.default_rng(7); th = {k: jnp.asarray(v) for k, v in THETA_TRUE.items()}; s = SHOCK_TRUE
    Lc = np.linalg.cholesky(C)
    e = (rng.standard_normal((n, 3)) @ Lc.T) * s["sd"] / np.sqrt(rng.chisquare(s["nu"], (n, 1)) / s["nu"])
    e = np.clip(e, -5 * s["sd"], 5 * s["sd"])
    u = np.zeros((n, 3)); u[0] = e[0] / np.sqrt(1 - s["rho"] ** 2)
    for t in range(1, n): u[t] = s["rho"] * u[t - 1] + e[t]
    u = u + s["mu"]
    cl = Mo.clear(th, x, jnp.asarray(u))
    obs = dict(pDE=cl["pDE"], pN=cl["pN"], m=cl["m"])
    for i, f in enumerate(Mo.FUELS):
        ee = s["sd_q"][i] * np.sqrt(1 - s["rho_q"] ** 2) * rng.standard_t(s["nu_q"], n); nu_ = np.zeros(n)
        for t in range(1, n): nu_[t] = s["rho_q"] * nu_[t - 1] + ee[t]    # persistent, as in the data (model misspecified)
        obs[f"q_{f}"] = cl[f] + s["beta_q"][i] * x["gap"] + jnp.asarray(nu_)
    print("synthetic pDE mean/sd/min/max", float(obs["pDE"].mean()), float(obs["pDE"].std()), float(obs["pDE"].min()), float(obs["pDE"].max()))
    print("observed  pDE mean/sd", float(Pn.p.mean()), float(Pn.p.std()), "| synthetic m mean/sd", float(obs["m"].mean()), float(obs["m"].std()))
    ok = np.asarray((obs["pDE"] > Mo.PMIN + 1) & (obs["pDE"] < Mo.PMAX - 1) & (obs["pN"] > Mo.PMIN + 1) & (obs["pN"] < Mo.PMAX - 1))
    print(f"hours at a price bound dropped: {int((~ok).sum())} of {n}")
    pos = np.flatnonzero(ok); gapnext = np.r_[True, np.diff(pos) != 1] | np.asarray(brk)[pos]
    x = {k: v[pos] for k, v in x.items()}; obs = {k: v[pos] for k, v in obs.items()}; brk = jnp.asarray(gapnext); u = u[pos]
    u_back, _, _ = Mo.invert(th, x, obs["pDE"], obs["pN"], obs["m"])
    print("inversion check max|u - u_back|:", float(jnp.abs(u_back - u).max()))

t0 = time.time()
mc = MCMC(NUTS(Mo.model, init_strategy=init_to_median(num_samples=15), target_accept_prob=0.85, max_tree_depth=8),
          num_warmup=WARM, num_samples=NS, num_chains=NC, chain_method="parallel", progress_bar=False)
mc.run(jax.random.PRNGKey(1), x, obs, brk, P)
S = {k: np.asarray(v) for k, v in mc.get_samples().items()}
ex = mc.get_extra_fields(); div = int(np.asarray(ex["diverging"]).sum())
summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
rh = {k: float(np.max(v["r_hat"])) for k, v in summ.items()}
print(f"time {time.time() - t0:.0f}s divergences {div} of {NS * NC}, max r-hat {max(rh.values()):.3f} ({max(rh, key=rh.get)})")
rows = []
for k in Mo.THETA_KEYS + ["mu", "rho", "sd_u", "nu", "sd_q", "beta_q"]:
    v = S[k].reshape(len(S[k]), -1)
    for j in range(v.shape[1]):
        q = np.quantile(v[:, j], [0.05, 0.5, 0.95]); r = dict(par=k + (f"[{j}]" if v.shape[1] > 1 else ""), q05=q[0], med=q[1], q95=q[2], sd=v[:, j].std())
        if MODE == "recover" and k in THETA_TRUE: r["true"] = THETA_TRUE[k]; r["inside"] = bool(q[0] <= THETA_TRUE[k] <= q[2])
        rows.append(r)
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False))
if MODE == "recover":
    ins = R.dropna(subset=["true"]); print(f"inside 90%: {int(ins.inside.sum())} of {len(ins)}")
np.savez(OUT, **S, divergences=div, max_rhat=max(rh.values()), variant=VAR, mode=MODE)
R.to_csv(OUT.replace(".npz", ".csv"), index=False)
