"""
Minimal structural model of the coupled day-ahead auction, DE_LU against an aggregate neighbour zone N (io_min).

Per hour t, given exogenous x_t and parameters theta, the auction clears (p_DE, p_N, m) from
    DE:        S_DE(p_DE) + m = D_DE + u_DE          S_DE = S_L + S_C + S_G (lignite, hard coal, gas)
    N:         S_N(p_N)       = D_N  + m + u_N
    exchange:  m = M(p_DE - p_N) + u_m               m = DE net import from N (GW)
Supply of fuel f (price-taking, offers at marginal cost; heat rates logistic across units):
    S_f(p) = F_f + (alpha_f K_f - F_f) * G((p - c_f) / (sigma_f c_f))
    c_L = a_L + h_L e_L tau,  c_C = a_C + h_C e_C tau,  c_G = h_G (P_g + e_G tau) + v_G
    F_G = phi0 + phi1 HDD_d,  F_L, F_C constant (must-run floor, GW)
N supply:   S_N(p) = A_N + B_N G((p - a_N - b_N Lc) / (sigma_N Lc)),  Lc = 1.85 P_g + 0.37 tau (gas-fired cost)
Exchange:   M(d) = -E + (I + E) G((d - delta) / s_m)
Shocks:     u_t = mu + rho * (u_{t-1} - mu) + e_t,  e_t ~ MVStudentT(nu, 0, Sigma)   (AR(1) per component)
Fuel obs:   q_f = S_f(p_DE) + beta_f gap + nu_f,  nu_f AR(1) with Student-t innovations
Likelihood: exact inversion u = u(p_DE, p_N, m; theta) with |det J| = S'_DE (S'_N + M') + S'_N M'.
"""
import jax, jax.numpy as jnp, numpy as np
import numpyro, numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
E_L, E_C, E_G, V_G = 0.364, 0.341, 0.202, 3.0       # t CO2 per MWh_th (IPCC defaults), gas variable O&M EUR/MWh
PMIN, PMAX = -500.0, 4000.0
KQ = 24.0
FUELS = ("L", "C", "G")

def sig(z): return jax.nn.sigmoid(z)
def dsig(z): s = jax.nn.sigmoid(z); return s * (1 - s)

def costs(th, x):
    cL = th["a_L"] + th["h_L"] * E_L * x["tau"]
    cC = th["a_C"] + th["h_C"] * E_C * x["tau"]
    cG = th["h_G"] * (x["Pg"] + E_G * x["tau"]) + V_G
    return {"L": cL, "C": cC, "G": cG}

def floors(th, x):
    return {"L": th["F_L"] + 0 * x["tau"], "C": th["F_C"] + 0 * x["tau"], "G": th["phi0"] + th["phi1"] * x["hdd"]}

def fuel_supply(th, x, p, outage=0.0, kind="cheapest"):
    """Supply and slope per fuel at DE price p. outage GW of gas removed (kind 'cheapest' or 'average')."""
    c, F = costs(th, x), floors(th, x); S, dS = {}, {}
    for f in FUELS:
        K = th[f"alpha_{f}"] * x[f"K_{f}"]
        if f == "G" and kind == "average": K = K - outage
        Ff = jnp.minimum(F[f], K); s = th[f"sigma_{f}"] * c[f]; z = (p - c[f]) / s
        S[f] = Ff + (K - Ff) * sig(z); dS[f] = (K - Ff) * dsig(z) / s
        if f == "G" and kind == "cheapest" and outage > 0:
            cut = jnp.minimum(outage, S[f] - Ff)               # the outage unit runs whenever gas is above its floor
            dS[f] = jnp.where(S[f] - Ff > outage, dS[f], 0.0); S[f] = S[f] - cut
    return S, dS

def n_supply(th, x, p):
    Lc = 1.85 * x["Pg"] + 0.37 * x["tau"]; s = th["sigma_N"] * Lc; z = (p - th["a_N"] - th["b_N"] * Lc) / s
    return th["A_N"] + th["B_N"] * sig(z), th["B_N"] * dsig(z) / s

def exch(th, d):
    z = (d - th["delta"]) / th["s_m"]
    return -th["E"] + (th["I"] + th["E"]) * sig(z), (th["I"] + th["E"]) * dsig(z) / th["s_m"]

def invert(th, x, pDE, pN, m):
    S, dS = fuel_supply(th, x, pDE); SN, dSN = n_supply(th, x, pN); M, dM = exch(th, pDE - pN)
    SDE = S["L"] + S["C"] + S["G"]; dSDE = dS["L"] + dS["C"] + dS["G"]
    u = jnp.stack([SDE + m - x["DDE"], SN - m - x["DN"], m - M], -1)
    logdet = jnp.log(dSDE * (dSN + dM) + dSN * dM + 1e-12)
    return u, logdet, S

def clear(th, x, u, outage=0.0, kind="cheapest", dD=0.0, iters=70):
    """Re-clear (p_DE, p_N, m) for given shocks u (n,3). Nested bisection; both conditions are monotone."""
    def pN_given(pDE):
        lo, hi = jnp.full_like(pDE, PMIN), jnp.full_like(pDE, PMAX)
        def body(_, lh):
            lo, hi = lh; mid = 0.5 * (lo + hi)
            g = n_supply(th, x, mid)[0] - x["DN"] - u[:, 1] - exch(th, pDE - mid)[0] - u[:, 2]
            return jnp.where(g > 0, lo, mid), jnp.where(g > 0, mid, hi)
        lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi)); return 0.5 * (lo + hi)
    def f(pDE):
        pN = pN_given(pDE); S, _ = fuel_supply(th, x, pDE, outage, kind)
        return S["L"] + S["C"] + S["G"] + exch(th, pDE - pN)[0] + u[:, 2] - x["DDE"] - dD - u[:, 0]
    lo, hi = jnp.full(u.shape[0], PMIN), jnp.full(u.shape[0], PMAX)
    def body(_, lh):
        lo, hi = lh; mid = 0.5 * (lo + hi); g = f(mid)
        return jnp.where(g > 0, lo, mid), jnp.where(g > 0, mid, hi)
    lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi)); pDE = 0.5 * (lo + hi)
    pN = pN_given(pDE); m = exch(th, pDE - pN)[0] + u[:, 2]
    S, _ = fuel_supply(th, x, pDE, outage, kind)
    return dict(pDE=pDE, pN=pN, m=m, L=S["L"], C=S["C"], G=S["G"])

# ---------------------------------------------------------------- priors (PRIOR dict lets runs swap them)
BASE_PRIOR = dict(
    alpha_L=("beta", 17.0, 3.0), alpha_C=("beta", 17.0, 3.0), alpha_G=("beta", 17.0, 3.0),   # mean 0.85, sd 0.08
    h_L=("normal", 2.6, 0.15), a_L=("normal", 15.0, 6.0),
    h_C=("normal", 2.5, 0.15), a_C=("normal", 35.0, 10.0),
    h_G=("normal", 2.0, 0.2),
    sigma_L=("lognormal", np.log(0.15), 0.6), sigma_C=("lognormal", np.log(0.15), 0.6), sigma_G=("lognormal", np.log(0.15), 0.6),
    F_L=("lognormal", np.log(4.0), 0.5), F_C=("lognormal", np.log(1.0), 0.7), phi0=("lognormal", np.log(1.2), 0.5),
    phi1=("lognormal", np.log(0.3), 0.6),
    A_N=("normal", 50.0, 20.0), B_N=("normal", 80.0, 30.0), a_N=("normal", 0.0, 30.0), b_N=("normal", 1.0, 0.3),
    sigma_N=("lognormal", np.log(0.3), 0.6),
    I=("normal", 16.0, 3.0), E=("normal", 21.0, 3.0), delta=("normal", 0.0, 10.0), s_m=("lognormal", np.log(15.0), 0.6),
)

def draw(name, spec):
    k = spec[0]
    if k == "beta": return numpyro.sample(name, dist.Beta(spec[1], spec[2]))
    if k == "normal": return numpyro.sample(name, dist.Normal(spec[1], spec[2]))
    if k == "lognormal": return numpyro.sample(name, dist.LogNormal(spec[1], spec[2]))
    if k == "fixed": return numpyro.deterministic(name, jnp.asarray(spec[1]))
    raise ValueError(k)

def ar1_t_logp(r, rho, scale, nu, brk):
    """log density of an AR(1) series r with Student-t innovations; brk marks the first hour after a gap."""
    e = r[1:] - rho * r[:-1]; e = jnp.where(brk[1:], r[1:] * jnp.sqrt(1 - rho ** 2), e)
    return dist.StudentT(nu, 0.0, scale).log_prob(e).sum()

def model(x, obs, brk, prior=BASE_PRIOR):
    th = {k: draw(k, v) for k, v in prior.items()}
    u, logdet, S = invert(th, x, obs["pDE"], obs["pN"], obs["m"])
    mu = numpyro.sample("mu", dist.Normal(jnp.array([-2.0, 0.0, 0.0]), jnp.array([5.0, 10.0, 5.0])))
    rho = numpyro.sample("rho", dist.Uniform(0.0, 0.995).expand([3]))
    sd = numpyro.sample("sd_u", dist.HalfNormal(jnp.array([3.0, 6.0, 3.0])))
    Lc = numpyro.sample("Lcorr", dist.LKJCholesky(3, 2.0)); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.8))
    r = u - mu; e = r[1:] - rho * r[:-1]; e = jnp.where(brk[1:, None], r[1:] * jnp.sqrt(1 - rho ** 2), e)
    numpyro.factor("shocks", dist.MultivariateStudentT(nu, jnp.zeros(3), sd[:, None] * Lc).log_prob(e).sum() + logdet[1:].sum())
    # fuel outputs: realised generation = curve at the DE price + after-auction gap + error. The errors are persistent
    # (unit commitment, not in the model); an AR(1) there makes curve levels nearly free, so instead iid Student-t with
    # the log-likelihood tempered by 1/KQ (a power likelihood: about one effective observation per day).
    sq = numpyro.sample("sd_q", dist.HalfNormal(2.0).expand([3]))
    bq = numpyro.sample("beta_q", dist.Normal(0.0, 0.3).expand([3])); nuq = numpyro.sample("nu_q", dist.Gamma(4.0, 0.8))
    for i, f in enumerate(FUELS):
        numpyro.factor(f"q_{f}", dist.StudentT(nuq, 0.0, sq[i]).log_prob(obs[f"q_{f}"] - S[f] - bq[i] * x["gap"]).sum() / KQ)

XCOLS = ["DDE", "DN", "Pg", "tau", "hdd", "K_lignite", "K_coal", "K_gas", "gap"]

def to_inputs(P):
    x = {c: jnp.asarray(P[c].to_numpy(float)) for c in XCOLS}
    x["K_L"], x["K_C"], x["K_G"] = x.pop("K_lignite"), x.pop("K_coal"), x.pop("K_gas")
    obs = dict(pDE=jnp.asarray(P.p.to_numpy(float)), pN=jnp.asarray(P.pN.to_numpy(float)), m=jnp.asarray(P.m.to_numpy(float)),
               q_L=jnp.asarray(P.lig.to_numpy(float)), q_C=jnp.asarray(P.coal.to_numpy(float)), q_G=jnp.asarray(P.gasg.to_numpy(float)))
    dt = P.index.to_series().diff().dt.total_seconds().to_numpy()
    brk = jnp.asarray(np.r_[True, dt[1:] != 3600])
    return x, obs, brk

THETA_KEYS = list(BASE_PRIOR)

def theta_from(samples, i):
    return {k: samples[k][i] for k in THETA_KEYS}
