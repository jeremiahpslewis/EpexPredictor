"""Generative market model, level M0e (notes 21.16): M0d plus a step in supply near a price of zero, in EUR/MWh, in thermal output and in net imports.
(M0d: M0c with the rest of the system in the right units.)
Storage is its own observed block in price / cost; other plant is taken as given (demand D = realised residual load - other plant); the unmeasured
part of the balance closes the system. No fuel split. Four shocks in GW; likelihood = shock densities x slope of excess supply at the observed price.
usage: market0e.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; numpyro.set_host_device_count(4); NU = 4.0
H = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(H) * 4 - len(E)), np.nan)]; H["co2"] = np.nanmean(E[: len(H) * 4].reshape(-1, 4), 1)
H["L"] = 1.85 * H.gas + 0.37 * H.co2; H["X"] = H.trade; H["S"] = H.psgen + H.pump; H["D"] = H.rlreal - H.other; H.index = H.index.tz_convert("Europe/Berlin"); ok = H[["p", "D", "T", "X", "S", "L"]].notna().all(1) & (H.L > 20); H = H[ok]
END = os.environ.get("END", "2025-12-01"); TAG = os.environ.get("TAG", "")   # walk-forward refits: END = first day NOT in the fit, TAG = suffix of the output
TR = np.asarray(H.index < END); HO = ~TR
PN = ("F_T", "K_T", "m_T", "s_T", "x0", "x1", "K_st", "m_st", "s_st", "B_g", "K_s", "sd_T", "sd_X", "sd_S", "sd_d", "K_z", "K_x", "c_z", "s_z"); EPS = 0.1; LSC = np.sqrt(3.0) / np.pi
def model(D, L, p=None, T=None, X=None, S=None):
    F_T = numpyro.sample("F_T", dist.LogNormal(np.log(4.0), 0.4)); K_T = numpyro.sample("K_T", dist.LogNormal(np.log(35.0), 0.3)); m_T = numpyro.sample("m_T", dist.Normal(1.0, 0.3)); s_T = numpyro.sample("s_T", dist.LogNormal(np.log(0.4), 0.5))
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6))
    K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5))
    B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_T = numpyro.sample("sd_T", dist.HalfNormal(5.0)); sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_S = numpyro.sample("sd_S", dist.HalfNormal(3.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(3.0))
    K_z = numpyro.sample("K_z", dist.LogNormal(np.log(2.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6))
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); r = p / L; zT = (r - m_T) / s_T; zS = (r - m_st) / s_st; zz = (p - c_z) / s_z
    gT = F_T + K_T * Nd.cdf(zT) + K_z * Nd.cdf(zz); gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)); gS = K_st * (2.0 * Nd.cdf(zS) - 1.0)
    slope = K_T * jnp.exp(Nd.log_prob(zT)) / (s_T * L) + x1 / L + 2.0 * K_st * jnp.exp(Nd.log_prob(zS)) / (s_st * L) + (K_z + K_x) * jnp.exp(Nd.log_prob(zz)) / s_z
    sh = jnp.abs(p) <= EPS; be = p < -EPS; u = D - T - X - S - B_g; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc     # unmeasured supply = B_g + eps - withdrawal
    ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc)))                       # log P(eps in [u, u + K_s]): cleared on the shelf
    ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be) + jnp.log(slope + 1e-12)
    ll = dist.StudentT(NU, 0.0, sd_T).log_prob(T - gT) + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + dist.StudentT(NU, 0.0, sd_S).log_prob(S - gS) + jnp.where(sh, ld_shelf, ld_cont)
    numpyro.factor("ll", ll.sum())
def curves(th, p, L):
    r = p / L; st = norm.cdf((p - th["c_z"]) / th["s_z"]); return th["F_T"] + th["K_T"] * norm.cdf((r - th["m_T"]) / th["s_T"]) + th["K_z"] * st, th["x0"] + th["x1"] * (r - 1.0) - th["K_x"] * (1.0 - st), th["K_st"] * (2.0 * norm.cdf((r - th["m_st"]) / th["s_st"]) - 1.0)
def clear(th, D, L, eps=0.0, eT=0.0, eX=0.0, eS=0.0):
    """market-clearing price for given shocks; excess supply rises in the price and drops by K_s below zero (the shelf)"""
    def ex(p, wd): gT, gX, gS = curves(th, p, L); return gT + eT + gX + eX + gS + eS + th["B_g"] + eps - wd - D
    up = ex(np.full(np.shape(D), EPS), 0.0) < 0; dn = ex(np.full(np.shape(D), -EPS), th["K_s"]) > 0; lo = np.where(up, EPS, -3000.0); hi = np.where(up, 20000.0, -EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    p = np.where(up | dn, 0.5 * (lo + hi), 0.0); gT, gX, gS = curves(th, p, L); return p, gT + eT, gX + eX, gS + eS
def simulate(th, D, L, rng):
    n = len(D); eps = rng.logistic(0.0, th["sd_d"] * LSC, n); eT, eX, eS = (tdist.rvs(NU, size=n, random_state=rng) * th[k] for k in ("sd_T", "sd_X", "sd_S")); return clear(th, D, L, eps, eT, eX, eS)
def fit(D, L, p, T, X, S, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(D), jnp.asarray(L), p=jnp.asarray(p), T=jnp.asarray(T), X=jnp.asarray(X), S=jnp.asarray(S), extra_fields=("diverging",))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
if __name__ == "__main__":
    mode = sys.argv[1]; tr = H[TR]; rng = np.random.default_rng(0); pri = Predictive(model, num_samples=2000)(jax.random.PRNGKey(7), jnp.asarray(tr.D.values[:5]), jnp.asarray(tr.L.values[:5])); pri = {k: np.asarray(v) for k, v in pri.items()}
    if mode == "prior":
        sub = tr.iloc[rng.choice(len(tr), 3000, replace=False)]; sims = []
        for i in range(200):
            th = {k: float(pri[k][i]) for k in PN}; p, T, X, S = simulate(th, sub.D.values, sub.L.values, rng); sims.append((p.astype(np.float32), T.astype(np.float32), X.astype(np.float32), S.astype(np.float32)))
        P, T_, X_, S_ = (np.array([s_[j] for s_ in sims]) for j in range(4)); np.savez_compressed(f"{SP}/out/market0e_prior.npz", D=sub.D.values, rl=sub.rlreal.values, L=sub.L.values, p_obs=sub.p.values, T_obs=sub["T"].values, X_obs=sub.X.values, S_obs=sub.S.values, P=P, T=T_, X=X_, S=S_, **{k: pri[k][:200] for k in PN})
        q = lambda v: " ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95]))
        print(f"prior predictive, 200 simulated markets on 3000 real hours (5-50-95% over markets; observed): median price / cost {q(np.median(P / sub.L.values[None], 1))} ({np.median(sub.p / sub.L):.2f}); at or below zero {q((P <= 0).mean(1))} ({(sub.p <= 0).mean():.3f}); on the shelf {q((P == 0).mean(1))} ({(np.abs(sub.p) <= EPS).mean():.3f}); mean thermal GW {q(T_.mean(1))} ({sub['T'].mean():.1f}); mean storage GW {q(S_.mean(1))} ({sub.S.mean():.2f}); sd storage {q(S_.std(1))} ({sub.S.std():.2f}); thermal below zero, share of hours {q((T_ < 0).mean(1))}; highest price ct {q(P.max(1) / 10)} ({sub.p.max() / 10:.0f}); lowest price ct {q(P.min(1) / 10)} ({sub.p.min() / 10:.0f})")
    if mode == "recover":
        TRUE = dict(F_T=3.0, K_T=30.0, m_T=1.1, s_T=0.35, x0=3.0, x1=8.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_T=2.0, sd_X=4.0, sd_S=1.5, sd_d=2.0, K_z=2.0, K_x=4.0, c_z=8.0, s_z=7.0); sub = tr.iloc[::2].copy(); miss = rng.random(len(sub)) < 0.05
        for g0 in rng.integers(0, len(sub) - 200, 8): miss[g0:g0 + rng.integers(24, 120)] = True
        sub = sub[~miss]; p, T, X, S = simulate(TRUE, sub.D.values, sub.L.values, rng); s, rh, ess, div = fit(sub.D.values, sub.L.values, p, T, X, S)
        print(f"simulate and recover: {len(sub)} hours (every second hour, 5% missing at random and eight gaps of 1-5 days); worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); print(f"   {k:5s} true {TRUE[k]:7.2f}  recovered {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})  {'ok' if q[0] <= TRUE[k] <= q[2] else 'OUTSIDE the 90% band'}")
        np.savez_compressed(f"{SP}/out/market0e_recover.npz", **{f"true_{k}": TRUE[k] for k in PN}, **{k: s[k] for k in PN}, rhat=rh, ess=ess)
    if mode == "fit":
        s, rh, ess, div = fit(tr.D.values, tr.L.values, tr.p.values, tr["T"].values, tr.X.values, tr.S.values, warm=int(os.environ.get("WARM", 700)), draws=int(os.environ.get("DRAWS", 600)), seed=int(os.environ.get("SEED", 0))); th = {k: float(np.median(s[k])) for k in PN}
        print(f"M0e fitted on {len(tr)} hours (Mar 2024 - {END}): worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['B_g'])}, divergent {div}")
        for k in PN: q = np.percentile(s[k], [5, 50, 95]); pc = float((pri[k] < q[1]).mean()); print(f"   {k:5s} {q[1]:7.2f}  ({q[0]:.2f} .. {q[2]:.2f})   posterior median at prior percentile {100 * pc:5.1f}{'   <-- prior and data in conflict' if pc < 0.005 or pc > 0.995 else ''}")
        np.savez_compressed(f"{SP}/out/market0e{TAG}_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div)
        if TAG: print(f"refit {TAG}: data to {END}, {len(tr)} hours, r-hat {rh:.3f}, ess {ess:.0f}, divergent {div}"); sys.exit(0)
        gT, gX, gS = curves(th, H.p.values, H.L.values); H["eT"] = H["T"] - gT; H["eX"] = H.X - gX; H["eS"] = H.S - gS; H["eps"] = H.D - H["T"] - H.X - H.S - th["B_g"] + th["K_s"] * (H.p.values < -EPS) + 0.5 * th["K_s"] * (np.abs(H.p.values) <= EPS)
        pz, Tz, Xz, Sz = clear(th, H.D.values, H.L.values); H["p0"] = pz; H["T0"] = Tz; H["X0"] = Xz; H["S0"] = Sz; H["tr"] = TR; H.to_parquet(f"{SP}/out/market0e_rows.parquet")
        for lab, m in (("fitting period", TR), ("held-out year", HO)):
            g = H[m]; print(f"   {lab}: price with all shocks at zero against observed: MAE {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} ct, median |price / cost error| {np.median(np.abs(g.p0 - g.p) / g.L):.3f}; MAE thermal {np.mean(np.abs(g.T0 - g['T'])):.2f}, imports {np.mean(np.abs(g.X0 - g.X)):.2f}, storage {np.mean(np.abs(g.S0 - g.S)):.2f} GW; mean shocks eT {g.eT.mean():+.2f}, eX {g.eX.mean():+.2f}, eS {g.eS.mean():+.2f}, eps {g.eps.mean():+.2f} GW; sd {g.eT.std():.2f}, {g.eX.std():.2f}, {g.eS.std():.2f}, {g.eps.std():.2f}")
