"""Generative market model, level M1d (notes 21.68): M1n on the day-ahead basis. Observed per hour: price, scheduled dispatchable minus other plant,
day-ahead net imports; demand = published residual load minus other plant. Three shocks, full-information likelihood with the Jacobian of the price.
usage: market1d.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; numpyro.set_host_device_count(4); NU = 4.0; EPS = 0.1; LSC = np.sqrt(3.0) / np.pi; FU = ("l", "c", "g"); CAP = dict(l=18.0, c=12.0, g=33.0)
P0 = np.load(f"{SP}/out/market1n_post.npz"); M0 = {k: float(np.median(P0[k])) for k in P0.files if P0[k].ndim == 1 and len(P0[k]) > 100}; NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]
SC = ("x0", "x1", "K_x", "c_z", "s_z", "K_g", "m_st", "s_st", "B_g", "B_c", "K_p", "m_p", "s_p", "K_s", "sd_T", "sd_X", "sd_d", "a_l", "a_c", "m_g"); PN = SC + tuple(f"{a}_{f}" for f in FU for a in ("phi", "kap", "s"))
def frame():
    H = pd.read_parquet(f"{SP}/out/da_rows.parquet"); H["D"] = H.rl - H.other; H["Td"] = H.disp_da - H.other; H["Xd"] = -H["np"]; H = H.dropna(subset=["p", "D", "Td", "Xd", "L", "co2"] + NBC); H = H[H.L > 20]; f = np.asarray(H.tr == True); H["tr"] = f; return H, (H[NBC].values - P0["NMU"]) / P0["NSD"]
def bp(m, c): return dist.Beta(m * c, (1 - m) * c)
def model(D, L, co2, Zn, p=None, Td=None, Xd=None):
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6))
    K_g = numpyro.sample("K_g", dist.LogNormal(np.log(3.0), 0.5)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5)); B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); B_c = numpyro.sample("B_c", dist.Normal(1.0, 2.0))
    K_p = numpyro.sample("K_p", dist.LogNormal(np.log(3.0), 0.5)); m_p = numpyro.sample("m_p", dist.Normal(0.7, 0.3)); s_p = numpyro.sample("s_p", dist.LogNormal(np.log(0.4), 0.5)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_T = numpyro.sample("sd_T", dist.HalfNormal(5.0)); sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(3.0))
    a_l = numpyro.sample("a_l", dist.Normal(M0["a_l"], 3.0)); a_c = numpyro.sample("a_c", dist.Normal(M0["a_c"], 8.0)); m_g = numpyro.sample("m_g", dist.LogNormal(np.log(M0["m_g"]), 0.1))
    Pf = {f: (numpyro.sample(f"phi_{f}", bp(M0[f"phi_{f}"], 50.0)), numpyro.sample(f"kap_{f}", bp(M0[f"kap_{f}"], 30.0)), numpyro.sample(f"s_{f}", dist.LogNormal(np.log(M0[f"s_{f}"]), 0.25))) for f in FU}
    kn = numpyro.sample("kn", dist.Normal(0.0, 1.5).expand([8]))
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); ph = lambda z: jnp.exp(Nd.log_prob(z)); r = p / L; cf = dict(l=a_l + co2, c=a_c + 0.85 * co2, g=m_g * L); zS = (r - m_st) / s_st; zz = (p - c_z) / s_z; zp = (r - m_p) / s_p
    gT = K_g * Nd.cdf(zS) + B_g; slope = K_g * ph(zS) / (s_st * L) + x1 / L + K_x * ph(zz) / s_z + K_p * ph(zp) / (s_p * L)
    for f in FU:
        ph_, ka, s_ = Pf[f]; z = (p / cf[f] - 1.0) / s_; gT = gT + CAP[f] * (ph_ + (1 - ph_) * ka * Nd.cdf(z)); slope = slope + CAP[f] * (1 - ph_) * ka * ph(z) / (s_ * cf[f])
    gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)) + Zn @ kn; gC = B_c - K_p * (1.0 - Nd.cdf(zp)); sh = jnp.abs(p) <= EPS; be = p < -EPS; u = D - Td - Xd - gC; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc
    ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc))); ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be) + jnp.log(slope + 1e-12)
    numpyro.factor("ll", (dist.StudentT(NU, 0.0, sd_T).log_prob(Td - gT) + dist.StudentT(NU, 0.0, sd_X).log_prob(Xd - gX) + jnp.where(sh, ld_shelf, ld_cont)).sum())
def curves(th, p, L, co2):
    """dispatchable (minus other plant), imports without neighbour term, closing part: all without shocks"""
    r = p / L; cf = dict(l=th["a_l"] + co2, c=th["a_c"] + 0.85 * co2, g=th["m_g"] * L); gT = th["K_g"] * norm.cdf((r - th["m_st"]) / th["s_st"]) + th["B_g"] + sum(CAP[f] * (th[f"phi_{f}"] + (1 - th[f"phi_{f}"]) * th[f"kap_{f}"] * norm.cdf((p / cf[f] - 1.0) / th[f"s_{f}"])) for f in FU)
    return gT, th["x0"] + th["x1"] * (r - 1.0) - th["K_x"] * (1.0 - norm.cdf((p - th["c_z"]) / th["s_z"])), th["B_c"] - th["K_p"] * (1.0 - norm.cdf((r - th["m_p"]) / th["s_p"]))
def supply(th, p, L, co2): return sum(curves(th, p, L, co2))
def clear(th, D, L, co2, eps=0.0):
    D = np.asarray(D, float); L = np.broadcast_to(np.asarray(L, float), D.shape); co2 = np.broadcast_to(np.asarray(co2, float), D.shape); eps = np.broadcast_to(np.asarray(eps, float), D.shape)
    def ex(p, wd): return supply(th, p, L, co2) + eps - wd - D
    up = ex(np.full(D.shape, EPS), 0.0) < 0; dn = ex(np.full(D.shape, -EPS), th["K_s"]) > 0; lo = np.where(up, EPS, -3000.0); hi = np.where(up, 20000.0, -EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
def simulate(th, kn, D, L, co2, Zn, rng):
    n = len(D); eT = tdist.rvs(NU, size=n, random_state=rng) * th["sd_T"]; eX = tdist.rvs(NU, size=n, random_state=rng) * th["sd_X"] + Zn @ kn; ee = rng.logistic(0.0, th["sd_d"] * LSC, n); p = clear(th, D, L, co2, eT + eX + ee); gT, gX, gC = curves(th, p, L, co2); return p, gT + eT, gX + eX
def fit(g, Zn, p, Td, Xd, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(g.D.values), jnp.asarray(g.L.values), jnp.asarray(g.co2.values), jnp.asarray(Zn), p=jnp.asarray(p), Td=jnp.asarray(Td), Xd=jnp.asarray(Xd), extra_fields=("diverging", "potential_energy"))
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); pe = np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"]).mean(1); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), pe
if __name__ == "__main__":
    mode = sys.argv[1]; H, ZN = frame(); TR = H.tr.values; tr = H[TR]; ZT = ZN[TR]; rng = np.random.default_rng(0); q3 = lambda v: " ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95]))
    pri = Predictive(model, num_samples=1000)(jax.random.PRNGKey(7), jnp.asarray(tr.D.values[:5]), jnp.asarray(tr.L.values[:5]), jnp.asarray(tr.co2.values[:5]), jnp.asarray(ZT[:5])); pri = {k: np.asarray(v) for k, v in pri.items()}
    if mode == "prior":
        ii = rng.choice(len(tr), 3000, replace=False); sub = tr.iloc[ii]; sims = [simulate({k: float(pri[k][i]) for k in PN}, pri["kn"][i], sub.D.values, sub.L.values, sub.co2.values, ZT[ii], rng) for i in range(150)]; Pm = np.array([s_[0] for s_ in sims]); print(f"prior predictive, 150 markets on 3000 real hours ({len(tr)} fitting hours; 5-50-95%; observed)")
        for lab, v, o in (("median price / cost", np.median(Pm / sub.L.values[None], 1), np.median(sub.p / sub.L)), ("share at or below zero", (Pm <= 0).mean(1), (sub.p <= 0).mean()), ("share on the shelf", (Pm == 0).mean(1), (np.abs(sub.p) <= EPS).mean()), ("highest price ct", Pm.max(1) / 10, sub.p.max() / 10), ("lowest price ct", Pm.min(1) / 10, sub.p.min() / 10), ("mean scheduled dispatchable minus other GW", [s_[1].mean() for s_ in sims], sub.Td.mean()), ("its 5% point", [np.percentile(s_[1], 5) for s_ in sims], np.percentile(sub.Td, 5)), ("its 95% point", [np.percentile(s_[1], 95) for s_ in sims], np.percentile(sub.Td, 95)), ("mean net imports GW", [s_[2].mean() for s_ in sims], sub.Xd.mean()), ("sd of net imports", [s_[2].std() for s_ in sims], sub.Xd.std())): lo_, md_, hi_ = np.percentile(v, [5, 50, 95]); print(f"   {lab:44s} {lo_:8.3g} {md_:8.3g} {hi_:8.3g}   observed {o:8.3g}   {'inside' if lo_ <= o <= hi_ else 'OUTSIDE'}")
    if mode == "recover":
        TRUE = dict(M0); TRUE.update(K_g=3.0, B_c=1.0, K_p=3.0, m_p=0.6, s_p=0.3, sd_T=2.5, sd_X=3.5, sd_d=1.0, x0=3.0, x1=12.0, B_g=3.0); TRUE = {k: TRUE[k] for k in PN}; KN0 = np.array([-0.1, -0.4, -1.5, -1.0, -0.4, -1.0, -0.6, 0.0]); keep = np.arange(len(tr))[::2]; sub = tr.iloc[keep]; p, Td, Xd = simulate(TRUE, KN0, sub.D.values, sub.L.values, sub.co2.values, ZT[keep], rng); s, rh, ess, div, pe = fit(sub, ZT[keep], p, Td, Xd, warm=500, draws=400); nin = 0
        print(f"simulate and recover: {len(sub)} hours; worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in PN:
            qq = np.percentile(s[k], [5, 50, 95]); okk = qq[0] <= TRUE[k] <= qq[2]; nin += okk; print(f"   {k:6s} true {TRUE[k]:7.3f}  recovered {qq[1]:7.3f}  ({qq[0]:.3f} .. {qq[2]:.3f})  {'ok' if okk else 'OUTSIDE the 90% band'}")
        qk = np.percentile(s["kn"], [5, 50, 95], axis=0); print(f"   {nin} of {len(PN)} inside their 90% bands; neighbour coefficients inside: {int(((qk[0] <= KN0) & (KN0 <= qk[2])).sum())} of 8")
    if mode == "fit":
        s, rh, ess, div, pe = fit(tr, ZT, tr.p.values, tr.Td.values, tr.Xd.values, warm=int(os.environ.get("WARM", 700)), draws=int(os.environ.get("DRAWS", 600)), seed=int(os.environ.get("SEED", 0))); print(f"M1d fitted on {len(tr)} hours: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}; log density by chain {np.round(-pe)}")
        np.savez_compressed(f"{SP}/out/market1d_post.npz", **{k: s[k] for k in PN}, kn=s["kn"], rhat=rh, ess=ess, div=div, NMU=P0["NMU"], NSD=P0["NSD"])
        for k in PN: qq = np.percentile(s[k], [5, 50, 95]); pr = float((pri[k] < qq[1]).mean()); print(f"   {k:6s} {qq[1]:8.3f}  ({qq[0]:.3f} .. {qq[2]:.3f})   prior percentile {100 * pr:5.1f}" + (f"   M1n {M0[k]:.3f}" if k in M0 else ""))
        th = {k: float(np.median(s[k])) for k in PN}; kn = np.median(s["kn"], 0); print("   kn:", np.round(kn, 2)); gT, gX, gC = curves(th, H.p.values, H.L.values, H.co2.values); H["eT"] = H.Td - gT; H["eXd"] = H.Xd - gX - ZN @ kn; H["ec"] = H.D - H.Td - H.Xd - gC; H["nbterm"] = ZN @ kn; H["p0"] = clear(th, H.D.values, H.L.values, H.co2.values, H.nbterm.values); H.to_parquet(f"{SP}/out/market1d_rows.parquet")
        for nm, m in (("fitting period", TR), ("after", ~TR)): print(f"   {nm}: bare price MAE {np.abs(H.p0 - H.p)[m].mean() / 10:.2f} ct; mean shocks GW dispatch {H.eT[m].mean():+.2f}, imports {H.eXd[m].mean():+.2f}, closing {H.ec[m].mean():+.2f}; sd {H.eT[m].std():.2f}, {H.eXd[m].std():.2f}, {H.ec[m].std():.2f}")
        F = H[TR].copy(); F["b"] = pd.cut(F.D, [-99, 0, 5, 10, 15, 20, 25, 30, 35, 99]); print("   fitting period, mean shock by demand bin:"); print(F.groupby("b").agg(n=("eT", "size"), dispatch=("eT", "mean"), imports=("eXd", "mean"), closing=("ec", "mean")).round(2).T.to_string())
