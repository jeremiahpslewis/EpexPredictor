"""Generative market model, level M1 (notes 21.8, 21.26): the merit order by fuel. Lignite, hard coal and gas are separate blocks with their own costs
(emission factors and the gas heat rate fixed), must-run share, capacity share and spread; shocks on the logit of utilisation (output stays inside
(0, installed capacity)). Imports (with the step near zero price), storage, the unmeasured part and the shelf as M0e.
M1n (notes 21.46): net imports also depend on the eight neighbours' published residual loads.
M1e (notes 21.148): demand responds to the price with a latent slope eta; the auction clears the curves against the price-blind
published demand; realised demand = cleared demand + forecast errors; imports and storage respond to the price in EUR (p / L_ref).
usage: market1e.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm, t as tdist
from scipy.special import expit, logit
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; numpyro.set_host_device_count(4); NU = 4.0; EPS = 0.1; LSC = np.sqrt(3.0) / np.pi
H = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(H) * 4 - len(E)), np.nan)]; H["co2"] = np.nanmean(E[: len(H) * 4].reshape(-1, 4), 1)
H["L"] = 1.85 * H.gas + 0.37 * H.co2; H["X"] = H.trade; H["S"] = H.psgen + H.pump; H["D"] = H.rlreal - H.other; H["D0"] = H.rl - H.other; H.index = H.index.tz_convert("Europe/Berlin"); FU = ("l", "c", "g"); QC = dict(l="lig", c="coal", g="gasg"); CAP = dict(l=18.0, c=12.0, g=33.0)
NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]; _n = pd.read_parquet(f"{SP}/out/star_data.parquet")[NBC].resample("1h").mean(); _n.index = _n.index.tz_convert(H.index.tz); H = H.join(_n, how="left")
ok = H[["p", "D", "D0", "lig", "coal", "gasg", "X", "S", "L", "co2"] + NBC].notna().all(1) & (H.L > 20) & (H.lig > 0) & (H.coal > 0) & (H.gasg > 0); H = H[ok]; END = os.environ.get("END", "2025-12-01"); TAG = os.environ.get("TAG", ""); TR = np.asarray(H.index < END); _f = np.asarray(H.index < "2025-12-01"); NMU = H[NBC][_f].mean().values; NSD = H[NBC][_f].std().values; ZN = (H[NBC].values - NMU) / NSD
LREF = float(H.L[np.asarray(H.index < "2025-12-01")].mean()); PREF = 90.0
SC = ("x0", "x1", "K_x", "c_z", "s_z", "K_st", "m_st", "s_st", "B_g", "K_s", "sd_X", "sd_S", "sd_d", "a_l", "a_c", "m_g", "eta", "sd_fc"); PN = SC + tuple(f"{a}_{f}" for f in FU for a in ("phi", "kap", "s", "sg"))
def cost(th, L, co2): return dict(l=th["a_l"] + co2, c=th["a_c"] + 0.85 * co2, g=th["m_g"] * L)
def model(D0, L, co2, Zn, p=None, X=None, S=None, lq=None, wq=None, qsum=None, Dr=None):
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6))
    K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5)); B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_S = numpyro.sample("sd_S", dist.HalfNormal(3.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(8.0)); eta = numpyro.sample("eta", dist.Normal(0.0, 1.0)); sd_fc = numpyro.sample("sd_fc", dist.HalfNormal(5.0)); a_l = numpyro.sample("a_l", dist.Normal(15.0, 8.0)); a_c = numpyro.sample("a_c", dist.Normal(35.0, 12.0)); m_g = numpyro.sample("m_g", dist.LogNormal(np.log(1.3), 0.3))
    P = {f: (numpyro.sample(f"phi_{f}", dist.Beta(2.0, 12.0)), numpyro.sample(f"kap_{f}", dist.Beta(4.0, 2.0)), numpyro.sample(f"s_{f}", dist.LogNormal(np.log(0.3), 0.5)), numpyro.sample(f"sg_{f}", dist.HalfNormal(1.0))) for f in FU}
    kn = numpyro.sample("kn", dist.Normal(0.0, 1.5).expand([8]))
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); phi_ = lambda z: jnp.exp(Nd.log_prob(z)); r = p / LREF; zS = (r - m_st) / s_st; zz = (p - c_z) / s_z; cf = dict(l=a_l + co2, c=a_c + 0.85 * co2, g=m_g * L)
    gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)) + Zn @ kn; gS = K_st * (2.0 * Nd.cdf(zS) - 1.0); slope = x1 / LREF + 2.0 * K_st * phi_(zS) / (s_st * LREF) + K_x * phi_(zz) / s_z + eta / 10.0; ll = 0.0; curve = gX + gS + B_g
    for j, f in enumerate(FU):
        ph, ka, s_, sg = P[f]; z = (p / cf[f] - 1.0) / s_; ut = ph + (1 - ph) * ka * Nd.cdf(z); e = lq[j] - (jnp.log(ut) - jnp.log1p(-ut)); ll = ll + dist.StudentT(NU, 0.0, sg).log_prob(e)
        curve = curve + CAP[f] * ut; slope = slope + CAP[f] * (1 - ph) * ka * phi_(z) / (s_ * cf[f])
    Dc = D0 - eta * (p - PREF) / 10.0; sh = jnp.abs(p) <= EPS; be = p < -EPS; u = Dc - curve; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc
    ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc))); ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be) + jnp.log(slope + 1e-12)
    numpyro.factor("ll", (ll + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + dist.StudentT(NU, 0.0, sd_S).log_prob(S - gS) + jnp.where(sh, ld_shelf, ld_cont) + dist.StudentT(NU, 0.0, sd_fc).log_prob(Dr - Dc)).sum())
def util(th, f, p, c): return th[f"phi_{f}"] + (1 - th[f"phi_{f}"]) * th[f"kap_{f}"] * norm.cdf((p / c - 1.0) / th[f"s_{f}"])
def other_curves(th, p, L): r = p / LREF; return th["x0"] + th["x1"] * (r - 1.0) - th["K_x"] * (1.0 - norm.cdf((p - th["c_z"]) / th["s_z"])), th["K_st"] * (2.0 * norm.cdf((r - th["m_st"]) / th["s_st"]) - 1.0)
def fuels(th, p, L, co2, e=None):
    cf = cost(th, L, co2); return {f: CAP[f] * expit(logit(util(th, f, p, cf[f])) + (0.0 if e is None else e[f])) for f in FU}
def supply(th, p, L, co2, e=None): gX, gS = other_curves(th, p, L); return sum(fuels(th, p, L, co2, e).values()) + gX + gS + th["B_g"]
def clear(th, D, L, co2, eps=0.0, e=None, eX=0.0, eS=0.0):
    """price where supply + eps meets demand D - eta (p - PREF) / 10 (D = price-blind demand)"""
    def ex(p, wd): return supply(th, p, L, co2, e) + eX + eS + eps - wd - D + th.get("eta", 0.0) * (p - PREF) / 10.0
    up = ex(np.full(np.shape(D), EPS), 0.0) < 0; dn = ex(np.full(np.shape(D), -EPS), th["K_s"]) > 0; lo = np.where(up, EPS, -3000.0); hi = np.where(up, 20000.0, -EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
def simulate(th, D0, L, co2, rng, sX=0.0):
    """the auction clears the curves (no block shocks) against D0 with slope eta and shock u; outputs and realised demand then scatter"""
    n = len(D0); p = clear(th, D0, L, co2, rng.logistic(0.0, th["sd_d"] * LSC, n) + sX); e = {f: tdist.rvs(NU, size=n, random_state=rng) * th[f"sg_{f}"] for f in FU}
    eX, eS = (tdist.rvs(NU, size=n, random_state=rng) * th[k] for k in ("sd_X", "sd_S")); gX, gS = other_curves(th, p, L); Dr = D0 - th["eta"] * (p - PREF) / 10.0 + tdist.rvs(NU, size=n, random_state=rng) * th["sd_fc"]
    return p, fuels(th, p, L, co2, e), gX + sX + eX, gS + eS, Dr
def data(g, p=None, q=None, X=None, S=None, Dr=None):
    q = q or {f: g[QC[f]].values for f in FU}; return dict(p=jnp.asarray(g.p.values if p is None else p), X=jnp.asarray(g.X.values if X is None else X), S=jnp.asarray(g.S.values if S is None else S), lq=jnp.asarray(np.stack([logit(q[f] / CAP[f]) for f in FU])), wq=jnp.asarray(np.stack([q[f] * (1 - q[f] / CAP[f]) for f in FU])), qsum=jnp.asarray(sum(q.values())), Dr=jnp.asarray(g.D.values if Dr is None else Dr))
def fit(g, dd, Zn, warm=700, draws=600, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(g.D0.values), jnp.asarray(g.L.values), jnp.asarray(g.co2.values), jnp.asarray(Zn), extra_fields=("diverging", "potential_energy"), **dd)
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(v["r_hat"])) for v in summ.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values()); pe = np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"]).mean(1); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), pe
if __name__ == "__main__":
    mode = sys.argv[1]; tr = H[TR]; rng = np.random.default_rng(0); ZT = ZN[TR]; pri = Predictive(model, num_samples=2000)(jax.random.PRNGKey(7), jnp.asarray(tr.D0.values[:5]), jnp.asarray(tr.L.values[:5]), jnp.asarray(tr.co2.values[:5]), jnp.asarray(ZT[:5])); pri = {k: np.asarray(v) for k, v in pri.items()}; q3 = lambda v: " ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95]))
    if mode == "prior":
        ii = rng.choice(len(tr), 3000, replace=False); sub = tr.iloc[ii]; sims = [simulate({k: float(pri[k][i]) for k in PN}, sub.D0.values, sub.L.values, sub.co2.values, rng, ZT[ii] @ pri["kn"][i]) for i in range(150)]; P = np.array([s_[0] for s_ in sims])
        print(f"prior predictive, 150 markets on 3000 real hours (5-50-95%; observed): median price / cost {q3(np.median(P / sub.L.values[None], 1))} ({np.median(sub.p / sub.L):.2f}); at or below zero {q3((P <= 0).mean(1))} ({(sub.p <= 0).mean():.3f}); on the shelf {q3((P == 0).mean(1))} ({(np.abs(sub.p) <= EPS).mean():.3f}); highest price ct {q3(P.max(1) / 10)} ({sub.p.max() / 10:.0f}); mean output GW " + "; ".join(f"{QC[f]} {q3([s_[1][f].mean() for s_ in sims])} ({sub[QC[f]].mean():.1f})" for f in FU) + "; highest output GW " + "; ".join(f"{QC[f]} {q3([s_[1][f].max() for s_ in sims])} ({sub[QC[f]].max():.1f})" for f in FU))
    if mode == "recover":
        TRUE = dict(x0=3.0, x1=8.0, K_x=4.0, c_z=5.0, s_z=7.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_X=4.0, sd_S=1.5, sd_d=5.0, eta=0.5, sd_fc=3.0, a_l=12.0, a_c=40.0, m_g=1.4, phi_l=0.17, kap_l=0.6, s_l=0.3, sg_l=0.5, phi_c=0.05, kap_c=0.6, s_c=0.3, sg_c=0.9, phi_g=0.04, kap_g=0.6, s_g=0.4, sg_g=0.7)
        KN0 = np.array([-0.1, 0.1, -0.3, -0.3, 0.1, -1.0, -0.6, 0.3]); keep = np.arange(len(tr))[::2]; keep = keep[rng.random(len(keep)) > 0.05]; sub = tr.iloc[keep]; p, q, X, S, Dr = simulate(TRUE, sub.D0.values, sub.L.values, sub.co2.values, rng, ZT[keep] @ KN0); s, rh, ess, div, pe = fit(sub, data(sub, p, q, X, S, Dr), ZT[keep], warm=500, draws=400); qk = np.percentile(s["kn"], [5, 50, 95], axis=0); print("   kn true", KN0, "recovered", np.round(qk[1], 2), "inside the 90% band:", int(((qk[0] <= KN0) & (KN0 <= qk[2])).sum()), "of 8"); nin = 0
        print(f"simulate and recover: {len(sub)} hours; worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in PN:
            qq = np.percentile(s[k], [5, 50, 95]); okk = qq[0] <= TRUE[k] <= qq[2]; nin += okk
            if not okk or k in ("kap_g", "m_g", "a_l", "a_c", "eta", "sd_fc", "sd_d"): print(f"   {k:6s} true {TRUE[k]:6.2f}  recovered {qq[1]:6.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})  {'ok' if okk else 'OUTSIDE the 90% band'}")
        print(f"   {nin} of {len(PN)} inside their 90% bands")
    if mode == "fit":
        s, rh, ess, div, pe = fit(tr, data(tr), ZT, warm=int(os.environ.get("WARM", 700)), draws=int(os.environ.get("DRAWS", 600)), seed=int(os.environ.get("SEED", 0))); th = {k: float(np.median(s[k])) for k in PN}
        print(f"M1e fitted on {len(tr)} hours (to {END}): worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['x0'])}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in PN: qq = np.percentile(s[k], [5, 50, 95]); pc = float((pri[k] < qq[1]).mean()); print(f"   {k:6s} {qq[1]:7.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})   prior percentile {100 * pc:5.1f}{'   <-- conflict' if pc < 0.005 or pc > 0.995 else ''}")
        kn = np.median(s["kn"], 0); print("   kn, GW of net imports per sd of each neighbour's residual load (fr nl be at pl cz ch dk):", np.round(kn, 2))
        print(f"   eta posterior sd {np.std(s['eta']):.3f} (prior sd 1.0); L_ref {LREF:.1f}; import slope per EUR/MWh x1 / L_ref {np.median(s['x1']) / LREF:.3f}"); np.savez_compressed(f"{SP}/out/market1e{TAG}_post.npz", **{k: s[k] for k in PN}, rhat=rh, ess=ess, div=div, kn=s["kn"], NMU=NMU, NSD=NSD)
        if TAG: sys.exit(0)
        cf = cost(th, H.L.values, H.co2.values); g0 = fuels(th, H.p.values, H.L.values, H.co2.values); gX, gS = other_curves(th, H.p.values, H.L.values)
        for f in FU: H[f"e_{f}"] = logit(H[QC[f]].values / CAP[f]) - logit(g0[f] / CAP[f]); H[f"d_{f}"] = H[QC[f]].values - g0[f]; H[f"rc_{f}"] = H.p.values / cf[f]
        H["eX"] = H.X - gX - ZN @ kn; H["eS"] = H.S - gS; H["eps"] = H.D - H.lig - H.coal - H.gasg - H.X - H.S - th["B_g"] + th["K_s"] * (H.p.values < -EPS) + 0.5 * th["K_s"] * (np.abs(H.p.values) <= EPS); H["p0"] = clear(th, H.D.values, H.L.values, H.co2.values, eX=ZN @ kn); H["tr"] = TR; H.to_parquet(f"{SP}/out/market1e_rows.parquet")
        for lab, m in (("fitting period", TR), ("held-out year", ~TR)):
            g = H[m]; print(f"   {lab}: bare price MAE {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} ct; mean output minus curve GW: lignite {g.d_l.mean():+.2f}, coal {g.d_c.mean():+.2f}, gas {g.d_g.mean():+.2f}, imports {g.eX.mean():+.2f}, storage {g.eS.mean():+.2f}, unmeasured {g.eps.mean():+.2f}; sd {g.d_l.std():.2f}, {g.d_c.std():.2f}, {g.d_g.std():.2f}, {g.eX.std():.2f}")
