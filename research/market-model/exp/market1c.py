"""Merit order by fuel with what the shock theories found (notes 21.35). Switches (env, 1/0): COV availability + temperature + weekend on each
fuel's utilisation; MEM a day-level shock with memory per fuel; BLK lignite and coal answer a mix of the hour's and the day's mean price.
usage: [COV=1 MEM=1 BLK=1] market1c.py prior | recover | fit"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, market1h as Mh
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import t as tdist
from scipy.special import expit, logit
SP = M.SP; NU = M.NU; EPS = M.EPS; LSC = M.LSC; FU = M.FU; CAP = M.CAP; QC = M.QC; COV, MEM, BLK = (os.environ.get(k, "1") == "1" for k in ("COV", "MEM", "BLK")); DAYRE = os.environ.get("DAYRE", "0") == "1"   # a shock common to all hours of TODAY per fuel (Gaussian, integrated out), on top of the memory of earlier days
TAG = "market1" + ("c" if COV else "") + ("m" if MEM else "") + ("b" if BLK else "") + ("r" if DAYRE else "")
PN = M.PN + (tuple(f"b{k}_{f}" for f in FU for k in "ATW") if COV else ()) + (tuple(f"{k}_{f}" for f in FU for k in ("rho", "kap")) if MEM else ()) + (("al_l", "al_c") if BLK else ())
def frame():
    import live_common as lc; H = M.H.copy(); z = np.load(f"{SP}/out/avail_states.npz"); AV = pd.DataFrame(z["av"][:, :2], index=pd.DatetimeIndex(z["index"]).tz_localize("UTC"), columns=["avs", "avg"]).resample("1h").mean(); AV.index = AV.index.tz_convert(H.index.tz); T = lc.load()["DE"][["eu_temperature"]].resample("1h").mean(); T.index = T.index.tz_convert(H.index.tz)
    H = H.join(AV, how="left").join(T, how="left"); tr = np.asarray(H.index < "2025-12-01"); st = {c: (float(H[c][tr].mean()), float(H[c][tr].std())) for c in ("avs", "avg", "eu_temperature")}
    for c in st: H[c + "_z"] = ((H[c] - st[c][0]) / st[c][1]).fillna(0.0)
    H["we"] = (H.index.dayofweek >= 5).astype(float); return H, st
def grid(H):
    G = H.copy(); G.index = G.index.tz_convert("Etc/GMT-1"); full = pd.date_range(G.index.min().floor("D"), G.index.max().floor("D") + pd.Timedelta(hours=23), freq="1h", tz="Etc/GMT-1"); G = G.reindex(full); v = G.p.notna().values
    A = {c: np.where(v, G[c].values, d).reshape(-1, 24).astype(np.float64) for c, d in Mh.COLS + (("avs_z", 0.0), ("avg_z", 0.0), ("eu_temperature_z", 0.0), ("we", 0.0))}; A["v"] = v.reshape(-1, 24); A["index"] = full; A["Z"] = np.stack([np.stack([A["avs_z"], A["eu_temperature_z"], A["we"]]), np.stack([A["avs_z"], A["eu_temperature_z"], A["we"]]), np.stack([A["avg_z"], A["eu_temperature_z"], A["we"]])]); return A   # Z: (fuel, 3, days, 24)
def model(D, L, co2, v, Z, p=None, X=None, S=None, lq=None, wq=None, qsum=None):
    x0 = numpyro.sample("x0", dist.Normal(2.0, 4.0)); x1 = numpyro.sample("x1", dist.LogNormal(np.log(8.0), 0.6)); K_x = numpyro.sample("K_x", dist.LogNormal(np.log(4.0), 0.6)); c_z = numpyro.sample("c_z", dist.Normal(0.0, 10.0)); s_z = numpyro.sample("s_z", dist.LogNormal(np.log(6.0), 0.6))
    K_st = numpyro.sample("K_st", dist.LogNormal(np.log(5.0), 0.3)); m_st = numpyro.sample("m_st", dist.Normal(1.0, 0.3)); s_st = numpyro.sample("s_st", dist.LogNormal(np.log(0.4), 0.5)); B_g = numpyro.sample("B_g", dist.Normal(3.0, 2.0)); K_s = numpyro.sample("K_s", dist.LogNormal(np.log(3.0), 0.6))
    sd_X = numpyro.sample("sd_X", dist.HalfNormal(5.0)); sd_S = numpyro.sample("sd_S", dist.HalfNormal(3.0)); sd_d = numpyro.sample("sd_d", dist.HalfNormal(3.0)); a_l = numpyro.sample("a_l", dist.Normal(15.0, 8.0)); a_c = numpyro.sample("a_c", dist.Normal(35.0, 12.0)); m_g = numpyro.sample("m_g", dist.LogNormal(np.log(1.3), 0.3))
    P = {f: (numpyro.sample(f"phi_{f}", dist.Beta(2.0, 12.0)), numpyro.sample(f"kap_{f}", dist.Beta(4.0, 2.0)), numpyro.sample(f"s_{f}", dist.LogNormal(np.log(0.3), 0.5)), numpyro.sample(f"sg_{f}", dist.HalfNormal(1.0))) for f in FU}
    Bc = {f: [numpyro.sample(f"b{k}_{f}", dist.Normal(0.0, 1.0)) for k in "ATW"] if COV else [0.0, 0.0, 0.0] for f in FU}; Mm = {f: (numpyro.sample(f"rho_{f}", dist.Beta(8.0, 2.0)), numpyro.sample(f"kap_{f}_", dist.Beta(2.0, 6.0))) if MEM else None for f in FU}
    tau = {f: numpyro.sample(f"tau_{f}", dist.HalfNormal(1.0)) if DAYRE else None for f in FU}
    al = dict(l=numpyro.sample("al_l", dist.Beta(2.0, 2.0)) if BLK else 1.0, c=numpyro.sample("al_c", dist.Beta(2.0, 2.0)) if BLK else 1.0, g=1.0)
    if MEM:
        for f in FU: numpyro.deterministic(f"kapm_{f}", Mm[f][1])
    if p is None: return
    Nd = dist.Normal(0.0, 1.0); phi_ = lambda z: jnp.exp(Nd.log_prob(z)); r = p / L; zS = (r - m_st) / s_st; zz = (p - c_z) / s_z; cf = dict(l=a_l + co2, c=a_c + 0.85 * co2, g=m_g * L); vf = v.astype(p.dtype); nv = jnp.maximum(vf.sum(1, keepdims=True), 1.0); pbar = (vf * p).sum(1, keepdims=True) / nv
    gX = x0 + x1 * (r - 1.0) - K_x * (1.0 - Nd.cdf(zz)); gS = K_st * (2.0 * Nd.cdf(zS) - 1.0); own = x1 / L + 2.0 * K_st * phi_(zS) / (s_st * L) + K_x * phi_(zz) / s_z + 1e-12; cross = 0.0; ll = 0.0; llre = 0.0
    for j, f in enumerate(FU):
        ph, ka, s_, sg = P[f]; pe = al[f] * p + (1 - al[f]) * pbar; z = (pe / cf[f] - 1.0) / s_; ut = ph + (1 - ph) * ka * Nd.cdf(z); e = lq[j] - (jnp.log(ut) - jnp.log1p(-ut)) - (Bc[f][0] * Z[j, 0] + Bc[f][1] * Z[j, 1] + Bc[f][2] * Z[j, 2])
        if MEM:
            rho, kp = Mm[f]; eb = (vf * e).sum(1) / nv[:, 0]; has = vf.sum(1) > 0
            def step(m, x, rho=rho, kp=kp): eb_d, has_d = x; pred = rho * m; return jnp.where(has_d, pred + kp * (eb_d - pred), pred), pred
            _, pred = jax.lax.scan(step, 0.0, (eb, has)); e = e - pred[:, None]
        if DAYRE:
            n_ = vf.sum(1); eb2 = (vf * e).sum(1) / nv[:, 0]; dev = vf * (e - eb2[:, None]); s2 = sg ** 2; t2 = s2 + n_ * tau[f] ** 2; llre = llre + jnp.where(n_ > 0, -0.5 * ((dev ** 2).sum(1) / s2 + n_ * eb2 ** 2 / t2 + (n_ - 1) * jnp.log(s2) + jnp.log(t2) + n_ * jnp.log(2 * jnp.pi)), 0.0).sum()
        else: ll = ll + dist.StudentT(NU, 0.0, sg).log_prob(e)
        b = CAP[f] * (1 - ph) * ka * phi_(z) / (s_ * cf[f]) * wq[j] / (CAP[f] * ut * (1 - ut)); own = own + al[f] * b; cross = cross + (1 - al[f]) * b
    sh = jnp.abs(p) <= EPS; be = p < -EPS; free = v & ~sh; logdet = jnp.where(free, jnp.log(own), 0.0).sum() + jnp.log1p(jnp.where(free, cross / own, 0.0).sum(1) / nv[:, 0]).sum()
    u = D - qsum - X - S - B_g; sc = sd_d * LSC; a_ = u / sc; b_ = (u + K_s) / sc; ld_shelf = jax.nn.log_sigmoid(b_) + jax.nn.log_sigmoid(-a_) + jnp.log1p(-jnp.exp(-(K_s / sc))); ld_cont = dist.Logistic(0.0, sc).log_prob(u + K_s * be)
    numpyro.factor("ll", jnp.where(v, ll + dist.StudentT(NU, 0.0, sd_X).log_prob(X - gX) + dist.StudentT(NU, 0.0, sd_S).log_prob(S - gS) + jnp.where(sh, ld_shelf, ld_cont), 0.0).sum() + logdet + llre)
def mean_shift(th, Z): return {f: sum(th.get(f"b{k}_{f}", 0.0) * Z[j, i] for i, k in enumerate("ATW")) for j, f in enumerate(FU)}
def simulate(th, A, sl, rng):
    D, L, co2, v = (A[k][sl] for k in ("D", "L", "co2", "v")); Z = A["Z"][:, :, sl]; n = D.shape; ms = mean_shift(th, Z); e = {}
    for f in FU:
        w = (rng.normal(size=n) * th[f"sg_{f}"] + rng.normal(size=(n[0], 1)) * th[f"tau_{f}"]) if DAYRE else tdist.rvs(NU, size=n, random_state=rng) * th[f"sg_{f}"]; lev = np.zeros(n[0]); m = 0.0
        if MEM:
            for d in range(n[0]): lev[d] = th[f"rho_{f}"] * m; m = lev[d] + th[f"kap_{f}_"] * w[d].mean()
        e[f] = ms[f] + lev[:, None] + w
    eX, eS = (tdist.rvs(NU, size=n, random_state=rng) * th[k] for k in ("sd_X", "sd_S")); thh = dict(th, al_l=th.get("al_l", 1.0), al_c=th.get("al_c", 1.0)); p, pbar = Mh.clear_day(thh, D, L, co2, v, rng.logistic(0.0, th["sd_d"] * LSC, n), e, eX, eS); gX, gS = M.other_curves(th, p, L); return p, Mh.fuels(thh, p, L, co2, pbar, e), gX + eX, gS + eS
def fit(A, sl, dd, warm=600, draws=500, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False); mc.run(jax.random.PRNGKey(seed), jnp.asarray(A["D"][sl]), jnp.asarray(A["L"][sl]), jnp.asarray(A["co2"][sl]), jnp.asarray(A["v"][sl]), jnp.asarray(A["Z"][:, :, sl]), extra_fields=("diverging", "potential_energy"), **dd)
    s = {k.replace("kapm_", "kap_").rstrip("_") if False else k: np.asarray(x) for k, x in mc.get_samples().items()}; summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True); rh = max(float(np.nanmax(x["r_hat"])) for x in summ.values()); ess = min(float(np.nanmin(x["n_eff"])) for x in summ.values()); return s, rh, ess, int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), np.asarray(mc.get_extra_fields(group_by_chain=True)["potential_energy"]).mean(1)
NAMES = [k for k in M.PN] + ([f"b{k}_{f}" for f in FU for k in "ATW"] if COV else []) + ([x for f in FU for x in (f"rho_{f}", f"kap_{f}_")] if MEM else []) + (["al_l", "al_c"] if BLK else []) + ([f"tau_{f}" for f in FU] if DAYRE else [])
if __name__ == "__main__":
    mode = sys.argv[1]; H, ST = frame(); A = grid(H[np.asarray(H.index < "2025-12-01")]); rng = np.random.default_rng(0); allsl = slice(0, len(A["D"])); print(f"{TAG}: COV={int(COV)} MEM={int(MEM)} BLK={int(BLK)}")
    pri = Predictive(model, num_samples=600)(jax.random.PRNGKey(7), jnp.asarray(A["D"][:2]), jnp.asarray(A["L"][:2]), jnp.asarray(A["co2"][:2]), jnp.asarray(A["v"][:2]), jnp.asarray(A["Z"][:, :, :2])); pri = {k: np.asarray(x) for k, x in pri.items()}; q3 = lambda x: " ".join(f"{y:.3g}" for y in np.percentile(x, [5, 50, 95]))
    if mode == "prior":
        sl = slice(120, 240); S_ = [simulate({k: float(pri[k][i]) for k in NAMES}, A, sl, rng) for i in range(40)]; P = np.array([s_[0] for s_ in S_]); po = np.where(A["v"][sl], A["p"][sl], np.nan); L = A["L"][sl]; dq = lambda q: np.corrcoef(q.mean(1)[1:], q.mean(1)[:-1])[0, 1]
        print(f"prior predictive, 40 markets on 120 real days (5-50-95%; observed): median price / cost {q3(np.median((P / L[None]).reshape(40, -1), 1))} ({np.nanmedian(po / L):.2f}); at or below zero {q3((P <= 0).mean((1, 2)))} ({np.nanmean(po <= 0):.3f}); day-to-day correlation of daily output " + "; ".join(f"{QC[f]} {q3([dq(s_[1][f]) for s_ in S_])} ({dq(A[QC[f]][sl]):.2f})" for f in FU) + f"; highest price ct {q3(P.max((1, 2)) / 10)} ({np.nanmax(po) / 10:.0f})")
    if mode == "recover":
        TRUE = dict(x0=3.0, x1=8.0, K_x=4.0, c_z=5.0, s_z=7.0, K_st=4.5, m_st=1.0, s_st=0.4, B_g=3.0, K_s=3.0, sd_X=4.0, sd_S=1.5, sd_d=2.0, a_l=12.0, a_c=40.0, m_g=1.4, phi_l=0.17, kap_l=0.6, s_l=0.3, sg_l=0.25, phi_c=0.05, kap_c=0.6, s_c=0.3, sg_c=0.4, phi_g=0.04, kap_g=0.6, s_g=0.4, sg_g=0.35)
        TRUE.update({f"b{k}_{f}": v_ for f, vs in (("l", (0.15, -0.05, -0.1)), ("c", (0.3, -0.25, -0.15)), ("g", (0.2, -0.5, -0.1))) for k, v_ in zip("ATW", vs)}); TRUE.update({x: v_ for f in FU for x, v_ in ((f"rho_{f}", 0.9), (f"kap_{f}_", 0.6))}); TRUE.update(al_l=0.8, al_c=0.7, tau_l=0.3, tau_c=0.4, tau_g=0.3); TRUE = {k: TRUE[k] for k in NAMES}
        sl = slice(100, 400); p, q, X, S = simulate(TRUE, A, sl, rng); s, rh, ess, div, pe = fit(A, sl, Mh.pack(dict(p=p, X=X, S=S), q), warm=500, draws=400); nin = 0
        print(f"simulate and recover: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in NAMES:
            qq = np.percentile(s[k], [5, 50, 95]); okk = qq[0] <= TRUE[k] <= qq[2]; nin += okk
            if not okk or k.startswith(("al_", "rho_", "tau_")) or k.endswith("_") : print(f"   {k:7s} true {TRUE[k]:6.2f}  recovered {qq[1]:6.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})  {'ok' if okk else 'OUTSIDE the 90% band'}")
        print(f"   {nin} of {len(NAMES)} inside their 90% bands")
    if mode == "fit":
        s, rh, ess, div, pe = fit(A, allsl, Mh.pack(A), seed=int(os.environ.get("SEED", 0))); print(f"{TAG} fitted: worst r-hat {rh:.3f}, smallest effective sample {ess:.0f} of {len(s['x0'])}, divergent {div}; log density by chain {np.round(-pe)}")
        for k in NAMES:
            if k in M.PN and not k.startswith(("sg_", "a_", "m_g", "s_l", "s_c", "s_g")): continue
            qq = np.percentile(s[k], [5, 50, 95]); print(f"   {k:7s} {qq[1]:7.2f}  ({qq[0]:.2f} .. {qq[2]:.2f})")
        np.savez_compressed(f"{SP}/out/{TAG}_post.npz", **{k: s[k] for k in NAMES}, rhat=rh, ess=ess, div=div, lp=-pe, **{f"st_{c}": np.array(v_) for c, v_ in ST.items()})
