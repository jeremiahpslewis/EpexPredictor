"""Compact hierarchical Bayesian price model for Germany (research benchmark, not part of the app).

price[t] = (gas/gas_ref)^g * (c(RL) + a[daytype, hour] + b[hour] * z(RL)) + w[hour, quarter] + u[day] + eps
  c:      P-spline in the residual load RL (GW), second-difference random-walk prior
  g:      gas pass-through exponent (1 = proportional), so the EUR/MWh effect of gas grows with the price level
  a:      hour x daytype intercepts, partially pooled around hour and daytype effects
  b:      hour-specific linear RL slope deviation, pooled
  w:      within-hour shape (quarter hours), fixed: training means of the deviation from the hourly mean
  u:      day effect, independent across days in the fit (models the day-level dependence of the hourly rows).
          Its persistence rho is estimated afterwards, per posterior draw, as the lag-1 autocorrelation of the fitted day
          effects (two-step approximation); MODE ar carries rho * last known day's effect forward, nocarry does not.
  eps:    Student-t, log scale quadratic in z(RL)
Fitted on hourly means of the last TRAIN_DAYS days before the forecast origin, with the residual load published
by the TSOs. Forecasts use the own residual-load forecast, with past errors of that forecast resampled.
usage: bayes.py NAME PERIOD MODE      PERIOD autumn|summer, MODE ar (AR(1) carry) | nocarry
"""
import os, sys, time, json
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import BSpline
import live_common as lc, dense_common as dc

numpyro.set_host_device_count(4)
NAME, PERIOD, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
TRAIN_DAYS, NK, REFIT_DAYS = int(os.environ.get("TRAIN_DAYS", 365)), 8, int(os.environ.get("REFIT_DAYS", 7))
WARM, DRAWS, CHAINS = int(os.environ.get("WARM", 400)), int(os.environ.get("DRAWS", 300)), 4
LIMIT = int(os.environ.get("LIMIT", 0))
OUT = f"{lc.SP}/out/bayes_{NAME}"; os.makedirs(OUT, exist_ok=True)

F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx)
local = idx.tz_convert("Europe/Berlin")
y_all = D["price"].to_numpy(float); gas_all = D["gasprice"].to_numpy(float); rl_all = fc["rl"].to_numpy(float)
hour_all = local.hour.to_numpy(); quarter_all = (local.minute // 15).to_numpy()
daytype_all = np.where((D["holiday"].to_numpy() >= 0.5) | (D["weekday"].to_numpy() == 6), 2, np.where(D["weekday"].to_numpy() == 5, 1, 0))
day_all = np.asarray(local.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64))

if PERIOD == "autumn":
    cuts = lc.cutoffs(idx, "2026-09-03", "2026-11-07", 1); G = np.load(f"{lc.SP}/out/gen_fc_app.npz"); prior_err = np.load(f"{lc.SP}/out/gen_ssn.npz")
else:
    cuts = lc.cutoffs(idx, "2026-04-01", "2026-08-27", 3); G = np.load(f"{lc.SP}/out/gen_ssn.npz"); prior_err = None
assert len(G["cutoffs"]) == len(cuts)
rl_hat = G["rl_pes"].astype(float); rl_err = G["rl_true"].astype(float) - rl_hat          # error paths of the own forecast, per origin
pool0 = (prior_err["rl_true"] - prior_err["rl_pes"]).astype(float) if prior_err is not None else np.zeros((0, lc.HZ))

def basis(x, knots):
    t = np.r_[[knots[0]] * 3, knots, [knots[-1]] * 3]
    return BSpline.design_matrix(np.clip(x, knots[0], knots[-1]), t, 3).toarray()

def rw2_basis(nb):
    """columns span the second-order random walk part of nb spline coefficients, orthogonal to level and slope"""
    M = np.tril(np.ones((nb, nb))) @ np.tril(np.ones((nb, nb))); M = M[:, 2:]
    X = np.c_[np.ones(nb), np.arange(nb) - (nb - 1) / 2]
    M = M - X @ np.linalg.lstsq(X, M, rcond=None)[0]
    return M / np.linalg.norm(M, axis=0), X[:, 1] / np.abs(X[:, 1]).max()

def model(Bs, gasn, z, hour, dtype, dayi, ndays, dday, y=None, carry=True):
    nb = Bs.shape[1]
    M, lin = rw2_basis(nb)
    def pspline(name):
        tau = numpyro.sample(f"tau_{name}", dist.HalfNormal(5.0))
        d0 = numpyro.sample(f"{name}0", dist.Normal(10, 10)); d1 = numpyro.sample(f"{name}1", dist.Normal(0, 10))
        e = numpyro.sample(f"{name}_e", dist.Normal(0, 1).expand([nb - 2]))
        return d0 + d1 * lin + tau * (M @ e)                          # level and slope at the centre, smooth rest
    c = pspline("c"); g = numpyro.sample("g", dist.Normal(1.0, 0.5))
    mh = numpyro.sample("mu_h", dist.Normal(0, 5).expand([23])); mu_h = jnp.r_[mh, -mh.sum()]       # sums to zero
    mu_d = numpyro.sample("mu_d", dist.Normal(0, 5).expand([2]))
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(2.0)); a_e = numpyro.sample("a_e", dist.Normal(0, 1).expand([3, 24]))
    a = mu_h[None, :] + jnp.r_[0.0, mu_d][:, None] + tau_a * a_e
    tau_b = numpyro.sample("tau_b", dist.HalfNormal(2.0)); be = numpyro.sample("b_e", dist.Normal(0, 1).expand([23])); b = tau_b * jnp.r_[be, -be.sum()]
    s = numpyro.sample("s", dist.Normal(0, 1).expand([3])); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4))
    sig_u = numpyro.sample("sig_u", dist.HalfNormal(3.0))
    e_u = numpyro.sample("e_u", dist.Normal(0, 1).expand([ndays]))
    gm = jnp.stack([jnp.sum(jnp.where(dday == t, e_u, 0)) / jnp.maximum(jnp.sum(dday == t), 1) for t in range(3)])
    u = sig_u * (e_u - gm[dday])                                      # centred within day type, so mu_d carries the weekend offsets
    mean = gasn ** g * (Bs @ c + a[dtype, hour] + b[hour] * z) + u[dayi]
    scale = jnp.exp(s[0] + s[1] * z + s[2] * z ** 2)
    numpyro.sample("y", dist.StudentT(nu, mean, scale), obs=y)

def hourly(lo, hi):
    """hourly means over rows lo..hi (rows aligned to whole hours)"""
    sl = slice(lo, hi); n = (hi - lo) // 4 * 4; r = lambda v: v[sl][:n].reshape(-1, 4)
    return {"y": np.nanmean(r(y_all), 1), "rl": r(rl_all).mean(1), "gas": r(gas_all)[:, 0], "hour": r(hour_all)[:, 0], "dtype": r(daytype_all)[:, 0], "day": r(day_all)[:, 0]}

done = {f.split(".")[0] for f in os.listdir(OUT)}; fit = None; t_start = time.time()
order = list(range(len(cuts)))[: LIMIT or None]
for j in order:
    d0 = cuts[j]; tag = str(idx[d0].date())
    if tag in done: continue
    if fit is None or j - fit["j"] >= (REFIT_DAYS if PERIOD == "autumn" else max(1, REFIT_DAYS // 3)):
        lo = d0 - TRAIN_DAYS * 96; lo += (-lo) % 4
        H = hourly(lo, d0); ok = np.isfinite(H["y"]) & np.isfinite(H["rl"]) & np.isfinite(H["gas"])
        H = {k: v[ok] for k, v in H.items()}
        knots = np.quantile(H["rl"], np.linspace(0, 1, NK - 2)); knots[0] -= 5; knots[-1] += 5
        zm, zs = H["rl"].mean(), H["rl"].std(); days = np.unique(H["day"]); dayi = np.searchsorted(days, H["day"])
        # within-hour shape, training means of the last 90 days
        lo90 = d0 - 90 * 96; dev = y_all[lo90:d0] - pd.Series(y_all[lo90:d0]).groupby(np.arange(d0 - lo90) // 4).transform("mean").to_numpy()
        w = pd.DataFrame({"h": hour_all[lo90:d0], "q": quarter_all[lo90:d0], "d": dev}).groupby(["h", "q"])["d"].mean().unstack().reindex(range(24)).fillna(0).to_numpy()
        gref = float(H["gas"].mean())
        args = dict(Bs=jnp.asarray(basis(H["rl"], knots)), gasn=jnp.asarray(H["gas"] / gref), z=jnp.asarray((H["rl"] - zm) / zs), hour=jnp.asarray(H["hour"]), dtype=jnp.asarray(H["dtype"]),
                    dayi=jnp.asarray(dayi), ndays=len(days), dday=jnp.asarray(pd.Series(H["dtype"]).groupby(dayi).first().to_numpy()), carry=(MODE == "ar"))
        t0 = time.time()
        mc = MCMC(NUTS(model, target_accept_prob=0.85, init_strategy=numpyro.infer.init_to_median(num_samples=15)), num_warmup=WARM, num_samples=DRAWS, num_chains=CHAINS, progress_bar=False, chain_method="parallel")
        mc.run(jax.random.PRNGKey(j), y=jnp.asarray(H["y"]), extra_fields=("diverging", "potential_energy"), **args)
        P = {k: np.asarray(v) for k, v in mc.get_samples().items()}
        dd = np.asarray(args["dday"]); U = P["sig_u"][:, None] * P["e_u"]
        for t in range(3): U[:, dd == t] -= U[:, dd == t].mean(1, keepdims=True)
        Uc = U - U.mean(1, keepdims=True)
        P["rho"] = np.clip((Uc[:, 1:] * Uc[:, :-1]).sum(1) / (Uc ** 2).sum(1), -0.5, 0.99)
        P["sig_u"] = U.std(1) * np.sqrt(1 - P["rho"] ** 2)          # innovation scale of the AR(1) for the forecast
        diag = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True)
        rhat = max(float(np.nanmax(diag[k]["r_hat"])) for k in diag); ess = min(float(np.nanmin(diag[k]["n_eff"])) for k in diag)
        worst = sorted(((float(np.nanmax(diag[k]["r_hat"])), k) for k in diag), reverse=True)[:4]
        if os.environ.get("CHAINS_DEBUG"):
            Sc = {k: np.asarray(v) for k, v in mc.get_samples(group_by_chain=True).items()}
            for k in ("g", "mu_d", "sig_u", "nu", "c0", "tau_a"):
                if k in Sc: print("   ", k, "per chain:", np.round(Sc[k].reshape(CHAINS, DRAWS, -1).mean(1)[:, :3].ravel(), 2), flush=True)
            print("    log density per chain:", np.round(np.asarray(mc.get_extra_fields(group_by_chain=True).get("potential_energy", np.zeros((CHAINS, 1)))).mean(1), 0), flush=True)
        div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
        fit = dict(j=j, gref=gref, P=P, knots=knots, zm=zm, zs=zs, w=w, days=days, H=H, args=args, secs=time.time() - t0, rhat=rhat, ess=ess, div=div, n=len(H["y"]))
        print(f"{tag} fit {fit['secs']:.0f}s rows {fit['n']} max r_hat {rhat:.3f} min ess {ess:.0f} divergences {div} worst {worst}" + "".join(f" {k}={P[k].mean():.2f}" for k in ("rho", "sig_u", "nu", "g") if k in P), flush=True)
    P, knots, zm, zs, w = fit["P"], fit["knots"], fit["zm"], fit["zs"], fit["w"]; S = len(P["nu"])

    def mean_fn(rl, gas, hour, dtype):
        Bs = basis(rl.ravel(), knots).reshape(*rl.shape, -1) if rl.ndim > 1 else basis(rl, knots)
        M, lin = rw2_basis(len(knots) + 2)
        c = P["c0"][:, None] + P["c1"][:, None] * lin[None, :] + P["tau_c"][:, None] * (P["c_e"] @ M.T)
        mh = np.c_[P["mu_h"], -P["mu_h"].sum(1)]
        a = mh[:, None, :] + np.c_[np.zeros(S), P["mu_d"]][:, :, None] + P["tau_a"][:, None, None] * P["a_e"]
        b = P["tau_b"][:, None] * np.c_[P["b_e"], -P["b_e"].sum(1)]
        z = (rl - zm) / zs
        Bc = Bs @ c.T if rl.ndim == 1 else np.einsum("stb,sb->ts", Bs, c)   # (T, S)
        zz = z.T if rl.ndim > 1 else z[:, None]
        return (gas / fit["gref"])[:, None] ** P["g"][None, :] * (Bc + a[:, dtype, hour].T + b[:, hour].T * zz), zz
    if os.environ.get("RESID_ACF") and fit["j"] == j:
        Ht = fit["H"]; mu_t, _ = mean_fn(Ht["rl"], Ht["gas"], Ht["hour"], Ht["dtype"])
        dd = np.asarray(fit["args"]["dday"]); U = P["sig_u"][:, None] * 0 + (P["e_u"] * 0)  # placeholder shape
        Uf = (np.asarray(mc.get_samples()["sig_u"])[:, None] * np.asarray(mc.get_samples()["e_u"]))
        for t in range(3): Uf[:, dd == t] -= Uf[:, dd == t].mean(1, keepdims=True)
        di = np.asarray(fit["args"]["dayi"]); resid = Ht["y"] - np.median(mu_t + Uf[:, di].T, 1)
        same = di[1:] == di[:-1]; r0, r1 = resid[:-1][same], resid[1:][same]
        print(f"training residuals: lag-1 hour autocorrelation within days {np.corrcoef(r0, r1)[0, 1]:.2f}, lag-3 {np.corrcoef(resid[:-3][di[3:] == di[:-3]], resid[3:][di[3:] == di[:-3]])[0, 1]:.2f}", flush=True)
    # state of the day effect on the last known day (D+1): plug-in per draw, mean residual of that day's hours
    last_day = day_all[d0 - 1]; m = (day_all[:d0] == last_day); rows = np.flatnonzero(m)
    Hl = {"y": y_all[rows], "rl": rl_all[rows], "gas": gas_all[rows], "hour": hour_all[rows], "dtype": daytype_all[rows]}
    mu_l, _ = mean_fn(Hl["rl"], Hl["gas"], Hl["hour"], Hl["dtype"])
    u_last = np.nanmean(Hl["y"][:, None] - w[Hl["hour"], quarter_all[rows]][:, None] - mu_l, 0)    # (S,)
    # forecast rows
    T = np.arange(d0, d0 + lc.HZ); gas_f = np.full(lc.HZ, gas_all[d0 - lc.ISSUE]); hours = hour_all[T]; dtyp = daytype_all[T]
    k_ahead = (day_all[T] - last_day).astype(int)                       # 1 = D+2, 2 = D+3
    rng = np.random.default_rng(j)
    pool = np.vstack([pool0, rl_err[:j]]) if j > 0 or len(pool0) else np.zeros((0, lc.HZ))
    if len(pool) < 3: pool = rng.normal(0, 2.7, (50, lc.HZ))            # no earlier errors: assume 2.7 GW (stated)
    errs = pool[rng.integers(0, len(pool), S)]                          # whole error paths, keeps their autocorrelation
    rl_draw = rl_hat[j][None, :] + errs                                 # (S, T)
    res = {}
    for label, rlx in (("exact", np.repeat(rl_hat[j][None, :], S, 0)), ("prop", rl_draw)):
        mu, zz = mean_fn(rlx, gas_f, hours, dtyp)                       # (T, S)
        rho = P["rho"] if MODE == "ar" else np.zeros(S)
        e1, e2 = rng.normal(size=S), rng.normal(size=S)
        u1 = rho * u_last + P["sig_u"] * e1; u2 = rho * u1 + P["sig_u"] * e2
        u = np.where(k_ahead[:, None] == 1, u1[None, :], u2[None, :])
        scale = np.exp(P["s"][:, 0][None, :] + P["s"][:, 1][None, :] * zz + P["s"][:, 2][None, :] * zz ** 2)
        draws = mu + u + w[hours, quarter_all[T]][:, None] + scale * rng.standard_t(P["nu"][None, :], size=mu.shape)
        res[label] = np.percentile(draws, [5, 10, 50, 90, 95], axis=1)  # (5, T)
        if label == "exact": res["mean_median"] = np.median(mu + u + w[hours, quarter_all[T]][:, None], 1)
    # response curve: EUR/MWh per GW at this origin's forecast state (+/-1 GW, same draws), first forecast day
    mu_p, _ = mean_fn(rl_hat[j] + 1, gas_f, hours, dtyp); mu_m, _ = mean_fn(rl_hat[j] - 1, gas_f, hours, dtyp)
    slope = 10 * (mu_p - mu_m) / 2                                       # (T, S) EUR/MWh per GW
    mu_g, _ = mean_fn(rl_hat[j], gas_f * 1.1, hours, dtyp); mu_0, _ = mean_fn(rl_hat[j], gas_f, hours, dtyp)
    np.savez_compressed(f"{OUT}/{tag}.npz", q_exact=res["exact"], q_prop=res["prop"], actual=y_all[T], rl_hat=rl_hat[j],
                        slope_q=np.percentile(slope, [5, 50, 95], 1), gas10_q=np.percentile(10 * (mu_g - mu_0), [5, 50, 95], 1),
                        fit=np.array([fit["secs"], fit["rhat"], fit["ess"], fit["div"], fit["n"], fit["j"] == j]))
    print(f"{tag} forecast done ({time.time() - t_start:.0f}s total)", flush=True)
print("finished", flush=True)
