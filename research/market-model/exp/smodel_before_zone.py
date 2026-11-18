"""Structural Bayesian price model for Germany (research harness).

price[EUR/MWh] = b(x) + gas * h(x) + AR(1) error
  x    effective residual load [GW] = RL + calendar shift[daytype, hour] + kappa . Europe-wide weather + weekly drift
  b    base cost curve, monotone piecewise linear in x (segment slopes exp(.), random-walk prior on the log slopes)
  h    gas pass-through [EUR/MWh electricity per EUR/MWh gas], monotone piecewise linear, zero at the surplus end
  both continue beyond the last knot with the slope of the last segment (extrapolation in scarcity)
  error: Student-t, scale depends on gas level and x. Hours are treated as independent for the fit, with the
  log-likelihood down-weighted by LW (default 1/8) for their dependence; an AR error term let a persistent level bias
  pass as autocorrelation. What the last known day's errors say about the next days is estimated in a second step:
  per target hour, a ridge regression of the error on the error of the same hour and of the last hour of the last known day.

Fit on hourly means of the TRAIN_DAYS before the forecast origin with the published residual load.
FIT=map (default): posterior mode (fast, for development). FIT=nuts: full sampling.
usage: struct.py NAME PERIOD SHARD NSHARDS     env: EU=1/0, TRAIN_DAYS, STRIDE, FIT
"""
import os, sys, time
if os.environ.get("FIT", "map") == "map": os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoDelta
import live_common as lc, dense_common as dc

NAME, PERIOD, SHARD, NSHARDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
USE_EU = os.environ.get("EU", "1") == "1"; TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", 540)); STRIDE = int(os.environ.get("STRIDE", 1))
FIT = os.environ.get("FIT", "map"); STEPS = int(os.environ.get("STEPS", 4000)); DRIFT = os.environ.get("DRIFT", "x")
ADD = os.environ.get("ADD", "0") == "1"       # additional price premium by hour (scaled with gas)
LW = float(os.environ.get("LW", 0.125)); TAUW = float(os.environ.get("TAUW", 0.5))
KNOWN_LAG = int(os.environ.get("KNOWN_LAG", 1))   # days before the first forecast day for which the published figures are missing at issue
HL = float(os.environ.get("HL", 0))          # half-life [days] of the weight of older training data, 0 = equal weights
SCALE_RW = float(os.environ.get("SCALE_RW", 0.02))   # weekly random-walk step of the price-scale drift (DRIFT s / xs)
NB = os.environ.get("NB", "")              # "sum": the neighbours' residual load (one regional figure) shifts the effective residual load
HX = os.environ.get("HX", "0") == "1"       # hour-specific extra shift of the effective residual load in tight conditions
TAUW_FIX = float(os.environ.get("TAUW_FIX", -1)); BTOP = float(os.environ.get("BTOP", -1))   # weekly drift scale fixed [GW]; base curve flat above BTOP GW
OUT = f"{lc.SP}/out/struct_{NAME}"; os.makedirs(OUT, exist_ok=True)
CHAINS = int(os.environ.get("CHAINS", 4)); ONLY = os.environ.get("ONLY", "")       # ONLY: comma-separated origin dates to run
if FIT == "nuts": numpyro.set_host_device_count(CHAINS)
K = np.arange(-10.0, 70.1, 5.0); NSEG = len(K) - 1
# DRIFT containing "c": the curve itself drifts, per segment of the effective residual load, as a trend with changepoints
# (Prophet's move, applied to the curve): values at knots every CK_DAYS days, linear in between, changes between knots with
# a Laplace prior (few, distinct changes) or a normal one; anchored at zero at the last knot, so the curve is today's curve.
CK = int(os.environ.get("CK_DAYS", 28)); CTAU = float(os.environ.get("CTAU", 3.0)); CPRIOR = os.environ.get("CPRIOR", "laplace")
CC = np.array([float(v) for v in os.environ.get("CCENTRES", "0,15,30,45").split(",")]); CBW = float(os.environ.get("CBW", 8.0))
def cbasis(x, xp):
    phi = xp.exp(-0.5 * ((x[..., None] - CC) / CBW) ** 2); return phi / (phi.sum(-1, keepdims=True) + 1e-9)
def cdrift(e, xp):
    steps = CTAU * e; return xp.concatenate([-xp.cumsum(steps[::-1], 0)[::-1], xp.zeros((1, len(CC)))])      # (knots, segments)
def cvalue(e, x, ct, xp):
    Dk = cdrift(e, xp); k0, w = ct; return (cbasis(x, xp) * (Dk[k0] * (1 - w)[:, None] + Dk[xp.minimum(k0 + 1, len(Dk) - 1)] * w[:, None])).sum(-1)

F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx); local = idx.tz_convert("Europe/Berlin")
EUC = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]
y_all = D["price"].to_numpy(float) * 10; gas_all = D["gasprice"].to_numpy(float); rl_all = fc["rl"].to_numpy(float); eu_all = D[EUC].to_numpy(float)
hour_all = local.hour.to_numpy(); quarter_all = (local.minute // 15).to_numpy()
daytype_all = np.where((D["holiday"].to_numpy() >= 0.5) | (D["weekday"].to_numpy() == 6), 2, np.where(D["weekday"].to_numpy() == 5, 1, 0))
day_all = np.asarray(local.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64))
I = np.load(f"{lc.SP}/out/inputs_{PERIOD}.npz"); cuts = I["cuts"]
if NB:
    _nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); nb_all = _nb[[c for c in _nb.columns if c.startswith("rl_")]].sum(axis=1, min_count=8).to_numpy(float)
    NBT = f"{lc.SP}/out/tsnb_{PERIOD}"            # Timestral forecasts of the regional figure (own-forecast arms and D+1)
else: nb_all = np.zeros(len(idx))
RLP = np.load(f"{lc.SP}/out/" + os.environ["RLFILE"] + ".npz")["rl_pes"] if os.environ.get("RLFILE") else (I["rl_pes"] if "rl_pes" in I.files else I["rl_true"])   # own residual-load forecast
MORE = {k.split("=")[0]: np.load(f"{lc.SP}/out/{k.split('=')[1]}.npz")["rl_pes"] for k in os.environ.get("RLFILES", "").split(",") if k}      # further own forecasts: arm=file,...
FALLBACK = {}
if os.environ.get("FALLBACK"):   # arm=file pairs of generation files from another period, for the error paths of the first forecasts
    for kv in os.environ["FALLBACK"].split(","):
        gf = np.load(f"{lc.SP}/out/{kv.split('=')[1]}.npz"); FALLBACK[kv.split("=")[0]] = (gf["rl_true"] - gf["rl_pes"]).astype(float)
D1F = np.load(f"{lc.SP}/out/" + os.environ["D1FILE"] + ".npz")["rl_d1"] if os.environ.get("D1FILE") else None                                 # own forecast for D+1

# STO="P,E,ETA": storage mechanism. The auction clears a day at once; a storage fleet (power P GW, energy E GWh charged per
# day, round-trip efficiency ETA) charges in the cheapest hours of the day and discharges in the dearest, as long as
# price(dearest level) * ETA > price(cheapest level). Its net demand shifts the effective residual load. The dispatch is
# computed on the curve of a reference fit without storage (STOREF); how much of that theoretical shift shows in prices is
# estimated: a_s[0] for charging, a_s[1] for discharging, prior Normal(0.5, 0.5).
STO = os.environ.get("STO", "")
if STO: STO_P, STO_E, STO_ETA = [float(v) for v in STO.split(",")]; STOREF = os.environ["STOREF"]
def sto_level(x, q, up):
    lo, hi = x.min() - STO_P - 1, x.max() + STO_P + 1
    for _ in range(40):
        L = 0.5 * (lo + hi); v = np.minimum(STO_P, np.maximum(L - x, 0)).sum() if up else np.minimum(STO_P, np.maximum(x - L, 0)).sum()
        if (v < q) == up: lo = L
        else: hi = L
    return 0.5 * (lo + hi)
def sto_dispatch(x, gas, val, emax):
    ok = lambda Ll, Lh: val(np.array([Lh]), gas)[0] * STO_ETA > val(np.array([Ll]), gas)[0] and Lh > Ll
    sol = lambda q: (sto_level(x, q, True), sto_level(x, STO_ETA * q, False)); lo, hi = 0.0, emax; Ll, Lh = sol(hi)
    if not ok(Ll, Lh):
        for _ in range(25):
            q = 0.5 * (lo + hi); Ll, Lh = sol(q)
            if ok(Ll, Lh): lo = q
            else: hi = q
        Ll, Lh = sol(lo)
    return np.minimum(STO_P, np.maximum(Ll - x, 0)) - np.minimum(STO_P, np.maximum(x - Lh, 0))
def sto_shift(x, gas, day, val):
    """x per hour (or quarter hour), its local day -> storage shift, dispatched per local day on hourly means"""
    x = np.asarray(x, float); out = np.zeros(len(x)); gas = np.broadcast_to(np.asarray(gas, float), x.shape)
    for dd in np.unique(day):
        m = np.flatnonzero(day == dd)
        if np.isfinite(x[m]).all() and len(m) >= 6: out[m] = sto_dispatch(x[m], float(gas[m][0]), val, STO_E * min(1.0, len(m) / 24))
    return out
def ref_val(zr):
    g = zr["grid"].astype(float); b = zr["b_curve"].astype(float); h = zr["h_curve"].astype(float); db = np.diff(b) / np.diff(g); dh = np.diff(h) / np.diff(g)
    return lambda x, gas: np.interp(np.clip(x, g[0], g[-1]), g, b) + gas * np.interp(np.clip(x, g[0], g[-1]), g, h) + np.minimum(x - g[0], 0) * db[0] + np.maximum(x - g[-1], 0) * (db[-1] + gas * dh[-1])

def pl_np(x, f0, slopes, left):
    seg = np.clip(x[..., None] - K[:-1], 0.0, K[1:] - K[:-1])
    return f0 + left * np.minimum(x - K[0], 0.0) + seg @ slopes + slopes[-1] * np.maximum(x - K[-1], 0.0)

def pl(x, f0, slopes, left):
    seg = jnp.clip(x[:, None] - K[None, :-1], 0.0, (K[1:] - K[:-1])[None, :])
    return f0 + left * jnp.minimum(x - K[0], 0.0) + seg @ slopes + slopes[-1] * jnp.maximum(x - K[-1], 0.0)

def unpack(p):
    """named parameter dict (jax or numpy) -> curves and shifts"""
    xp = jnp if isinstance(p["b0"], jnp.ndarray) else np
    lb = p["lb0"] + xp.concatenate([xp.zeros(1), xp.cumsum(p["tb"] * p["lbe"])]); lh = p["lh0"] + xp.concatenate([xp.zeros(1), xp.cumsum(p["th"] * p["lhe"])])
    mu_h = xp.concatenate([p["mu_h"], -p["mu_h"].sum(keepdims=True)]); cal = p["x0"] + mu_h[None, :] + xp.concatenate([xp.zeros(1), p["mu_d"]])[:, None] + xp.concatenate([xp.zeros((1, 24)), p["tau_c"] * p["c_e"]])
    if "w_e" in p:
        steps = p["tau_w"] * p["w_e"]; drift = xp.concatenate([-xp.cumsum(steps[::-1])[::-1], xp.zeros(1)])     # anchored at the last week
    else: drift = xp.zeros(1)
    return xp.exp(lb), xp.exp(lh), cal, drift

def sdrift(v_e, xp):
    steps = SCALE_RW * v_e; return xp.concatenate([-xp.cumsum(steps[::-1])[::-1], xp.zeros(1)])

def model(d, y=None):
    p = {"b0": numpyro.sample("b0", dist.Normal(-20.0, 40.0)), "lb0": numpyro.sample("lb0", dist.Normal(np.log(3.0), 1.5)),
         "lbe": numpyro.sample("lbe", dist.Normal(0, 1).expand([NSEG - 1])), "tb": numpyro.sample("tb", dist.HalfNormal(0.7)),
         "lh0": numpyro.sample("lh0", dist.Normal(np.log(0.03), 1.5)), "lhe": numpyro.sample("lhe", dist.Normal(0, 1).expand([NSEG - 1])), "th": numpyro.sample("th", dist.HalfNormal(0.7)),
         "x0": numpyro.sample("x0", dist.Normal(0, 5)), "mu_h": numpyro.sample("mu_h", dist.Normal(0, 5).expand([23])), "mu_d": numpyro.sample("mu_d", dist.Normal(0, 5).expand([2])),
         "tau_c": numpyro.sample("tau_c", dist.HalfNormal(2.0)), "c_e": numpyro.sample("c_e", dist.Normal(0, 1).expand([2, 24])),
         }
    if "x" in DRIFT or DRIFT == "p": p.update({"tau_w": (numpyro.sample("tau_w", dist.HalfNormal(TAUW)) if TAUW_FIX < 0 else TAUW_FIX), "w_e": numpyro.sample("w_e", dist.Normal(0, 1).expand([d["nw"] - 1]))})
    if "s" in DRIFT: p["v_e"] = numpyro.sample("v_e", dist.Normal(0, 1).expand([d["nw"] - 1]))
    if "c" in DRIFT: cd_e = numpyro.sample("cd_e", (dist.Laplace(0, 1) if CPRIOR == "laplace" else dist.Normal(0, 1)).expand([d["nk"] - 1, len(CC)]))
    sb, sh, cal, drift = unpack(p)
    if BTOP >= 0:   # above BTOP GW gas plants set the price: the base curve is flat there and the rise goes through the pass-through
        top = np.flatnonzero(K[:-1] >= BTOP); numpyro.factor("btop", dist.Normal(np.log(0.1), 0.5).log_prob(jnp.log(sb[top])).sum())
    x = d["rl"] + cal[d["dtype"], d["hour"]]
    if HX:
        hx = numpyro.sample("hx", dist.Normal(0, 2).expand([24])); x = x + hx[d["hour"]] * jnp.maximum(d["rl"] - 30.0, 0.0) / 10.0
    if "x" in DRIFT: x = x + drift[d["week"]]
    if USE_EU:
        kappa = numpyro.sample("kappa", dist.Normal(0, 3).expand([4])); x = x + d["eu"] @ kappa
    if NB:
        w_nb = numpyro.sample("w_nb", dist.Normal(0.0, 0.5)); x = x + w_nb * d["nb"]
    if STO:
        a_s = numpyro.sample("a_s", dist.Normal(0.5, 0.5).expand([2])); x = x + a_s[0] * d["sc"] - a_s[1] * d["sd"]
    mean = pl(x, p["b0"], sb, sb[0]) + d["gas"] * pl(x, 0.0, sh, 0.0)
    if DRIFT == "p": mean = mean + 5.0 * drift[d["week"]]
    if "s" in DRIFT: mean = mean * jnp.exp(sdrift(p["v_e"], jnp)[d["week"]])
    if "c" in DRIFT: mean = mean + cvalue(cd_e, x, (d["ck0"], d["cw"]), jnp)
    if ADD:
        ah = numpyro.sample("a_h", dist.Normal(0, 3).expand([24])); mean = mean + (ah - ah.mean())[d["hour"]] * d["gas"] / d["gref"]
    s = numpyro.sample("s", dist.Normal(0, 1).expand([4])); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4))
    zx = (x - 25.0) / 15.0; scale = 10.0 * jnp.exp(s[0] + s[1] * jnp.log(d["gas"] / d["gref"]) + s[2] * zx + s[3] * zx ** 2)
    numpyro.factor("ll", LW * (d["wt"] * dist.StudentT(nu, 0.0, scale).log_prob(y - mean)).sum())

def hourly(lo, hi):
    n = (hi - lo) // 4 * 4; r = lambda v: v[lo:lo + n].reshape(-1, 4, *v.shape[1:])
    return {"y": np.nanmean(r(y_all), 1), "rl": r(rl_all).mean(1), "gas": r(gas_all)[:, 0], "hour": r(hour_all)[:, 0], "dtype": r(daytype_all)[:, 0], "day": r(day_all)[:, 0], "eu": r(eu_all).mean(1), "nb": r(nb_all).mean(1), "pos": lo + 4 * np.arange(n // 4)}

def predict(p, eun, rl, gas, hour, dtype, eu, gref, week=None, nb=None, ct=None, sto=None):
    sb, sh, cal, drift = unpack(p); dr = 0.0 if len(drift) == 1 else (drift[-1] if week is None else drift[week])
    x = rl + cal[dtype, hour] + (dr if "x" in DRIFT else 0.0)
    if HX: x = x + p["hx"][hour] * np.maximum(rl - 30.0, 0.0) / 10.0
    if USE_EU: x = x + ((eu - eun[0]) / eun[1]) @ p["kappa"]
    if NB: x = x + p["w_nb"] * ((nb if nb is not None else NB_FC) - NBM[0])
    if STO and sto is not None: x = x + p["a_s"][0] * np.maximum(sto, 0) - p["a_s"][1] * np.maximum(-sto, 0)
    m = pl_np(x, p["b0"], sb, sb[0]) + gas * pl_np(x, 0.0, sh, 0.0)
    if DRIFT == "p": m = m + 5.0 * dr
    if "s" in DRIFT and week is not None: m = m * np.exp(sdrift(p["v_e"], np)[week])
    if "c" in DRIFT and ct is not None: m = m + cvalue(p["cd_e"], x, ct, np)
    if ADD: m = m + (p["a_h"] - p["a_h"].mean())[hour] * gas / gref
    return m, x

mine = [j for j in range(0, len(cuts), STRIDE)][SHARD::NSHARDS]
for j in mine:
    d0 = int(cuts[j]); tag = str(idx[d0].date())
    if os.path.exists(f"{OUT}/{tag}.npz") or (ONLY and tag not in ONLY.split(",")): continue
    t0 = time.time(); lo = d0 - TRAIN_DAYS * 96; lo += (-lo) % 4; lo = max(lo, 4 * 96)
    # the published wind and solar figures for the day before the first forecast day (D+1) appear only in the evening of
    # the issue day, so nothing of D+1 that needs them is used: training and the error history end with the issue day D
    d1 = d0 - KNOWN_LAG * 96
    H = hourly(lo, d1); ok = np.isfinite(H["y"]) & np.isfinite(H["rl"]) & np.isfinite(H["gas"]) & np.isfinite(H["eu"]).all(1) & np.isfinite(H["nb"])
    H = {k: v[ok] for k, v in H.items()}; eun = (H["eu"].mean(0), H["eu"].std(0)); gref = float(H["gas"].mean()); NBM = (float(H["nb"].mean()),)
    week = (H["day"] - H["day"].min()) // 7; cont = np.r_[False, np.diff(H["pos"]) == 4]
    nk = int(np.ceil((H["day"].max() - H["day"].min()) / CK)) + 1; tk = np.maximum((nk - 1) - (H["day"].max() - H["day"]) / CK, 0.0); ck0 = np.minimum(np.floor(tk).astype(int), nk - 1); cw = tk - ck0
    sh_tr = None
    if STO:
        zr = np.load(f"{lc.SP}/out/struct_{STOREF}/{tag}.npz"); rval = ref_val(zr)
        hq = lambda v: np.asarray(v, float)                                            # hourly means of quarter-hour values, per (day, hour) of the slots
        xr = H["rl"] + zr["cal"][H["dtype"], H["hour"]] + ((H["eu"] - eun[0]) / eun[1]) @ zr["kappa"]; sh_tr = sto_shift(xr, H["gas"], H["day"], rval)
    d = {"rl": jnp.asarray(H["rl"]), "gas": jnp.asarray(H["gas"]), "hour": jnp.asarray(H["hour"]), "dtype": jnp.asarray(H["dtype"]), "week": jnp.asarray(week), "nw": int(week.max()) + 1,
         "sc": jnp.asarray(np.maximum(sh_tr, 0) if STO else np.zeros(1)), "sd": jnp.asarray(np.maximum(-sh_tr, 0) if STO else np.zeros(1)), "nk": nk, "ck0": jnp.asarray(ck0), "cw": jnp.asarray(cw), "eu": jnp.asarray((H["eu"] - eun[0]) / eun[1]), "nb": jnp.asarray(H["nb"] - NBM[0]), "cont": jnp.asarray(cont), "gref": gref,
         "wt": jnp.asarray(0.5 ** ((H["day"].max() - H["day"]) / HL) if HL > 0 else np.ones(len(H["y"])))}
    if True:
        guide = AutoDelta(model, init_loc_fn=numpyro.infer.init_to_median(num_samples=10))
        sched = numpyro.optim.optax_to_numpyro(__import__("optax").adam(__import__("optax").cosine_decay_schedule(0.03, STEPS, 0.05)))
        svi = SVI(model, guide, sched, Trace_ELBO()); res = svi.run(jax.random.PRNGKey(j), STEPS, d, y=jnp.asarray(H["y"]), progress_bar=False)
        p = {k: np.asarray(v) for k, v in guide.median(res.params).items()}; loss = float(res.losses[-1]); draws = None
        if TAUW_FIX >= 0 and "w_e" in p: p["tau_w"] = np.asarray(TAUW_FIX)
    p_map = p
    if FIT == "nuts":   # full sampling, started at the posterior mode
        tn = time.time(); init = {k: jnp.asarray(v) for k, v in p.items() if k != "tau_w" or TAUW_FIX < 0}
        mc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=9, init_strategy=numpyro.infer.init_to_value(values=init)), num_warmup=int(os.environ.get("WARM", 500)), num_samples=int(os.environ.get("DRAWS", 300)), num_chains=CHAINS, chain_method="parallel", progress_bar=False)
        mc.run(jax.random.PRNGKey(j), d, y=jnp.asarray(H["y"]), extra_fields=("diverging",)); draws = {k: np.asarray(v) for k, v in mc.get_samples().items()}
        diag = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), group_by_chain=True)
        rhat = max(float(np.nanmax(v["r_hat"])) for v in diag.values()); ess = min(float(np.nanmin(v["n_eff"])) for v in diag.values()); div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
        worst = sorted(((float(np.nanmax(v["r_hat"])), k) for k, v in diag.items()), reverse=True)[:3]
        print(f"{tag} sampling {time.time() - tn:.0f}s: max r_hat {rhat:.3f} min ess {ess:.0f} divergences {div} worst {worst}", flush=True)
        if TAUW_FIX >= 0 and "w_e" in draws: draws["tau_w"] = np.full(len(draws["b0"]), TAUW_FIX)
        p = {k: np.median(v, 0) for k, v in draws.items()}
    m_tr, x_tr = predict(p, eun, H["rl"], H["gas"], H["hour"], H["dtype"], H["eu"], gref, week=week, nb=H["nb"], ct=(ck0, cw), sto=sh_tr); r_tr = H["y"] - m_tr
    # second step: what the errors of the last known day say about the next two days, per target hour
    S = pd.Series(r_tr, index=H["pos"]).reindex(np.arange(d1 - 200 * 96, d1, 4)); R = S.to_numpy().reshape(-1, 24)[1:]      # days x hours, local days (d0 is local midnight)
    XH = pd.Series(x_tr, index=H["pos"]).reindex(np.arange(d1 - 200 * 96, d1, 4)).to_numpy().reshape(-1, 24)[1:]; PH = pd.Series(m_tr, index=H["pos"]).reindex(np.arange(d1 - 200 * 96, d1, 4)).to_numpy().reshape(-1, 24)[1:]
    carry = np.zeros((2, lc.HZ // 4 + 1)); coef = np.zeros((2, 24, 2))
    for li, lag in enumerate((1 + KNOWN_LAG, 2 + KNOWN_LAG)):
        Xd = np.stack([R[:-lag], np.repeat(R[:-lag, 23:24], 24, 1)], -1); Yd = R[lag:]                                   # (days, 24, 2), (days, 24)
        for h in range(24):
            okd = np.isfinite(Xd[:, h]).all(1) & np.isfinite(Yd[:, h]); A = Xd[okd, h]; coef[li, h] = np.linalg.solve(A.T @ A + 0.3 * okd.sum() * np.var(A) * np.eye(2), A.T @ Yd[okd, h]) if okd.sum() > 30 else 0.0
    last = R[-1] if np.isfinite(R[-1]).all() else np.zeros(24); hh = hour_all[np.arange(d0, d0 + lc.HZ)]; lagq = np.where(day_all[np.arange(d0, d0 + lc.HZ)] - day_all[d0 - 1] <= 1, 0, 1)
    if H["pos"][-1] != d1 - 4: last = np.zeros(24)
    r_carry = coef[lagq, hh, 0] * last[hh] + coef[lagq, hh, 1] * last[23]; phi = float(np.nanmean(coef[0, :, 0])); r_last = float(last[23])
    def ridge1(xv, yv, lam):
        okd = np.isfinite(xv) & np.isfinite(yv); return float((xv[okd] * yv[okd]).sum() / ((xv[okd] ** 2).sum() + lam * okd.sum() * np.nanvar(xv))) if okd.sum() > 30 else 0.0
    Rc = np.clip(R, *np.nanpercentile(R, [1, 99])); l3 = np.nanmean(Rc[:, 21:24], 1); dm = np.nanmean(Rc, 1); last3 = float(np.nanmean(np.clip(last, *np.nanpercentile(R, [1, 99]))[21:24])); lastm = float(np.nanmean(np.clip(last, *np.nanpercentile(R, [1, 99]))))
    cB = np.array([[ridge1(l3[:-lag], Rc[lag:, h], 0.3) for h in range(24)] for lag in (1 + KNOWN_LAG, 2 + KNOWN_LAG)]); cC = np.array([[ridge1(dm[:-lag], Rc[lag:, h], 0.3) for h in range(24)] for lag in (1 + KNOWN_LAG, 2 + KNOWN_LAG)])
    nd = len(R); cut = int(nd * 2 / 3); coefV = np.zeros_like(coef)
    for li, lag in enumerate((1 + KNOWN_LAG, 2 + KNOWN_LAG)):
        Xd = np.stack([Rc[:-lag], np.repeat(Rc[:-lag, 23:24], 24, 1)], -1); Yd = Rc[lag:]
        for h in range(24):
            okd = np.isfinite(Xd[:, h]).all(1) & np.isfinite(Yd[:, h]); tr = okd & (np.arange(len(Yd)) < cut); te = okd & (np.arange(len(Yd)) >= cut)
            if tr.sum() < 30 or te.sum() < 15: continue
            A = Xd[tr, h]; c = np.linalg.solve(A.T @ A + 0.3 * tr.sum() * np.var(A) * np.eye(2), A.T @ Yd[tr, h])
            if np.abs(Yd[te, h] - Xd[te, h] @ c).mean() < 0.98 * np.abs(Yd[te, h]).mean(): coefV[li, h] = coef[li, h]
    lastc = np.clip(last, *np.nanpercentile(R, [1, 99])); r_cv = coefV[lagq, hh, 0] * lastc[hh] + coefV[lagq, hh, 1] * lastc[23]
    carries = {"cV": r_cv, "cB": cB[lagq, hh] * last3, "cC": cC[lagq, hh] * lastm, "cBC": 0.5 * (cB[lagq, hh] * last3 + cC[lagq, hh] * lastm), "cN": np.where(np.arange(lc.HZ) < 32, r_carry, 0.0), "cBn": np.where(np.arange(lc.HZ) < 48, cB[lagq, hh] * last3, 0.0)}
    # within-hour shape from the last 60 days of quarter-hour residuals
    q0 = d1 - 60 * 96; mq, _ = predict(p, eun, rl_all[q0:d1], gas_all[q0:d1], hour_all[q0:d1], daytype_all[q0:d1], eu_all[q0:d1], gref, nb=nb_all[q0:d1])
    rq = y_all[q0:d1] - mq; dev = rq - pd.Series(rq).groupby(np.arange(d1 - q0) // 4).transform("mean").to_numpy()
    w = pd.DataFrame({"h": hour_all[q0:d1], "q": quarter_all[q0:d1], "d": dev}).groupby(["h", "q"])["d"].mean().unstack().reindex(range(24)).fillna(0).to_numpy()
    T = np.arange(d0, d0 + lc.HZ); hrs_ahead = (np.arange(lc.HZ) // 4) + 1
    out = {}
    if NB:
        tn = np.load(f"{NBT}/{tag}.npz") if os.path.exists(f"{NBT}/{tag}.npz") else None
        fcn = tn["pes"] if tn is not None else nb_all[T]; nbf = {"known": nb_all[T], "own": fcn, "ts": fcn, "ownbc": fcn}; nbd1 = tn["d1"] if tn is not None else nb_all[np.arange(d1, d0)]
        for kv in os.environ.get("NBMAP", "").split(","):      # arm:true|fc - which regional figure an extra arm gets
            if kv: nbf[kv.split(":")[0]] = nb_all[T] if kv.split(":")[1] == "true" else fcn
    else: nbf = {}; nbd1 = None
    NB_FC = nbf.get("own")
    E = (I["rl_true"] - RLP).astype(float); known = [i for i in range(len(E)) if cuts[i] <= d0 - 2 * 96]
    recent = [i for i in known if cuts[i] > d0 - 9 * 96]; bc = float(np.mean([E[i].mean() for i in recent])) if len(recent) >= 2 else 0.0
    out["rl_bias_correction"] = np.array(bc)
    def arm_sto(xq, pos):
        """storage shift for quarter-hour slots: dispatch per local day on the hourly means of the reference x"""
        if not STO: return None
        key = day_all[pos] * 24 + hour_all[pos]; u, inv = np.unique(key, return_inverse=True); xh = np.array([np.nanmean(np.asarray(xq, float)[inv == i]) for i in range(len(u))])
        return sto_shift(xh, float(I["gas"][j]), u // 24, rval)[inv]
    STOA = {}
    for arm, rl in [("own", RLP[j].astype(float)), ("known", I["rl_true"][j].astype(float)), ("ownbc", RLP[j].astype(float) + bc)] + [(k, v[j].astype(float)) for k, v in MORE.items()]:
        STOA[arm] = arm_sto(zr[f"x_{arm}"], T) if STO else None
        m, x = predict(p, eun, rl, np.full(lc.HZ, float(I["gas"][j])), hour_all[T], daytype_all[T], I["eu"][j].astype(float), gref, nb=nbf.get(arm, nbf.get("own")) if NB else None, sto=STOA[arm])
        if STO: out[f"sto_{arm}"] = STOA[arm].astype(np.float32)
        out[f"{arm}_struct"] = m / 10; out[f"{arm}_shape"] = (m + w[hour_all[T], quarter_all[T]]) / 10; out[f"{arm}"] = (m + w[hour_all[T], quarter_all[T]] + r_carry) / 10; out.update({f"{arm}_{k}": (m + w[hour_all[T], quarter_all[T]] + v) / 10 for k, v in carries.items()}); out[f"x_{arm}"] = x
    if os.environ.get("DEBUG"):
        np.savez_compressed(f"{OUT}/debug_{tag}.npz", eu=H["eu"], dtype=H["dtype"], wind=fc["wind"].to_numpy()[H["pos"]], solar=fc["solar"].to_numpy()[H["pos"]], load=fc["load"].to_numpy()[H["pos"]], r=r_tr, x=x_tr, m=m_tr, y=H["y"], rl=H["rl"], gas=H["gas"], hour=H["hour"], day=H["day"], pos=H["pos"], week=week, losses=np.asarray(res.losses) if FIT == "map" else np.zeros(1))
    # own forecast with its error propagated: error paths of earlier forecasts (known two days back), centred
    rng = np.random.default_rng(j); pool = E[known[-60:]] - bc if len(known) >= 8 else rng.normal(0, 3.0, (60, lc.HZ))
    mm = np.stack([predict(p, eun, RLP[j].astype(float) + bc + e, np.full(lc.HZ, float(I["gas"][j])), hour_all[T], daytype_all[T], I["eu"][j].astype(float), gref)[0] for e in pool])
    extra = w[hour_all[T], quarter_all[T]] + r_cv
    out["ownprop_med"] = (np.median(mm, 0) + extra) / 10; out["ownprop_mean"] = (mm.mean(0) + extra) / 10; out["ownbc_cV_q"] = (np.percentile(mm, [5, 10, 50, 90, 95], 0) + extra) / 10
    # predictive inputs: error scale per quarter hour, and for the own forecasts the mean under resampled error paths of that
    # forecast (errors of earlier forecasts whose published figures exist at issue; too few of them: errors from FALLBACK)
    def scale_of(x): zx = (x - 25.0) / 15.0; return 10.0 * np.exp(p["s"][0] + p["s"][1] * np.log(float(I["gas"][j]) / gref) + p["s"][2] * zx + p["s"][3] * zx ** 2)
    out["scale_known"] = scale_of(out["x_known"]) / 10; rngm = np.random.default_rng(1000 + j)
    for arm, RLk in [("own", RLP)] + list(MORE.items()):
        Ek = (I["rl_true"] - RLk).astype(float); kn = [i for i in range(len(Ek)) if cuts[i] <= d0 - 2 * 96]; pool = Ek[kn[-60:]] if len(kn) >= 10 else FALLBACK.get(arm, FALLBACK.get("own"))
        if pool is None: continue
        paths = pool[rngm.integers(0, len(pool), 40)]; mk, xk = zip(*[predict(p, eun, RLk[j].astype(float) + e, np.full(lc.HZ, float(I["gas"][j])), hour_all[T], daytype_all[T], I["eu"][j].astype(float), gref, nb=nbf.get(arm, nbf.get("own")) if NB else None) for e in paths])
        out[f"mix_{arm}"] = (np.array(mk) / 10).astype(np.float32); out[f"mixs_{arm}"] = (np.array([scale_of(x) for x in xk]) / 10).astype(np.float32); out[f"mixc_{arm}"] = out[f"{arm}_struct"]
    # error of D+1 (prices known, published figures not): with the own forecast of its residual load
    R1 = R; XH1 = XH; PH1 = PH
    if D1F is not None and KNOWN_LAG == 1:
        rows = np.arange(d1, d0); m1, x1 = predict(p, eun, D1F[j].astype(float), np.full(len(rows), float(I["gas"][j])), hour_all[rows], daytype_all[rows], eu_all[rows], gref, nb=nbd1, sto=(arm_sto(np.repeat(zr["XH1"][-1].astype(float), 4)[:len(rows)], rows) if STO else None))   # gas as known at issue
        R1 = np.vstack([R, np.nanmean((y_all[rows] - m1).reshape(24, 4), 1)[None]]); XH1 = np.vstack([XH, x1.reshape(24, 4).mean(1)[None]]); PH1 = np.vstack([PH, m1.reshape(24, 4).mean(1)[None]])
    sb, sh, cal, drift = unpack(p); grid = np.arange(-10, 71, 5.0)
    if FIT == "nuts":
        S_ = len(draws["b0"]); sel = np.arange(0, S_, max(1, S_ // 800)); gasf = np.full(lc.HZ, float(I["gas"][j])); post = {"mean": {}, "diag": np.array([rhat, ess, div, time.time() - tn])}
        arms = [("known", I["rl_true"][j].astype(float))] + [(k, v[j].astype(float)) for k, v in MORE.items()] + [("own", RLP[j].astype(float))]
        bc_, hc_, sc_, nu_ = [], [], [], []; mm_ = {a_: [] for a_, _ in arms}; sl_ = {a_: [] for a_, _ in arms}; mapm = {}
        for i in sel:
            pi = {k: v[i] for k, v in draws.items()}; sbi, shi, _, _ = unpack(pi); bc_.append(pl_np(grid, pi["b0"], sbi, sbi[0])); hc_.append(pl_np(grid, 0.0, shi, 0.0)); nu_.append(pi["nu"])
            for a_, rl in arms:
                m_, x_ = predict(pi, eun, rl, gasf, hour_all[T], daytype_all[T], I["eu"][j].astype(float), gref); mm_[a_].append(m_)
                zx = (x_ - 25.0) / 15.0; sl_[a_].append(10.0 * np.exp(pi["s"][0] + pi["s"][1] * np.log(gasf / gref) + pi["s"][2] * zx + pi["s"][3] * zx ** 2))
        for a_, rl in arms: mapm[a_] = predict(p_map, eun, rl, gasf, hour_all[T], daytype_all[T], I["eu"][j].astype(float), gref)[0]
        np.savez_compressed(f"{OUT}/post_{tag}.npz", grid=grid, b=np.array(bc_, np.float32), h=np.array(hc_, np.float32), nu=np.array(nu_, np.float32), diag=post["diag"], w=w[hour_all[T], quarter_all[T]], gas=float(I["gas"][j]),
                            **{f"m_{a_}": np.array(v, np.float32) for a_, v in mm_.items()}, **{f"s_{a_}": np.array(v, np.float32) for a_, v in sl_.items()}, **{f"map_{a_}": v.astype(np.float32) for a_, v in mapm.items()},
                            kappa=draws.get("kappa", np.zeros((1, 4)))[sel], g_slope=np.array([draws[k][sel] for k in ("lb0", "lh0")]))
    np.savez_compressed(f"{OUT}/{tag}.npz", actual=y_all[T] / 10, **out, grid=grid, b_curve=pl_np(grid, p["b0"], sb, sb[0]), h_curve=pl_np(grid, 0.0, sh, 0.0), cal=cal, drift=drift,
                        a_s=p.get("a_s", np.zeros(2)), cdrift=(cdrift(p["cd_e"], np) if "c" in DRIFT else np.zeros((1, len(CC)))), kappa=p.get("kappa", np.zeros(4)), w_nb=float(p.get("w_nb", 0.0)), R=R, R1=R1, XH1=XH1.astype(np.float32), PH1=PH1.astype(np.float32), gref=gref, misc=np.array([phi, float(p["nu"]), r_last, loss, time.time() - t0, len(H["y"]), np.abs(r_tr).mean(), np.abs(r_tr[-24 * 30:]).mean()]), s=p["s"])
    print(f"{tag} {time.time() - t0:.0f}s rows {len(H['y'])} carry {phi:.2f} nu {float(p['nu']):.1f} in-sample MAE {np.abs(r_tr).mean() / 10:.2f} last 30 d {np.abs(r_tr[-24 * 30:]).mean() / 10:.2f} ct  kappa {np.round(p.get('kappa', np.zeros(4)), 1)}", flush=True)
print("finished", flush=True)
