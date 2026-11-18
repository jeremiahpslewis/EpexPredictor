"""Coupled-market model (research harness): supply stacks per zone, exchange with Germany, market clearing.

Zones: DE and its neighbours FR, NL, BE, AT, PL, CZ, CH, DK. Hourly.
Supply of zone z at price p:   S_z(p, t) = sum_k cap_zk(t) * sigmoid((p - mc_zk(t)) / wd_zk) + sl_z * p / 100
  blocks k: 0 low-cost (must-run, run-of-river, nuclear, curtailable renewables), 1 non-gas A (lignite / reservoir hydro),
            2 non-gas B (hard coal), 3 gas combined cycle, 4 peakers
  marginal cost  mc_zk(t) = alpha_zk + beta_zk * gas(t) + EF_k * c(t)
     beta: heat rate (tight priors for the gas blocks), EF: emission rate [t/MWh] fixed from engineering,
     c(t): the one latent state, a common non-gas cost level (stands for the CO2 price, which is not in the data),
           weekly random walk, zero in the last week
  capacity  cap_zk(t) = free capacity (prior from installed power) + K * measured availability, where availability is the
     trailing 14-day maximum of realised output, known with one day's delay (French nuclear, German lignite and coal)
Exchange: Germany exports to z   e_z = C_z * tanh((p_z - p_DE) / kappa_z)   (C_z prior from the interconnector capacity)
Clearing: DE   S_0(p_0) = RL_0 + shift[daytype, hour] + sum_z e_z ;   neighbour  S_z(p_z) = RL_z + delta_z - e_z
Fit (posterior mode): each zone's price is the root of its own clearing equation with the exchanges at the observed prices
  (no equilibrium solve needed in training). Forecast: the full equilibrium of all nine prices is solved.
Version 2 (the first check showed prices alone do not identify the parts): measured quantities are data in the model.
  German output by fuel:  lignite, hard coal, gas ~ floor_k + block output at the observed price  (floor = must-run part)
  exchange per border:    scheduled export ~ e_z(price difference),  e_z = -Ci_z + (Ce_z + Ci_z) * sigmoid((dp - m_z) / kappa_z)
     (separate export and import capacity and an offset, as the scheduled exchanges show)
usage: star.py NAME PERIOD TAG [TAG ...]     env: SEED, STEPS, TRAIN_DAYS"""
import os, sys, time
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import live_common as lc

ZN = ["de", "fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]; Z, KB = len(ZN), 6
SEED = int(os.environ.get("SEED", 0)); STEPS = int(os.environ.get("STEPS", 2500)); TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", 540)); LW = 0.125; WNB = 0.3
# prior medians of the free capacities [GW] per block (installed power 2025, energy-charts); measured blocks get their capacity from availability
# version 3 (after the prior predictive check): centres are what is available, not what is installed (German hard coal: 15 GW
# installed, never above 6 GW; gas: 35 GW installed, never above 13.5). Spread unchanged (lognormal, sd 0.5): centres moved, nothing capped.
# Block 5 is curtailment: wind and solar are already netted out of residual load, so this is not extra supply. It is the part of
# them that switches off below about a zero price: its output runs from -capacity (far below zero) to 0 (above), which is
# coded as a block of that capacity with the same capacity added to demand.
CAP0 = {"de": [12, 0.2, 0.2, 7, 10, 25], "fr": [12, 10, 1.5, 8, 3, 8], "nl": [4, 0.5, 3, 9, 3, 8], "be": [4, 1.3, 0.3, 4.5, 1.5, 4], "at": [7, 5, 0.3, 2.5, 1, 3],
        "pl": [3, 5.5, 13, 3.5, 1.5, 6], "cz": [5, 5, 0.5, 1, 1.5, 1.5], "ch": [7, 9, 0.3, 0.4, 0.5, 2], "dk": [2.6, 0.3, 1.2, 1.2, 0.6, 3]}
MEAS = {("de", 1): "g_de_lignite", ("de", 2): "g_de_coal", ("fr", 0): "g_fr_nuclear"}                    # blocks whose capacity follows measured availability
EF = np.array([0.0, 1.0, 0.85, 0.37, 0.55, 0.0]); EF_A = {"fr": 0.0, "at": 0.0, "ch": 0.0, "nl": 0.0, "be": 0.0, "dk": 0.0}       # block A is hydro there, not lignite
ALPHA0 = np.array([0.0, 75.0, 85.0, 28.0, 45.0, -3.0]); ALPHA_SD = np.array([10.0, 20.0, 20.0, 8.0, 15.0, 5.0]); WD0 = np.array([8.0, 8.0, 8.0, 10.0, 40.0, 4.0])
BA0 = np.array([0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0]); BA_SD = np.array([0.1, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2])     # gas link of block A: reservoir hydro (FR, AT, CH) prices off gas; lignite does not
NTC = np.array([3.0, 4.0, 1.0, 4.9, 2.0, 2.5, 3.5, 3.5])                                              # FR NL BE AT PL CZ CH DK
PLO, PHI, NBIS = -300.0, 2000.0, 26

D = pd.read_parquet(f"{lc.SP}/out/star_data.parquet"); idx = D.index; local = idx.tz_convert("Europe/Berlin")
F = lc.load(); wdv = F["DE"]["weekday"].to_numpy(); hol = F["DE"]["holiday"].to_numpy()
dtype_all = np.where((hol >= 0.5) | (wdv == 6), 2, np.where(wdv == 5, 1, 0)); hour_all = local.hour.to_numpy(); day_all = np.asarray(local.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64))
P_all = D[[f"p_{z}" for z in ZN]].to_numpy(float); RL_all = D[[f"rl_{z}" for z in ZN]].to_numpy(float); gas_all = D["gas"].to_numpy(float)
AV = {}                                                                                               # availability: trailing 14-day maximum of hourly output, one day late
for key, col in MEAS.items():
    h = D[col].rolling(4, min_periods=1).mean(); AV[key] = h.rolling(14 * 96, min_periods=96).max().shift(96).to_numpy(float)

def supply(p, cap, mc, wd, sl, xp=jnp):
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    return (cap * sg((p[..., None] - mc) / wd)).sum(-1) + sl * p / 100.0
def dsupply(p, cap, mc, wd, sl, xp=jnp):
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    s = sg((p[..., None] - mc) / wd); return (cap * s * (1 - s) / wd).sum(-1) + sl / 100.0
def root(Q, cap, mc, wd, sl):
    """price at which supply equals Q; bisection without gradient, then one Newton step that carries the gradient"""
    capS, mcS, wdS, slS, QS = map(jax.lax.stop_gradient, (cap, mc, wd, sl, Q))
    def body(i, st):
        lo, hi = st; mid = 0.5 * (lo + hi); f = supply(mid, capS, mcS, wdS, slS) - QS; return jnp.where(f < 0, mid, lo), jnp.where(f < 0, hi, mid)
    lo, hi = jax.lax.fori_loop(0, NBIS, body, (jnp.full_like(QS, PLO), jnp.full_like(QS, PHI))); pb = 0.5 * (lo + hi)
    return pb - (supply(pb, cap, mc, wd, sl) - Q) / jax.lax.stop_gradient(dsupply(pb, cap, mc, wd, sl))

def exch(par, dp, xp=jnp):
    """German exports to each neighbour [GW] at price difference dp = p_z - p_DE (n, Z-1)"""
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    return -par["Ci"] + (par["Ce"] + par["Ci"]) * sg((dp - par["m"]) / par["kappa"])
GCOL = ["g_de_lignite", "g_de_coal", "g_de_gas"]; XCOL = [f"x_{z}" for z in ZN[1:]]
# Version 4 (after the energy balance by source): pumped storage is a measured part of German supply, about -5..+5 GW. It
# follows the hour's price relative to the day's mean (correlation 0.81-0.87 against 0.72 with the price level):
#     net output = P_ps * tanh((p - mean price of the day) / kappa_ps)
def day_mean(v, day):
    """mean of v over each local day, returned per element"""
    u, inv = np.unique(day, return_inverse=True); s_ = np.bincount(inv, weights=np.nan_to_num(v)); c_ = np.bincount(inv, weights=np.isfinite(v).astype(float)); return (s_ / np.maximum(c_, 1))[inv]

def build(par, d, xp=jnp):
    """capacities (n, Z, K), marginal costs (n, Z, K), widths (Z, K), slopes (Z,), demand shift for DE (n,)"""
    ct = xp.concatenate([-xp.cumsum((2.0 * par["c_e"])[::-1])[::-1], xp.zeros(1)])[d["week"]]                    # common cost level, zero in the last week
    cap = par["capf"][None] + par["kmeas"][None] * d["avail"]
    mc = par["alpha"][None] + par["beta"][None] * d["gas"][:, None, None] + d["ef"][None] * ct[:, None, None]
    mu_h = xp.concatenate([par["sh_h"], -par["sh_h"].sum(keepdims=True)]); cal = mu_h[None, :] + xp.concatenate([xp.zeros(1), par["sh_d"]])[:, None] + xp.concatenate([xp.zeros((1, 24)), par["sh_c"]])
    return cap, mc, par["wd"], par["sl"], cal[d["dtype"], d["hour"]], ct

def model(d, P=None):
    capf = numpyro.sample("capf", dist.LogNormal(jnp.log(d["cap0"]), 0.5)); kmeas = numpyro.sample("kmeas", dist.LogNormal(np.log(0.8), 0.25).expand([Z, KB])) * d["meas_mask"]
    alpha = numpyro.sample("alpha", dist.Normal(np.broadcast_to(ALPHA0, (Z, KB)), np.broadcast_to(ALPHA_SD, (Z, KB))))
    b_0 = numpyro.sample("b_0", dist.Normal(0.0, 0.05).expand([Z])); b_a = numpyro.sample("b_a", dist.Normal(BA0, BA_SD)); b_b = numpyro.sample("b_b", dist.Normal(0.0, 0.15).expand([Z]))
    b_c = numpyro.sample("b_c", dist.Normal(1.85, 0.12).expand([Z])); b_p = numpyro.sample("b_p", dist.Normal(2.7, 0.3).expand([Z]))
    b_5 = numpyro.sample("b_5", dist.Normal(0.0, 0.05).expand([Z])); beta = jnp.stack([b_0, b_a, b_b, b_c, b_p, b_5], -1)
    wd = numpyro.sample("wd", dist.LogNormal(np.log(np.broadcast_to(WD0, (Z, KB))), 0.5)); sl = numpyro.sample("sl", dist.LogNormal(np.log(0.5), 0.5).expand([Z]))
    delta = numpyro.sample("delta", dist.Normal(0, 5).expand([Z - 1]))
    xp_ = {"Ce": numpyro.sample("Ce", dist.LogNormal(np.log(NTC), 0.5)), "Ci": numpyro.sample("Ci", dist.LogNormal(np.log(NTC), 0.5)),
           "m": numpyro.sample("m", dist.Normal(0, 10).expand([Z - 1])), "kappa": numpyro.sample("kappa", dist.LogNormal(np.log(10.0), 0.7).expand([Z - 1]))}
    floor = numpyro.sample("floor", dist.LogNormal(np.log(np.array([2.5, 0.6, 1.5])), 0.5))             # must-run output of German lignite, hard coal, gas [GW]
    p_ps = numpyro.sample("p_ps", dist.LogNormal(np.log(5.0), 0.3)); k_ps = numpyro.sample("k_ps", dist.LogNormal(np.log(25.0), 0.5)); s_ps = numpyro.sample("s_ps", dist.LogNormal(np.log(1.0), 0.5))
    psmod = p_ps * jnp.tanh((P[:, 0] - d["pdm"]) / k_ps)                                                # pumped storage at the observed price and day mean
    par = {"capf": capf, "kmeas": kmeas, "alpha": alpha, "beta": beta, "wd": wd, "sl": sl, "c_e": numpyro.sample("c_e", dist.Normal(0, 1).expand([d["nw"] - 1])),
           "sh_h": numpyro.sample("sh_h", dist.Normal(0, 3).expand([23])), "sh_d": numpyro.sample("sh_d", dist.Normal(0, 3).expand([2])), "sh_c": numpyro.sample("sh_c", dist.Normal(0, 1.5).expand([2, 24]))}
    cap, mc, wd_, sl_, shift, ct = build(par, d)
    e = jnp.where(d["ok"][:, 1:] & d["ok"][:, :1], exch(xp_, P[:, 1:] - P[:, :1]), 0.0)                 # exports DE -> z at the observed prices
    Q = jnp.concatenate([(d["rl"][:, 0] + shift + e.sum(1) - floor.sum() - psmod)[:, None], d["rl"][:, 1:] + delta[None] - e], 1) + capf[None, :, 5]   # floors serve demand at any price; + curtailment capacity (see CAP0)
    phat = root(Q, cap, mc, wd_, sl_)
    s = numpyro.sample("s", dist.Normal(np.log(12.0), 1.0).expand([Z])); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4))
    w = jnp.concatenate([jnp.ones(1), jnp.full(Z - 1, WNB)])
    numpyro.factor("ll", LW * (w[None] * d["ok"] * dist.StudentT(nu, 0.0, jnp.exp(s)[None]).log_prob(jnp.where(d["ok"], P - phat, 0.0))).sum())
    # measured quantities: German output by fuel at the observed price, scheduled exchange at the observed price difference
    blk = cap[:, 0, :] * jax.nn.sigmoid((P[:, :1] - mc[:, 0, :]) / wd_[0][None])                        # (n, K) output of the German blocks
    gmod = jnp.stack([floor[0] + blk[:, 1], floor[1] + blk[:, 2], floor[2] + blk[:, 3] + blk[:, 4]], 1)
    sg_ = numpyro.sample("s_g", dist.LogNormal(np.log(1.0), 0.5).expand([3])); sx_ = numpyro.sample("s_x", dist.LogNormal(np.log(0.7), 0.5).expand([Z - 1]))
    numpyro.factor("lg", LW * (d["gok"] * dist.StudentT(4.0, 0.0, sg_[None]).log_prob(jnp.where(d["gok"], d["g"] - gmod, 0.0))).sum())
    numpyro.factor("lp", LW * (d["psok"] * dist.StudentT(4.0, 0.0, s_ps).log_prob(jnp.where(d["psok"], d["ps"] - psmod, 0.0))).sum())
    numpyro.factor("lx", LW * (d["xok"] * dist.StudentT(4.0, 0.0, sx_[None]).log_prob(jnp.where(d["xok"], d["x"] - exch(xp_, P[:, 1:] - P[:, :1]), 0.0))).sum())

def unpack(p):
    """numpy parameter dict from the guide's medians"""
    q = {k: np.asarray(v, float) for k, v in p.items()}; q["beta"] = np.stack([q["b_0"], q["b_a"], q["b_b"], q["b_c"], q["b_p"], q["b_5"]], -1); q["kmeas"] = q["kmeas"] * MEAS_MASK; return q

def equilibrium(q, d, rl=None):
    """all nine prices that clear the coupled market (numpy; nested bisection). rl: residual loads (n, Z), default the data's"""
    rl = d["rl"] if rl is None else rl; cap, mc, wd, sl, shift, ct = build(q, d, np); n = len(rl)
    def neigh(p0):                                                                                       # neighbours' prices and DE exports given the German price
        lo = np.full((n, Z - 1), PLO); hi = np.full((n, Z - 1), PHI)
        for _ in range(NBIS):
            mid = 0.5 * (lo + hi); f = supply(mid, cap[:, 1:], mc[:, 1:], wd[1:], sl[1:], np) + exch(q, mid - p0[:, None], np) - (rl[:, 1:] + q["delta"] + q["capf"][1:, 5]); lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
        pz = 0.5 * (lo + hi); return pz, exch(q, pz - p0[:, None], np)
    pbar = None
    for it in range(4):                                                                                 # pass 0 without storage, then with the day mean of the previous pass
        lo = np.full(n, PLO); hi = np.full(n, PHI)
        for _ in range(NBIS):
            mid = 0.5 * (lo + hi); _, e = neigh(mid); ps = 0.0 if pbar is None else q["p_ps"] * np.tanh((mid - pbar) / q["k_ps"])
            f = supply(mid[:, None], cap[:, :1], mc[:, :1], wd[:1], sl[:1], np)[:, 0] + ps - (rl[:, 0] + shift + e.sum(1) - q["floor"].sum() + q["capf"][0, 5]); lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
        p0 = 0.5 * (lo + hi); pbar = day_mean(p0, d["day"]) if pbar is None else 0.5 * (pbar + day_mean(p0, d["day"]))
    pz, e = neigh(p0); equilibrium.ps = q["p_ps"] * np.tanh((p0 - pbar) / q["k_ps"]); return np.column_stack([p0, pz]), e

def window(lo, hi):
    """hourly means of the quarter-hour rows lo..hi"""
    n = (hi - lo) // 4 * 4; r = lambda v: v[lo:lo + n].reshape(-1, 4, *v.shape[1:])
    H = {"P": np.nanmean(r(P_all), 1), "rl": np.nanmean(r(RL_all), 1), "gas": r(gas_all)[:, 0], "hour": r(hour_all)[:, 0], "dtype": r(dtype_all)[:, 0], "day": r(day_all)[:, 0], "pos": lo + 4 * np.arange(n // 4)}
    av = np.zeros((n // 4, Z, KB))
    for (z, k), a in AV.items(): av[:, ZN.index(z), k] = r(a)[:, 0]
    H["avail"] = av; H["g"] = np.stack([np.nanmean(r(D[c].to_numpy(float)), 1) for c in GCOL], 1); H["x"] = -np.stack([np.nanmean(r(D[c].to_numpy(float)), 1) for c in XCOL], 1)
    H["ps"] = np.nanmean(r((D["g_de_psgen"] + D["g_de_pump"]).to_numpy(float)), 1); return H

MEAS_MASK = np.zeros((Z, KB)); HYDRO_A = np.ones(Z); HYDRO_A[0] = 0.0                                    # German block A is lignite: no gas link
for (z, k) in MEAS: MEAS_MASK[ZN.index(z), k] = 1.0
EFM = np.tile(EF, (Z, 1))
for z, v in EF_A.items(): EFM[ZN.index(z), 1] = v
CAP0M = np.array([CAP0[z] for z in ZN], float)

def data_of(H, week, nw):
    ok = np.isfinite(H["P"]) & np.isfinite(H["rl"]) & np.isfinite(H["gas"])[:, None] & np.isfinite(H["avail"]).all((1, 2))[:, None]
    return {"rl": np.nan_to_num(H["rl"]), "gas": np.nan_to_num(H["gas"], nan=40.0), "hour": H["hour"], "dtype": H["dtype"], "week": week, "nw": nw, "avail": np.nan_to_num(H["avail"]), "ok": ok,
            "cap0": CAP0M, "meas_mask": MEAS_MASK, "hydro_a": HYDRO_A, "ef": EFM,
            "day": H["day"], "pdm": day_mean(H["P"][:, 0], H["day"]), "ps": np.nan_to_num(H["ps"]), "psok": np.isfinite(H["ps"]) & ok[:, 0],
            "g": np.nan_to_num(H["g"]), "gok": np.isfinite(H["g"]) & ok[:, :1], "x": np.nan_to_num(H["x"]), "xok": np.isfinite(H["x"]) & ok[:, 1:] & ok[:, :1]}

def fit(d0, seed=SEED, steps=STEPS, init=None, sim=None):
    """sim: dict with simulated P, g, x to fit instead of the data (simulated-data check)"""
    d1 = d0 - 96; lo = max(d1 - TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4; H = window(lo, d1)
    if sim is not None: H.update(sim)
    week = (H["day"] - H["day"].min()) // 7; nw = int(week.max()) + 1
    d = data_of(H, week, nw); dj = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    init_fn = numpyro.infer.init_to_median(num_samples=10) if init is None else numpyro.infer.init_to_value(values={k: jnp.asarray(v) for k, v in init.items()})
    guide = AutoDelta(model, init_loc_fn=init_fn); sched = numpyro.optim.optax_to_numpyro(__import__("optax").adam(__import__("optax").cosine_decay_schedule(0.02, steps, 0.05)))
    svi = SVI(model, guide, sched, Trace_ELBO()); res = svi.run(jax.random.PRNGKey(seed), steps, dj, P=jnp.asarray(np.nan_to_num(H["P"])), progress_bar=False)
    raw = {k: np.asarray(v) for k, v in guide.median(res.params).items()}; return unpack(raw), raw, d, H, float(res.losses[-1])

if __name__ == "__main__":
    name, period = sys.argv[1], sys.argv[2]; I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; OUT = f"{lc.SP}/out/star_{name}"; os.makedirs(OUT, exist_ok=True)
    for tag in sys.argv[3:]:
        j = tags.index(tag); d0 = int(I["cuts"][j]); t0 = time.time(); q, raw, d, H, loss = fit(d0)
        pe, e = equilibrium(q, d); ok = d["ok"][:, 0]; err = (H["P"][:, 0] - pe[:, 0])[ok] / 10
        np.savez_compressed(f"{OUT}/{tag}_s{SEED}.npz", **{f"raw_{k}": v for k, v in raw.items()}, peq=pe.astype(np.float32), eeq=e.astype(np.float32), P=H["P"].astype(np.float32), pos=H["pos"], loss=loss)
        print(f"{tag} seed {SEED}: {time.time() - t0:.0f}s loss {loss:.1f}; German price in sample, full equilibrium: MAE {np.abs(err).mean():.2f} ct (median error {np.median(err):+.2f}); export cap {np.round(q['Ce'], 1)}; import cap {np.round(q['Ci'], 1)}; floors {np.round(q['floor'], 1)}; storage {float(q['p_ps']):.1f} GW, kappa {float(q['k_ps']):.0f}", flush=True)
