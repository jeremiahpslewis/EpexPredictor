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
# Version 5 (LIK=q, default): the clearing condition is fitted in quantity space. Fitting the price as a function of the
# quantity the stack must serve attenuates the price response, because that quantity is noisy (published against realised
# residual load alone differs by 3.5 GW; the fit answered with a 35 GW catch-all block). So, like the output of each fuel,
# the balance is a quantity given the observed price:  supply_z(observed price) - demand_z ~ noise [GW], with the
# scheduled exchange and realised pumped storage used in the German balance where they exist. No root-finding in the fit;
# the price is what the estimated parts imply when the market is cleared. LIK=p is the earlier price-space fit.
# Version 7 (LIK=f, default): the generative model is shocks to every part -> clearing of all nine zones -> observed prices
# and quantities. Prices and quantities are determined together, so the density of the observables is the density of the
# shocks times the Jacobian of the clearing system: the determinant of d(balances)/d(prices). With every neighbour linked
# to Germany only, that matrix is an arrowhead and the determinant is closed form:
#     prod_z M_zz * (M_00 - sum_z e_z'^2 / M_zz),   M_zz = S_z' + e_z',   M_00 = German supply slope (incl. storage) + sum_z e_z'
# Without it (LIK=q) each part is a regression of quantity on an endogenous price, which understates supply when demand is high.
# Version 9: in a large share of hours a neighbour's price equals the German one exactly. Then the two form one market and the
# exchange is whatever clears it, not a response to a price difference. A smooth exchange function fitted through those
# hours degenerates to a step (the Jacobian rewards an ever steeper slope at a price difference of exactly zero). So coupled
# hours (|price difference| <= CEPS) are treated as what they are: the neighbour's supply slope adds directly to Germany's in
# the Jacobian, and the exchange function is fitted on the uncoupled hours only.
CEPS = float(os.environ.get("CEPS", 0.5))
LIK = os.environ.get("LIK", "f")
SEED = int(os.environ.get("SEED", 0)); STEPS = int(os.environ.get("STEPS", 2500)); TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", 540)); LW = 0.125; WNB = float(os.environ.get("WNB", 1.0))   # version 8: a joint likelihood has no per-zone weights; with the neighbours' balances down-weighted to 0.3
# but their Jacobian terms at full weight, capacity and noise scale could grow together without bound (a Polish block went to 1e13 GW)
# prior medians of the free capacities [GW] per block (installed power 2025, energy-charts); measured blocks get their capacity from availability
# version 3 (after the prior predictive check): centres are what is available, not what is installed (German hard coal: 15 GW
# installed, never above 6 GW; gas: 35 GW installed, never above 13.5). Spread unchanged (lognormal, sd 0.5): centres moved, nothing capped.
# Block 5 is curtailment: wind and solar are already netted out of residual load, so this is not extra supply. It is the part of
# them that switches off below about a zero price: its output runs from -capacity (far below zero) to 0 (above), which is
# coded as a block of that capacity with the same capacity added to demand.
CAP0 = {"de": [12, 0.2, 0.2, 7, 10, 25], "fr": [12, 10, 1.5, 8, 3, 8], "nl": [4, 0.5, 3, 9, 3, 8], "be": [4, 1.3, 0.3, 4.5, 1.5, 4], "at": [7, 5, 0.3, 2.5, 1, 3],
        "pl": [3, 5.5, 13, 3.5, 1.5, 6], "cz": [5, 5, 0.5, 1, 1.5, 1.5], "ch": [7, 9, 0.3, 0.4, 0.5, 2], "dk": [2.6, 0.3, 1.2, 1.2, 0.6, 3]}
# version 6: gas availability is seasonal (output at high prices reaches 15 GW in winter, 7-8 GW in summer), so the gas blocks follow measured availability too
MEAS = {("de", 1): "g_de_lignite", ("de", 2): "g_de_coal", ("de", 3): "g_de_gas", ("fr", 0): "g_fr_nuclear", ("fr", 3): "g_fr_gas"}   # blocks whose capacity follows measured availability
EF = np.array([0.0, 1.0, 0.85, 0.37, 0.55, 0.0]); EF_A = {"fr": 0.0, "at": 0.0, "ch": 0.0, "nl": 0.0, "be": 0.0, "dk": 0.0}       # block A is hydro there, not lignite
ALPHA0 = np.array([0.0, 75.0, 85.0, 28.0, 45.0, -3.0]); ALPHA_SD = np.array([10.0, 20.0, 20.0, 8.0, 15.0, 5.0]); WD0 = np.array([8.0, 8.0, 8.0, 10.0, 40.0, 4.0])
BA0 = np.array([0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0]); BA_SD = np.array([0.1, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2])
BB0 = np.zeros(9); BB_SD = np.full(9, 0.15)
if os.environ.get("GASLINK"):      # prior centres (and sd) for the gas link of German lignite and hard coal: "lignite,coal,sd"
    _g = [float(v) for v in os.environ["GASLINK"].split(",")]; BA0[0], BB0[0] = _g[0], _g[1]; BA_SD[0] = BB_SD[0] = _g[2]     # gas link of block A: reservoir hydro (FR, AT, CH) prices off gas; lignite does not
NTC = np.array([3.0, 4.0, 1.0, 4.9, 2.0, 2.5, 3.5, 3.5])                                              # FR NL BE AT PL CZ CH DK
PLO, PHI, NBIS = -300.0, 2000.0, 26

D = pd.read_parquet(f"{lc.SP}/out/star_data.parquet"); idx = D.index; local = idx.tz_convert("Europe/Berlin")
F = lc.load(); wdv = F["DE"]["weekday"].to_numpy(); hol = F["DE"]["holiday"].to_numpy()
dtype_all = np.where((hol >= 0.5) | (wdv == 6), 2, np.where(wdv == 5, 1, 0)); hour_all = local.hour.to_numpy(); day_all = np.asarray(local.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64))
P_all = D[[f"p_{z}" for z in ZN]].to_numpy(float); RL_all = D[[f"rl_{z}" for z in ZN]].to_numpy(float); gas_all = D["gas"].to_numpy(float)
# Version 10: (a) the observed carbon price (EEX primary auctions, last auction carried forward) enters every block's marginal
# cost with its emission rate; the latent cost level is what remains. (b) Plant inertia: thermal blocks respond to a mix of
# the day's mean price and the hour's price, lam * day mean + (1 - lam) * hour, lam per zone and block estimated (logit scale).
# Measured: lignite moves 0.47 GW per ct of the day's mean price and 0.24 per ct within the day; hard coal 0.30 / 0.12; gas 0.48 / 0.22.
_e = pd.read_parquet(f"{lc.SP}/out/eua_daily.parquet")["eua"]; _dl = pd.DatetimeIndex(local.tz_localize(None).normalize())
_u = _dl.unique(); eua_all = _e.reindex(_e.index.union(_u)).sort_index().ffill().shift(1).reindex(_dl).to_numpy(float) - 70.0     # yesterday's value, centred on 70 EUR/t
INERT = os.environ.get("INERT", "1") == "1"; USE_EUA = os.environ.get("EUA", "1") == "1"
AV = {}                                                                                               # availability: trailing 14-day maximum of hourly output, one day late
for key, col in MEAS.items():
    h = D[col].rolling(4, min_periods=1).mean(); AV[key] = h.rolling(14 * 96, min_periods=96).max().shift(96).to_numpy(float)

# Version 11: (a) heat-led plants. A combined heat and power plant delivers power when heat is wanted, whatever the price: in the
# hours when running clearly does not pay, German gas still delivers 1.4 GW on mild days and 7 GW on days near freezing, hard
# coal 0.8 against 2.9 GW, the same in every year. So the must-run output of each German fuel is f0 + f1 * heating degrees, and
# each zone's demand for its price-driven stack falls by hz * heating degrees. Heating degrees: 15 C minus the daily mean
# temperature (today 0.6, yesterday 0.3, the day before 0.1), smoothly floored at zero.
# (b) Border capacity varies with German weather (the Danish exchange at a given price difference moves with German wind and
# sun): export and import capacity per border are multiplied by exp(a_wind * wind + a_sun * sun), wind and sun standardised.
# Version 12: (a) German lignite and hard coal each carry their own latent offer premium (weekly random walk, zero in the last week) in place of the
# common cost level, which also loaded on gas: gas plants sit on one curve in every period, the same lignite and hard coal units ask
# 20-50 EUR/MWh more from spring 2026 on (unit data; not carbon, coal price, outages or heat; Czech and Polish plants do not show it).
# (b) the heating threshold is estimated (the must-run of gas flattens near 11 C, not 15). (c) the residual load the market cleared on is
# not the published day-ahead figure: rl = published + theta * (realised - published), theta estimated; what is left of the difference moves
# realised output by fuel (rho per fuel). (d) German wind and sun shift the scheduled exchange additively (WXA); the capacity scaling of
# version 11 (WXC) reproduced neither the Czech nor the Danish pattern and is off by default.
HEAT = os.environ.get("HEAT", "1") == "1"; WXC = os.environ.get("WXC", "0") == "1"; WXA = os.environ.get("WXA", "1") == "1"
PREM = os.environ.get("PREM", "1") == "1"; TSTAR = os.environ.get("TSTAR", "1") == "1"; BLEND = os.environ.get("BLEND", "1") == "1"
# Version 13: (a) the exchange curve of each border (export and import bound, offset, scale, weather shift) is learned from the scheduled
# exchange alone; the balances and the Jacobian take it as given (no gradient flows back). In the joint fit the bounds came out at
# +4.3 / -6.5 GW for the Netherlands where the scheduled exchange saturates at +1.9 / -3.1: in 40% of hours the two prices are within
# 0.5-5 EUR/MWh while the exchange ranges over 4 GW, and the Jacobian rewards a steep, tall curve for that. (b) one heating threshold per zone.
XCUT = os.environ.get("XCUT", "1") == "1"; TSTARZ = os.environ.get("TSTARZ", "1") == "1"
# Version 14 (HOMO=1), step one of making the price level flow through functions of the drivers instead of through fitted levels.
# Finding: German price / (1.85 gas + 0.37 carbon) is the same function of load over 2021-2026 (gas 22..158 EUR/MWh) wherever residual load
# exceeds the available nuclear + lignite + hard coal capacity (spread 5-7% across twelve half-years, against 44-51% for the price itself),
# while the fitted model moved the price by 1.12 EUR per EUR of gas where the data moved 1.67. So:
#   cost of a gas block      = small fixed term + heat rate * gas + emission factor * carbon          (no free level; carbon absolute)
#   cost of hard coal        = small fixed term + b_coal * coal price + 0.85 * carbon                  (World Bank monthly, one month late)
#   cost of lignite          = small fixed term + emission factor * carbon (+ the latent offer premium for Germany)
#   water value (FR, AT, CH) = omega * gas-fired cost                                                 (a share, no level)
#   width of blocks 1-4      = relative width * the block's cost   (a fleet is a spread of heat rates, so its spread in EUR scales with fuel)
# The low-cost block and the curtailment block stay in absolute EUR (their thresholds are near zero by construction).
HOMO = os.environ.get("HOMO", "0") == "1"
# Version 15 (SHARE=1, needs HOMO=1), step two. Step one alone changed nothing: the level deficit is at 5-35 GW, where part-loaded German lignite and hard
# coal hold most of the model's supply slope at costs that do not move with gas, while the observed price there is a stable multiple of the gas-fired cost.
#   offer of German lignite / hard coal = own cost + share * max(gas-fired cost - own cost, 0)
# share in (0, 1), a weekly random walk on the logit scale per fuel, in place of the premium in EUR. Prior for its level: about 0.15 (2021-2023: lignite
# ran at 0.8-0.9 of capacity at any margin over its own cost while gas-fired power cost two to three times as much, so the share was near zero then).
SHARE = os.environ.get("SHARE", "0") == "1"
HYDZ = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0], float)                                                   # block A is reservoir hydro in FR, AT, CH
ALPHA0_H = np.tile(np.array([0.0, 12.0, 4.0, 3.0, 6.0, -3.0]), (9, 1)); ALPHA_SD_H = np.tile(np.array([10.0, 5.0, 3.0, 3.0, 5.0, 5.0]), (9, 1))
for _z in (1, 4, 7): ALPHA0_H[_z, 1], ALPHA_SD_H[_z, 1] = 0.0, 3.0                                    # hydro: the level comes from omega * gas-fired cost
for _z in (2, 3, 8): ALPHA0_H[_z, 1], ALPHA_SD_H[_z, 1] = 75.0, 20.0                                  # NL, BE, DK block A: other non-gas supply, level free as before
WREL0 = np.array([1.0, 0.10, 0.10, 0.10, 0.25, 1.0]); WABS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_c = pd.read_csv(f"{lc.SP}/out/wb_coal.csv", index_col=0, parse_dates=True)["Coal, South African **"] / 6.978 * 0.92           # EUR per MWh thermal (USD at 0.92)
_m = pd.DatetimeIndex(local.tz_localize(None)).to_period("M").to_timestamp() - pd.DateOffset(months=1); coal_all = _c.reindex(_c.index.union(_m.unique())).sort_index().ffill().reindex(_m).to_numpy(float)   # the previous month's value
def w3(wd, n):
    """widths as (n, Z, K) whatever build returned"""
    return wd if np.ndim(wd) == 3 else np.broadcast_to(np.asarray(wd)[None], (n,) + np.shape(wd))
TZONE = {"de": "DE", "fr": "FR", "nl": "NL", "be": "BE", "at": "AT", "pl": "DE", "cz": "DE", "ch": "AT", "dk": "DK1"}        # whose temperature stands for the zone
def _hdd(zf):
    t = F[zf][[c for c in F[zf].columns if c.startswith("temp_")]].mean(1).to_numpy(float); td = pd.Series(t).groupby(day_all).mean()
    return (0.6 * td + 0.3 * td.shift(1).bfill() + 0.1 * td.shift(2).bfill()).reindex(day_all).to_numpy(float)
teff_all = np.column_stack([_hdd(TZONE[z]) for z in ZN]); hdd_all = 2.0 * np.logaddexp(0.0, (15.0 - teff_all) / 2.0) * (1.0 if HEAT else 0.0)
drl_all = (D["rlreal_de"] - D["rl_de"]).to_numpy(float) if "rlreal_de" in D else np.zeros(len(D))        # realised minus published German residual load
_w = F["DE"][[c for c in F["DE"].columns if c.startswith("wind_")]].mean(1).to_numpy(float); _s = F["DE"][[c for c in F["DE"].columns if c.startswith("irradiance_")]].mean(1).to_numpy(float)
wx_all = np.column_stack([(_w - np.nanmean(_w)) / np.nanstd(_w), (_s - np.nanmean(_s)) / np.nanstd(_s)]) * (1.0 if (WXC or WXA) else 0.0)
HZ0 = np.array([0.05, 0.10, 0.15, 0.05, 0.08, 0.30, 0.12, 0.02, 0.12]); F10 = np.array([0.03, 0.10, 0.30])                # prior medians, GW per degree

def hddf(par, d, xp=jnp):
    """heating degrees per row and zone (n, Z); with an estimated threshold from version 12"""
    return d["hdd"] if "tstar" not in par else 2.0 * xp.logaddexp(0.0, ((par["tstar"][None] if xp.ndim(par["tstar"]) == 1 else par["tstar"]) - d["teff"]) / 2.0) * (1.0 if HEAT else 0.0)
def floors(par, d, xp=jnp):
    """must-run output of German lignite, hard coal, gas per row (n, 3)"""
    return par["floor"][None] + (par["f1"][None] * hddf(par, d, xp)[:, :1] if "f1" in par else 0.0)
def nbdem(par, d, xp=jnp):
    """shift of each neighbour's demand for its price-driven stack (n, Z-1): constant offset minus heat-led output"""
    return par["delta"][None] - (par["hz"][None, 1:] * hddf(par, d, xp)[:, 1:] if "hz" in par else 0.0)
def caps(par, wx=None, xp=jnp):
    """export and import capacity per border, (Z-1,) or (n, Z-1) with weather"""
    if wx is None or "ae" not in par: return par["Ce"], par["Ci"]
    return par["Ce"][None] * xp.exp(wx @ par["ae"]), par["Ci"][None] * xp.exp(wx @ par["ai"])

def supply(p, cap, mc, wd, sl, xp=jnp, pbar=None, lam=None):
    """pbar: day mean price per row and zone, lam: inertia weight per zone and block; the block sees lam * pbar + (1 - lam) * p"""
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    pe = p[..., None] if pbar is None else p[..., None] + lam * (pbar - p)[..., None]
    return (cap * sg((pe - mc) / wd)).sum(-1) + sl * p / 100.0
def dsupply(p, cap, mc, wd, sl, xp=jnp, pbar=None, lam=None):
    """slope with respect to the hour's own price, the day mean held fixed"""
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    pe = p[..., None] if pbar is None else p[..., None] + lam * (pbar - p)[..., None]; s = sg((pe - mc) / wd)
    return (cap * s * (1 - s) / wd * (1.0 if pbar is None else (1 - lam))).sum(-1) + sl / 100.0
def root(Q, cap, mc, wd, sl):
    """price at which supply equals Q; bisection without gradient, then one Newton step that carries the gradient"""
    capS, mcS, wdS, slS, QS = map(jax.lax.stop_gradient, (cap, mc, wd, sl, Q))
    def body(i, st):
        lo, hi = st; mid = 0.5 * (lo + hi); f = supply(mid, capS, mcS, wdS, slS) - QS; return jnp.where(f < 0, mid, lo), jnp.where(f < 0, hi, mid)
    lo, hi = jax.lax.fori_loop(0, NBIS, body, (jnp.full_like(QS, PLO), jnp.full_like(QS, PHI))); pb = 0.5 * (lo + hi)
    return pb - (supply(pb, cap, mc, wd, sl) - Q) / jax.lax.stop_gradient(dsupply(pb, cap, mc, wd, sl))

def exch(par, dp, xp=jnp, wx=None):
    """German exports to each neighbour [GW] at price difference dp = p_z - p_DE (n, Z-1); wx: German wind and sun (n, 2), None = average weather"""
    sg = jax.nn.sigmoid if xp is jnp else (lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60))))
    ce, ci = caps(par, wx, xp); return -ci + (ce + ci) * sg((dp - par["m"]) / par["kappa"]) + (wx @ par["gx"] if (wx is not None and "gx" in par) else 0.0)
GCOL = ["g_de_lignite", "g_de_coal", "g_de_gas"]; XCOL = [f"x_{z}" for z in ZN[1:]]
# Version 4 (after the energy balance by source): pumped storage is a measured part of German supply, about -5..+5 GW. It
# follows the hour's price relative to the day's mean (correlation 0.81-0.87 against 0.72 with the price level):
#     net output = P_ps * tanh((p - mean price of the day) / kappa_ps)
def day_mean(v, day):
    """mean of v over each local day, returned per element"""
    u, inv = np.unique(day, return_inverse=True); s_ = np.bincount(inv, weights=np.nan_to_num(v)); c_ = np.bincount(inv, weights=np.isfinite(v).astype(float)); return (s_ / np.maximum(c_, 1))[inv]

PM1 = np.zeros((Z, KB)); PM1[0, 1] = 1.0; PM2 = np.zeros((Z, KB)); PM2[0, 2] = 1.0
PCOAL = np.zeros((Z, KB)); PCOAL[:, 2] = 1.0; PHYD = np.zeros((Z, KB)); PHYD[:, 1] = HYDZ
def build(par, d, xp=jnp):
    """capacities (n, Z, K), marginal costs (n, Z, K), widths (Z, K), slopes (Z,), demand shift for DE (n,)"""
    cap = par["capf"][None] + par["kmeas"][None] * d["avail"]
    if par["c_e"].ndim == 2:                                                                             # version 12: offer premium of German lignite and hard coal (n, 2), zero in the last week
        ct = xp.concatenate([-xp.cumsum((2.0 * par["c_e"])[::-1], 0)[::-1], xp.zeros((1, 2))], 0)[d["week"]]
        if "wdrel" in par:                                                                               # version 14: homogeneous costs
            ea = d["eua"] + 70.0; Lg = 1.85 * d["gas"] + 0.37 * ea
            own = (par["alpha"][None] + par["beta"][None] * d["gas"][:, None, None] + d["ef"][None] * ea[:, None, None] + par["bcoal"] * d["coal"][:, None, None] * PCOAL[None]
                   + (par["omA"][:, None] * PHYD)[None] * Lg[:, None, None])
            if "phi0" in par:                                                                            # version 15: offer share of the gap to the gas-fired cost
                ct = 1.0 / (1.0 + xp.exp(-(par["phi0"][None] + xp.concatenate([-xp.cumsum((0.25 * par["c_e"])[::-1], 0)[::-1], xp.zeros((1, 2))], 0)[d["week"]])))
                gap1 = 5.0 * xp.logaddexp(0.0, (Lg - own[:, 0, 1]) / 5.0); gap2 = 5.0 * xp.logaddexp(0.0, (Lg - own[:, 0, 2]) / 5.0)
                mc = own + (ct[:, 0] * gap1)[:, None, None] * PM1[None] + (ct[:, 1] * gap2)[:, None, None] * PM2[None]
            else: mc = own + ct[:, 0][:, None, None] * PM1[None] + ct[:, 1][:, None, None] * PM2[None]
            wdt = WABS[None, None] * par["wd"][None] + (1.0 - WABS)[None, None] * par["wdrel"][None] * xp.maximum(mc, 20.0)
            mu_h = xp.concatenate([par["sh_h"], -par["sh_h"].sum(keepdims=True)]); cal = mu_h[None, :] + xp.concatenate([xp.zeros(1), par["sh_d"]])[:, None] + xp.concatenate([xp.zeros((1, 24)), par["sh_c"]])
            return cap, mc, wdt, par["sl"], cal[d["dtype"], d["hour"]], ct
        mc = par["alpha"][None] + par["beta"][None] * d["gas"][:, None, None] + d["ef"][None] * d["eua"][:, None, None] + ct[:, 0][:, None, None] * PM1[None] + ct[:, 1][:, None, None] * PM2[None]
    else:
        ct = xp.concatenate([-xp.cumsum((2.0 * par["c_e"])[::-1])[::-1], xp.zeros(1)])[d["week"]]                # common cost level, zero in the last week
        mc = par["alpha"][None] + par["beta"][None] * d["gas"][:, None, None] + d["ef"][None] * (ct + d["eua"])[:, None, None]
    mu_h = xp.concatenate([par["sh_h"], -par["sh_h"].sum(keepdims=True)]); cal = mu_h[None, :] + xp.concatenate([xp.zeros(1), par["sh_d"]])[:, None] + xp.concatenate([xp.zeros((1, 24)), par["sh_c"]])
    return cap, mc, par["wd"], par["sl"], cal[d["dtype"], d["hour"]], ct

def model(d, P=None):
    capf = numpyro.sample("capf", dist.LogNormal(jnp.log(d["cap0"]), 0.5)); kmeas = numpyro.sample("kmeas", dist.LogNormal(np.log(0.7), 0.3).expand([Z, KB])) * d["meas_mask"]
    alpha = numpyro.sample("alpha", dist.Normal(ALPHA0_H, ALPHA_SD_H) if HOMO else dist.Normal(np.broadcast_to(ALPHA0, (Z, KB)), np.broadcast_to(ALPHA_SD, (Z, KB))))
    b_0 = numpyro.sample("b_0", dist.Normal(0.0, 0.05).expand([Z])); b_a = numpyro.sample("b_a", dist.Normal(BA0 * (1.0 - HYDZ) if HOMO else BA0, np.where(HYDZ > 0, 0.05, BA_SD) if HOMO else BA_SD)); b_b = numpyro.sample("b_b", dist.Normal(BB0, np.full(9, 0.05) if HOMO else BB_SD))
    b_c = numpyro.sample("b_c", dist.Normal(1.85, 0.12).expand([Z])); b_p = numpyro.sample("b_p", dist.Normal(2.7, 0.3).expand([Z]))
    b_5 = numpyro.sample("b_5", dist.Normal(0.0, 0.05).expand([Z])); beta = jnp.stack([b_0, b_a, b_b, b_c, b_p, b_5], -1)
    wd = numpyro.sample("wd", dist.LogNormal(np.log(np.broadcast_to(WD0, (Z, KB))), 0.5)); sl = numpyro.sample("sl", dist.LogNormal(np.log(0.5), 0.5).expand([Z]))
    delta = numpyro.sample("delta", dist.Normal(0, 5).expand([Z - 1]))
    xp_ = {"Ce": numpyro.sample("Ce", dist.LogNormal(np.log(NTC), 0.5)), "Ci": numpyro.sample("Ci", dist.LogNormal(np.log(NTC), 0.5)),
           "m": numpyro.sample("m", dist.Normal(0, 10).expand([Z - 1])), "kappa": numpyro.sample("kappa", dist.LogNormal(np.log(10.0), 0.7).expand([Z - 1])),
           }
    if WXC: xp_["ae"] = numpyro.sample("ae", dist.Normal(0, 0.3).expand([2, Z - 1])); xp_["ai"] = numpyro.sample("ai", dist.Normal(0, 0.3).expand([2, Z - 1]))
    if WXA: xp_["gx"] = numpyro.sample("gx", dist.Normal(0, 0.3).expand([2, Z - 1]))                    # GW per standard deviation of German wind, sun
    f0 = numpyro.sample("floor", dist.LogNormal(np.log(np.array([2.5, 0.6, 1.5])), 0.5))                # must-run output of German lignite, hard coal, gas on a warm day [GW]
    hp = {"floor": f0, "f1": numpyro.sample("f1", dist.LogNormal(np.log(F10), 0.8)), "hz": numpyro.sample("hz", dist.LogNormal(np.log(HZ0), 0.8)), "delta": delta}
    if TSTAR: hp["tstar"] = numpyro.sample("tstar", dist.Normal(13.0, 2.5).expand([Z])) if TSTARZ else numpyro.sample("tstar", dist.Normal(13.0, 2.5))
    floor = floors(hp, d); nbd = nbdem(hp, d); hde = hp["hz"][0] * hddf(hp, d)[:, 0]                     # (n, 3), (n, Z-1), (n,)
    theta = numpyro.sample("theta", dist.Beta(2.0, 2.0)) if BLEND else 0.0; rho = numpyro.sample("rho", dist.LogNormal(np.log(0.1), 1.0).expand([3])) if BLEND else jnp.zeros(3)
    rl0 = d["rl"][:, 0] + theta * d["drl"]                                                              # the German residual load the market cleared on
    p_ps = numpyro.sample("p_ps", dist.LogNormal(np.log(5.0), 0.3)); k_ps = numpyro.sample("k_ps", dist.LogNormal(np.log(25.0), 0.5)); s_ps = numpyro.sample("s_ps", dist.LogNormal(np.log(1.0), 0.5))
    psmod = p_ps * jnp.tanh((P[:, 0] - d["pdm"]) / k_ps)                                                # pumped storage at the observed price and day mean
    par = {"capf": capf, "kmeas": kmeas, "alpha": alpha, "beta": beta, "wd": wd, "sl": sl, "c_e": numpyro.sample("c_e", dist.Normal(0, 1).expand([d["nw"] - 1] + ([2] if PREM else []))),
           **({"wdrel": numpyro.sample("wdrel", dist.LogNormal(np.log(np.broadcast_to(WREL0, (Z, KB))), 0.5)), "bcoal": numpyro.sample("bcoal", dist.Normal(2.6, 0.3)), "omA": numpyro.sample("omA", dist.LogNormal(np.log(0.9), 0.4).expand([Z]))} if HOMO else {}),
           **({"phi0": numpyro.sample("phi0", dist.Normal(-1.73, 1.0).expand([2]))} if (HOMO and SHARE) else {}),
           "sh_h": numpyro.sample("sh_h", dist.Normal(0, 3).expand([23])), "sh_d": numpyro.sample("sh_d", dist.Normal(0, 3).expand([2])), "sh_c": numpyro.sample("sh_c", dist.Normal(0, 1.5).expand([2, 24]))}
    lam_raw = numpyro.sample("lam_raw", dist.Normal(0.0, 1.5).expand([Z, 4])); lam = jnp.concatenate([jnp.zeros((Z, 1)), jax.nn.sigmoid(lam_raw), jnp.zeros((Z, 1))], 1) * (1.0 if INERT else 0.0)
    cap, mc, wd_, sl_, shift, ct = build(par, d)
    xg = {k: jax.lax.stop_gradient(v) for k, v in xp_.items()} if XCUT else xp_                          # the curve as the rest of the model sees it
    e = jnp.where(d["ok"][:, 1:] & d["ok"][:, :1], exch(xg, P[:, 1:] - P[:, :1], wx=d["wx"]), 0.0)      # exports DE -> z at the observed prices
    Q = jnp.concatenate([(rl0 + shift + e.sum(1) - floor.sum(1) - hde - psmod)[:, None], d["rl"][:, 1:] + nbd - e], 1) + capf[None, :, 5]   # floors serve demand at any price; + curtailment capacity (see CAP0)
    s = numpyro.sample("s", dist.Normal(np.log(12.0), 1.0).expand([Z])); nu = numpyro.sample("nu", dist.Gamma(4.0, 0.4))
    w = jnp.concatenate([jnp.ones(1), jnp.full(Z - 1, WNB)])
    if LIK == "p":
        phat = root(Q, cap, mc, wd_, sl_)
        numpyro.factor("ll", LW * (w[None] * d["ok"] * dist.StudentT(nu, 0.0, jnp.exp(s)[None]).log_prob(jnp.where(d["ok"], P - phat, 0.0))).sum())
    else:
        e_used = jnp.where(d["xok"], d["x"], e); ps_used = jnp.where(d["psok"], d["ps"], psmod)                 # measured exchange and storage where they exist
        Qm = jnp.concatenate([(rl0 + shift + e_used.sum(1) - floor.sum(1) - hde - ps_used)[:, None], d["rl"][:, 1:] + nbd - e_used], 1) + capf[None, :, 5]
        s_q = numpyro.sample("s_q", dist.LogNormal(np.log(4.0), 0.5).expand([Z]))
        numpyro.factor("ll", LW * (w[None] * d["ok"] * dist.StudentT(10.0, 0.0, s_q[None]).log_prob(jnp.where(d["ok"], supply(jnp.where(d["ok"], P, 0.0), cap, mc, wd_, sl_, pbar=d["pdmz"], lam=lam) - Qm, 0.0))).sum())
        if LIK == "f":
            allok = d["ok"].all(1); Pz = jnp.where(d["ok"], P, 0.0); ds = dsupply(Pz, cap, mc, wd_, sl_, pbar=d["pdmz"], lam=lam)     # (n, Z) own supply slopes at the observed prices
            sgu = jax.nn.sigmoid((Pz[:, 1:] - Pz[:, :1] - xg["m"]) / xg["kappa"]); ce_, ci_ = caps(xg, d["wx"]); de = (ce_ + ci_) * sgu * (1 - sgu) / xg["kappa"]   # exchange slopes
            dps = p_ps * (1 - jnp.tanh((Pz[:, 0] - d["pdm"]) / k_ps) ** 2) / k_ps
            Mzz = ds[:, 1:] + de; rz = jnp.where(d["cpl"], ds[:, 1:], de * ds[:, 1:] / Mzz)                           # slope a neighbour contributes to the German balance
            numpyro.factor("jac", LW * (allok * (jnp.where(d["cpl"], 0.0, jnp.log(Mzz)).sum(1) + jnp.log(ds[:, 0] + dps + rz.sum(1)))).sum())
    # measured quantities: German output by fuel at the observed price, scheduled exchange at the observed price difference
    blk = cap[:, 0, :] * jax.nn.sigmoid((P[:, :1] + lam[0][None] * (d["pdmz"][:, :1] - P[:, :1]) - mc[:, 0, :]) / (wd_[:, 0, :] if wd_.ndim == 3 else wd_[0][None]))   # (n, K) output of the German blocks
    gmod = jnp.stack([floor[:, 0] + blk[:, 1], floor[:, 1] + blk[:, 2], floor[:, 2] + blk[:, 3] + blk[:, 4]], 1) + rho[None] * ((1.0 - theta) * d["drl"])[:, None]
    sg_ = numpyro.sample("s_g", dist.LogNormal(np.log(1.0), 0.5).expand([3])); sx_ = numpyro.sample("s_x", dist.LogNormal(np.log(0.7), 0.5).expand([Z - 1]))
    numpyro.factor("lg", LW * (d["gok"] * dist.StudentT(4.0, 0.0, sg_[None]).log_prob(jnp.where(d["gok"], d["g"] - gmod, 0.0))).sum())
    numpyro.factor("lp", LW * (d["psok"] * dist.StudentT(4.0, 0.0, s_ps).log_prob(jnp.where(d["psok"], d["ps"] - psmod, 0.0))).sum())
    xfit = d["xok"] & ~d["cpl"]
    numpyro.factor("lx", LW * (xfit * dist.StudentT(4.0, 0.0, sx_[None]).log_prob(jnp.where(xfit, d["x"] - exch(xp_, P[:, 1:] - P[:, :1], wx=d["wx"]), 0.0))).sum())

def unpack(p):
    """numpy parameter dict from the guide's medians"""
    q = {k: np.asarray(v, float) for k, v in p.items()}
    q["lam"] = np.concatenate([np.zeros((Z, 1)), 1 / (1 + np.exp(-q["lam_raw"])), np.zeros((Z, 1))], 1) * (1.0 if INERT else 0.0) if "lam_raw" in q else np.zeros((Z, KB)); q["beta"] = np.stack([q["b_0"], q["b_a"], q["b_b"], q["b_c"], q["b_p"], q["b_5"]], -1); q["kmeas"] = q["kmeas"] * MEAS_MASK; return q

def equilibrium(q, d, rl=None, blend=True):
    """all nine prices that clear the coupled market (numpy; nested bisection). rl: residual loads (n, Z), default the data's"""
    if rl is None:
        rl = d["rl"]
        if blend and "theta" in q: rl = rl.copy(); rl[:, 0] = rl[:, 0] + float(q["theta"]) * d["drl"]      # in sample: the residual load the market cleared on
    cap, mc, wd, sl, shift, ct = build(q, d, np); n = len(rl); wd = w3(wd, n); wx = d.get("wx"); fl = floors(q, d, np).sum(1) if "f1" in q else q["floor"].sum()
    nbd = nbdem(q, d, np); hde = q["hz"][0] * hddf(q, d, np)[:, 0] if "hz" in q else 0.0
    PB = [None]                                                                                          # day mean prices of all zones from the previous pass
    def neigh(p0):                                                                                       # neighbours' prices and DE exports given the German price
        lo = np.full((n, Z - 1), PLO); hi = np.full((n, Z - 1), PHI)
        for _ in range(NBIS):
            mid = 0.5 * (lo + hi); f = supply(mid, cap[:, 1:], mc[:, 1:], wd[:, 1:], sl[1:], np, pbar=None if PB[0] is None else PB[0][:, 1:], lam=q["lam"][1:]) + exch(q, mid - p0[:, None], np, wx) - (rl[:, 1:] + nbd + q["capf"][1:, 5]); lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
        pz = 0.5 * (lo + hi); return pz, exch(q, pz - p0[:, None], np, wx)
    pbar = None
    for it in range(4):                                                                                 # pass 0 without storage, then with the day mean of the previous pass
        lo = np.full(n, PLO); hi = np.full(n, PHI)
        for _ in range(NBIS):
            mid = 0.5 * (lo + hi); _, e = neigh(mid); ps = 0.0 if pbar is None else q["p_ps"] * np.tanh((mid - pbar) / q["k_ps"])
            f = supply(mid[:, None], cap[:, :1], mc[:, :1], wd[:, :1], sl[:1], np, pbar=None if PB[0] is None else PB[0][:, :1], lam=q["lam"][:1])[:, 0] + ps - (rl[:, 0] + shift + e.sum(1) - fl - hde + q["capf"][0, 5]); lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
        p0 = 0.5 * (lo + hi); pbar = day_mean(p0, d["day"]) if pbar is None else 0.5 * (pbar + day_mean(p0, d["day"]))
        pz_, _ = neigh(p0); pall = np.column_stack([p0, pz_]); pbn = np.column_stack([day_mean(pall[:, i], d["day"]) for i in range(Z)]); PB[0] = pbn if PB[0] is None else 0.5 * (PB[0] + pbn)
    pz, e = neigh(p0); equilibrium.ps = q["p_ps"] * np.tanh((p0 - pbar) / q["k_ps"]); return np.column_stack([p0, pz]), e

def window(lo, hi):
    """hourly means of the quarter-hour rows lo..hi"""
    n = (hi - lo) // 4 * 4; r = lambda v: v[lo:lo + n].reshape(-1, 4, *v.shape[1:])
    H = {"P": np.nanmean(r(P_all), 1), "rl": np.nanmean(r(RL_all), 1), "gas": r(gas_all)[:, 0], "hour": r(hour_all)[:, 0], "dtype": r(dtype_all)[:, 0], "day": r(day_all)[:, 0], "pos": lo + 4 * np.arange(n // 4)}
    av = np.zeros((n // 4, Z, KB))
    for (z, k), a in AV.items(): av[:, ZN.index(z), k] = r(a)[:, 0]
    H["avail"] = av; H["g"] = np.stack([np.nanmean(r(D[c].to_numpy(float)), 1) for c in GCOL], 1); H["x"] = -np.stack([np.nanmean(r(D[c].to_numpy(float)), 1) for c in XCOL], 1)
    H["ps"] = np.nanmean(r((D["g_de_psgen"] + D["g_de_pump"]).to_numpy(float)), 1); H["eua"] = r(eua_all)[:, 0]; H["hdd"] = r(hdd_all)[:, 0]; H["wx"] = np.nanmean(r(wx_all), 1); H["teff"] = r(teff_all)[:, 0]; H["drl"] = np.nanmean(r(drl_all), 1); H["coal"] = r(coal_all)[:, 0]; return H

MEAS_MASK = np.zeros((Z, KB)); HYDRO_A = np.ones(Z); HYDRO_A[0] = 0.0                                    # German block A is lignite: no gas link
for (z, k) in MEAS: MEAS_MASK[ZN.index(z), k] = 1.0
EFM = np.tile(EF, (Z, 1))
for z, v in EF_A.items(): EFM[ZN.index(z), 1] = v
CAP0M = np.array([CAP0[z] for z in ZN], float)

def data_of(H, week, nw):
    ok = np.isfinite(H["P"]) & np.isfinite(H["rl"]) & np.isfinite(H["gas"])[:, None] & np.isfinite(H["avail"]).all((1, 2))[:, None]
    return {"rl": np.nan_to_num(H["rl"]), "gas": np.nan_to_num(H["gas"], nan=40.0), "hour": H["hour"], "dtype": H["dtype"], "week": week, "nw": nw, "avail": np.nan_to_num(H["avail"]), "ok": ok,
            "cap0": CAP0M, "meas_mask": MEAS_MASK, "hydro_a": HYDRO_A, "ef": EFM,
            "cpl": (np.abs(np.nan_to_num(H["P"][:, 1:] - H["P"][:, :1], nan=99.0)) <= CEPS) & ok[:, 1:] & ok[:, :1],
            "eua": np.nan_to_num(H["eua"]) * (1.0 if USE_EUA else 0.0), "pdmz": np.column_stack([day_mean(H["P"][:, i], H["day"]) for i in range(Z)]),
            "hdd": np.nan_to_num(H["hdd"]), "wx": np.nan_to_num(H["wx"]), "teff": np.nan_to_num(H["teff"], nan=10.0), "drl": np.nan_to_num(H["drl"]), "coal": np.nan_to_num(H["coal"], nan=13.0), "day": H["day"], "pdm": day_mean(H["P"][:, 0], H["day"]), "ps": np.nan_to_num(H["ps"]), "psok": np.isfinite(H["ps"]) & ok[:, 0],
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
        pe, e = equilibrium(q, d); ok = d["ok"][:, 0]; err = (H["P"][:, 0] - pe[:, 0])[ok] / 10; pp_, _ = equilibrium(q, d, blend=False) if "theta" in q else (pe, e)
        np.savez_compressed(f"{OUT}/{tag}_s{SEED}.npz", **{f"raw_{k}": v for k, v in raw.items()}, peq=pe.astype(np.float32), peq_pub=pp_.astype(np.float32), eeq=e.astype(np.float32), P=H["P"].astype(np.float32), pos=H["pos"], loss=loss)
        if "theta" in q: print(f"   market-expected residual load = published + {float(q['theta']):.2f} x (realised - published); with the published figure alone the in-sample MAE is {np.abs((H['P'][:, 0] - pp_[:, 0])[ok]).mean() / 10:.2f} ct", flush=True)
        print(f"{tag} seed {SEED}: {time.time() - t0:.0f}s loss {loss:.1f}; German price in sample, full equilibrium: MAE {np.abs(err).mean():.2f} ct (median error {np.median(err):+.2f}); export cap {np.round(q['Ce'], 1)}; import cap {np.round(q['Ci'], 1)}; floors {np.round(q['floor'], 1)}; storage {float(q['p_ps']):.1f} GW, kappa {float(q['k_ps']):.0f}", flush=True)
