"""Index model with informative priors (notes section 16.3). Arms by environment PT: 0 = P1 (tight priors on heat threshold and slope, neighbours'
weights; curve knots as one location and two differences; heat term centred at 10 C), 1 = P2 (P1 + prior on the pass-through of gas-fired cost at
four index points, values from 2021-2023)."""
import os, numpy as np, pandas as pd
import oc, ocx2
from oc import *
from ocx2 import rows2, pcurve, hx, SOLM, GASYM, NUCM
PT = os.environ.get("PT", "0") == "1"; PT_X = np.array([20.0, 30.0, 42.5, 60.0]); PT_M = np.array([0.51, 0.81, 1.08, 1.39]); PT_SD = float(os.environ.get("PT_SD", 0.10))
def index_p(par, d, xp=jnp):
    mu_h = xp.concatenate([par["h"], -par["h"].sum(keepdims=True)]); cal = mu_h[d["hour"]] + xp.concatenate([xp.zeros(1), par["dd"]])[d["dt"]]
    hdd = 2.0 * xp.logaddexp(0.0, (par["tstar"] - d["teff"]) / 2.0) - 2.0 * xp.logaddexp(0.0, (par["tstar"] - 10.0) / 2.0)
    return d["rl"][:, 0] + (d["rlz"] * par["wn"][None]).sum(1) - par["ks"] * (d["sol"] - SOLM) - par["kg"] * (d["gasy"] - GASYM) - par["kn"] * (d["nuc"] - NUCM) - par["kh"] * hdd + cal + par["k3"] * d["drl"]
def passthru(par, x, xp=jnp):
    """share of a change in gas-fired cost that reaches the price at index x: w(x) * b(x)"""
    sgm = jax.nn.sigmoid if xp is jnp else oc.sg; sp_ = oc.sp if xp is jnp else oc.nsp
    return sgm((x - par["qw"]) / par["ww"]) * (par["b0"] + par["s1"] * sp_((x - par["qw"]) / 3.0) * 3.0 / 10.0 + par["s2"] * sp_((x - par["qb"]) / 3.0) * 3.0 / 10.0)
EL = os.environ.get("EL", "0") == "1"; EL_M = np.array([0.78, 0.91, 0.95, 0.97]); EL_SD = np.array([0.05, 0.02, 0.02, 0.02]); L0, EUA0 = 1.85 * 35.0 + 0.37 * 75.0, 75.0
def elasticity(par, x, xp=jnp):
    """d ln price / d ln gas-fired cost at index x, at reference costs (gas 35, carbon 75): the gas-priced share of the price"""
    sgm = jax.nn.sigmoid if xp is jnp else oc.sg; w = sgm((x - par["qw"]) / par["ww"]); solid = (1 - w) * par["amax"] * sgm((x - par["qa"]) / par["wa"]) * (par["c0"] + par["c1"] * EUA0); gasp = passthru(par, x, xp) * L0
    return gasp / (solid + gasp)
def modelP(d):
    par = {"wn": numpyro.sample("wn", dist.LogNormal(np.log(0.3), 0.25).expand([8])), "ks": numpyro.sample("ks", dist.LogNormal(np.log(0.5), 0.5)), "kg": numpyro.sample("kg", dist.LogNormal(np.log(0.3), 0.7)),
           "kn": numpyro.sample("kn", dist.LogNormal(np.log(0.3), 0.5)), "kh": numpyro.sample("kh", dist.LogNormal(np.log(0.4), 0.3)), "tstar": numpyro.sample("tstar", dist.Normal(13.0, 1.0)), "h": numpyro.sample("h", dist.Normal(0, 2).expand([23])),
           "dd": numpyro.sample("dd", dist.Normal(0, 2).expand([2])), "k3": numpyro.sample("k3", dist.Normal(0.5, 0.3)), "c0": numpyro.sample("c0", dist.Normal(12.0, 5.0)), "c1": numpyro.sample("c1", dist.Normal(1.0, 0.1)),
           "amax": numpyro.sample("amax", dist.Normal(0.9, 0.15)), "qa": numpyro.sample("qa", dist.Normal(8.0, 5.0)), "wa": numpyro.sample("wa", dist.LogNormal(np.log(3.0), 0.5)), "dqw": numpyro.sample("dqw", dist.LogNormal(np.log(10.0), 0.5)),
           "ww": numpyro.sample("ww", dist.LogNormal(np.log(4.0), 0.5)), "b0": numpyro.sample("b0", dist.Normal(0.9, 0.15)), "s1": numpyro.sample("s1", dist.LogNormal(np.log(0.15), 0.7)), "s2": numpyro.sample("s2", dist.LogNormal(np.log(0.3), 0.7)),
           "dqb": numpyro.sample("dqb", dist.LogNormal(np.log(22.0), 0.4))}
    par["qw"] = par["qa"] + par["dqw"]; par["qb"] = par["qw"] + par["dqb"]
    if PT: numpyro.factor("pt", dist.Normal(jnp.asarray(PT_M), PT_SD).log_prob(passthru(par, jnp.asarray(PT_X))).sum())
    if EL: numpyro.factor("el", dist.Normal(jnp.asarray(EL_M), jnp.asarray(EL_SD)).log_prob(elasticity(par, jnp.asarray(PT_X))).sum())
    ph, L = pcurve(par, index_p(par, d), d["gas"], d["eua"]); s0 = numpyro.sample("s0", dist.LogNormal(np.log(5.0), 0.7)); sr = numpyro.sample("sr", dist.LogNormal(np.log(0.08), 0.7))
    numpyro.factor("ll", 0.1 * dist.StudentT(4.0, ph, s0 + sr * L).log_prob(d["p"]).sum())
def pred_p(A, d): return pcurve(A, index_p(A, d, np), d["gas"], d["eua"], np)[0]
def fit_p(h1, steps=8000):
    sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows2(sl); ok = np.isfinite(d["p"]) & np.isfinite(d["sol"]) & np.isfinite(d["sol2"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]); dayk = oc.DAY[sl][ok]
    d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}; A = run(modelP, ({k: jnp.asarray(v) for k, v in d.items()},), steps); A["qw"] = A["qa"] + A["dqw"]; A["qb"] = A["qw"] + A["dqb"]
    rel = (d["p"] - pred_p(A, d)) / (1.85 * d["gas"] + 0.37 * d["eua"]); dm = pd.Series(rel).groupby(dayk).mean(); a_, b_ = dm.shift(1).to_numpy()[1:], dm.to_numpy()[1:]; m = np.isfinite(a_) & np.isfinite(b_)
    A["rho"] = np.array(float(np.sum(a_[m] * b_[m]) / np.sum(a_[m] ** 2))); A["mae_in"] = np.array(float(np.mean(np.abs(d["p"] - pred_p(A, d))) / 10)); return A, d
if __name__ == "__main__":
    import sys, time
    t0 = time.time(); A, d = fit_p(hx(sys.argv[1])); e = (d["p"] - pred_p(A, d)) / 10
    print(f"{'P3' if EL else 'P2' if PT else 'P1'} fit to the 540 days before {sys.argv[1]} in {time.time() - t0:.0f}s: in sample MAE {np.mean(np.abs(e)):.3f} ct, bias {np.mean(e):+.2f} | heat threshold {float(A['tstar']):.1f} C ({(float(A['tstar']) - 13) / 1.0:+.1f} prior sd), slope {float(A['kh']):.2f} | neighbours " + " ".join(f"{z} {v:.2f}" for z, v in zip(NB, A["wn"])) + f" (CH {np.log(float(A['wn'][6]) / 0.3) / 0.25:+.1f} prior sd)")
    print(f"   knots qa {float(A['qa']):.1f} qw {float(A['qw']):.1f} qb {float(A['qb']):.1f} ww {float(A['ww']):.1f} c0 {float(A['c0']):.1f} amax {float(A['amax']):.2f} b0 {float(A['b0']):.2f} | pass-through at 20 / 30 / 42.5 / 60 GW: " + " ".join(f"{v:.2f}" for v in passthru(A, PT_X, np)) + f"  (2021-2023: {' '.join(f'{v:.2f}' for v in PT_M)}) | elasticity: " + " ".join(f"{v:.2f}" for v in elasticity(A, PT_X, np)) + f"  (2021-2023: {' '.join(f'{v:.2f}' for v in EL_M)})")
