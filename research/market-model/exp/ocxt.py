"""Index model, third version = second version + a trough-day term (notes section 15.4).
  On a day whose lowest index falls into surplus, every hour of that day is priced as if the index were higher:
      x_h  ->  x_h + kc * sigmoid((qc - min over the day's hours of x) / wc)
  kc [GW], qc [GW] and wc [GW] are learned with everything else. Reading: plant that cannot ride through the midday trough stays off for the day, so the
  rest of the day clears higher up the offer curve. The term needs only the day's load profile, so it is available for D+2 from the load forecast."""
import os, numpy as np, pandas as pd
import oc, ocx2
from oc import *
from ocx2 import rows2, index2, pcurve, hx, SOL, GASY, NUC
CENS = os.environ.get("CENS", "0") == "1"; CENS_AT = float(os.environ.get("CENS_AT", 0.0)); TROUGH = os.environ.get("TROUGH", "1") == "1"
def gather(days):
    """per row: the row numbers of all rows of the same day (padded with the row itself)"""
    days = np.asarray(days); n = len(days); G = np.tile(np.arange(n)[:, None], (1, 25)); start = np.r_[0, np.flatnonzero(np.diff(days) != 0) + 1]; end = np.r_[start[1:], n]
    for a, b in zip(start, end): m = min(b - a, 25); G[a:b, :m] = np.arange(a, a + m)[None]
    return G
def rows_t(sl, rl=None, drl=None, state_at=None):
    d = rows2(sl, rl=rl, drl=drl, state_at=state_at); d["G"] = gather(oc.DAY[sl]); return d
def trough(par, x0, d, xp=jnp):
    if not TROUGH: return 0.0 * x0
    sgm = jax.nn.sigmoid if xp is jnp else oc.sg; xm = x0 if d.get("okx") is None else xp.where(d["okx"], x0, 1e3); return par["kc"] * sgm((par["qc"] - xm[d["G"]].min(1)) / par["wc"])
def index_t(par, d, xp=jnp):
    x0 = index2(par, d, xp); return x0 + trough(par, x0, d, xp)
def modelT(d):
    par = {"wn": numpyro.sample("wn", dist.LogNormal(np.log(0.3), 0.5).expand([8])), "ks": numpyro.sample("ks", dist.LogNormal(np.log(0.5), 0.5)), "kg": numpyro.sample("kg", dist.LogNormal(np.log(0.3), 0.7)),
           "kn": numpyro.sample("kn", dist.LogNormal(np.log(0.3), 0.5)), "kh": numpyro.sample("kh", dist.LogNormal(np.log(0.4), 0.5)), "tstar": numpyro.sample("tstar", dist.Normal(13.0, 2.5)), "h": numpyro.sample("h", dist.Normal(0, 2).expand([23])),
           "dd": numpyro.sample("dd", dist.Normal(0, 2).expand([2])), "k3": numpyro.sample("k3", dist.Normal(0.5, 0.3)), "c0": numpyro.sample("c0", dist.Normal(12.0, 5.0)), "c1": numpyro.sample("c1", dist.Normal(1.0, 0.1)),
           "amax": numpyro.sample("amax", dist.Normal(0.9, 0.15)), "qa": numpyro.sample("qa", dist.Normal(8.0, 5.0)), "wa": numpyro.sample("wa", dist.LogNormal(np.log(3.0), 0.5)), "qw": numpyro.sample("qw", dist.Normal(18.0, 6.0)),
           "ww": numpyro.sample("ww", dist.LogNormal(np.log(4.0), 0.5)), "b0": numpyro.sample("b0", dist.Normal(0.9, 0.15)), "s1": numpyro.sample("s1", dist.LogNormal(np.log(0.15), 0.7)), "s2": numpyro.sample("s2", dist.LogNormal(np.log(0.3), 0.7)),
           "qb": numpyro.sample("qb", dist.Normal(40.0, 8.0)),
           "kc": numpyro.sample("kc", dist.LogNormal(np.log(2.0), 0.8)), "qc": numpyro.sample("qc", dist.Normal(8.0, 6.0)), "wc": numpyro.sample("wc", dist.LogNormal(np.log(3.0), 0.5))}
    ph, L = pcurve(par, index_t(par, d), d["gas"], d["eua"]); s0 = numpyro.sample("s0", dist.LogNormal(np.log(5.0), 0.7)); sr = numpyro.sample("sr", dist.LogNormal(np.log(0.08), 0.7))
    D_ = dist.StudentT(4.0, ph, s0 + sr * L); lp = D_.log_prob(d["p"])
    if CENS: lp = jnp.where(d["p"] <= CENS_AT, jnp.log(jnp.clip(D_.cdf(jnp.full_like(ph, CENS_AT)), 1e-10, 1.0)), lp)     # prices at or below the bound are censored: all the curve is told is that the price was that low
    if not TROUGH: pass
    numpyro.factor("ll", 0.1 * (d["ok"] * lp).sum())
def pred_t(A, d): return pcurve(A, index_t(A, d, np), d["gas"], d["eua"], np)[0]
def fit_t(h1, steps=2500):
    sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows_t(sl); okx = np.isfinite(d["sol"]) & np.isfinite(d["sol2"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]) & np.isfinite(d["eua"]) & np.isfinite(d["drl"]) & np.isfinite(d["teff"])
    ok = okx & np.isfinite(d["p"]); d = {k: (v if k == "G" else np.nan_to_num(v)) for k, v in d.items()}; d["okx"] = okx; d["ok"] = ok.astype(float); A = run(modelT, ({k: jnp.asarray(v) for k, v in d.items()},), steps)
    rel = ((d["p"] - pred_t(A, d)) / (1.85 * d["gas"] + 0.37 * d["eua"]))[ok]; dm = pd.Series(rel).groupby(oc.DAY[sl][ok]).mean(); a_, b_ = dm.shift(1).to_numpy()[1:], dm.to_numpy()[1:]; m = np.isfinite(a_) & np.isfinite(b_)
    A["rho"] = np.array(float(np.sum(a_[m] * b_[m]) / np.sum(a_[m] ** 2))); return A, d
if __name__ == "__main__":
    import sys, time
    t0 = time.time(); A, d = fit_t(hx(sys.argv[1])); ok = d["ok"] > 0; ph = pred_t(A, d); e = (d["p"] - ph)[ok] / 10; n = ok.sum(); blk = (np.arange(n) * 9) // n; x0 = index2(A, d, np); tr = trough(A, x0, d, np)
    print(f"fit to the 540 days before {sys.argv[1]} in {time.time() - t0:.0f}s: in sample MAE {np.mean(np.abs(e)):.2f} ct, bias {np.mean(e):+.2f}")
    print(f"  trough term: up to {float(A['kc']):.2f} GW, half of it when the day's lowest index is {float(A['qc']):.1f} GW, width {float(A['wc']):.1f} GW (curve's surplus point qa {float(A['qa']):.1f}); mean shift over all hours {tr[ok].mean():.2f} GW, share of hours with more than half of it {100 * np.mean(tr[ok] > 0.5 * float(A['kc'])):.0f}%")
    print("  index: neighbours' weights " + " ".join(f"{z} {v:.2f}" for z, v in zip(NB, A["wn"])) + f"; solid {float(A['ks']):.2f}; gas {float(A['kg']):.2f}; neighbours' nuclear {float(A['kn']):.2f}; heat {float(A['kh']):.2f} GW per degree below {float(A['tstar']):.1f} C; realised-load term {float(A['k3']):.2f}")
    print("  error (actual - model) by ninth of the window, ct: " + " ".join(f"{np.mean(e[blk == k]):+.2f}" for k in range(9)))
