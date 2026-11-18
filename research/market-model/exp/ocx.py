"""Curve over an exogenous regional load index, in cost units (research harness; specification fixed before any forecast was scored).
  index  x = German residual load + wn * (neighbours' residual loads - their means) - ks * (available lignite + hard coal - mean)
             - kh * heating degrees - kf * (French nuclear availability - mean) + shift[hour] + shift[day type]                      [GW]
  price  p = (1 - w(x)) * a(x) * Cs + w(x) * b(x) * L ,  Cs = c0 + c1 * carbon (lignite's cost), L = 1.85 gas + 0.37 carbon (gas-fired cost)
         a rises from 0 to about 1 (surplus -> solid fuel), w moves the weight to gas pricing, b is a convex market heat rate.
No state, no fitted price level. Inputs at a forecast origin: residual loads (forecast or published), availability as measured to D-1, gas known at the
origin, last carbon auction, temperature.   usage: ocx.py fit DATE | ocx.py fwd FITDATE ENDDATE | ocx.py fc PERIOD [EVERY]"""
import sys, os, time, numpy as np, pandas as pd
import oc
from oc import *
def index(par, d, xp=jnp):
    mu_h = xp.concatenate([par["h"], -par["h"].sum(keepdims=True)]); cal = mu_h[d["hour"]] + xp.concatenate([xp.zeros(1), par["dd"]])[d["dt"]]
    hdd = 2.0 * xp.logaddexp(0.0, (par["tstar"] - d["teff"]) / 2.0)
    return d["rl"][:, 0] + (d["rlz"] * par["wn"][None]).sum(1) - par["ks"] * (d["sav"] - SAVM) - par["kh"] * hdd - par["kf"] * (d["frn"] - FRNM) + cal + par["k3"] * d["drl"]
def pcurve(par, x, gas, eua, xp=jnp):
    sgm = jax.nn.sigmoid if xp is jnp else oc.sg; sp_ = oc.sp if xp is jnp else oc.nsp
    L = 1.85 * gas + 0.37 * eua; Cs = par["c0"] + par["c1"] * eua; w = sgm((x - par["qw"]) / par["ww"]); a = par["amax"] * sgm((x - par["qa"]) / par["wa"])
    b = par["b0"] + par["s1"] * sp_((x - par["qw"]) / 3.0) * 3.0 / 10.0 + par["s2"] * sp_((x - par["qb"]) / 3.0) * 3.0 / 10.0
    return (1 - w) * a * Cs + w * b * L, L
SAVM = float(np.nanmean(oc.SAV))
def modelX(d):
    par = {"wn": numpyro.sample("wn", dist.LogNormal(np.log(0.3), 0.5).expand([8])), "ks": numpyro.sample("ks", dist.LogNormal(np.log(0.5), 0.5)), "kh": numpyro.sample("kh", dist.LogNormal(np.log(0.4), 0.5)),
           "kf": numpyro.sample("kf", dist.LogNormal(np.log(0.3), 0.5)), "tstar": numpyro.sample("tstar", dist.Normal(13.0, 2.5)), "h": numpyro.sample("h", dist.Normal(0, 2).expand([23])), "dd": numpyro.sample("dd", dist.Normal(0, 2).expand([2])),
           "k3": numpyro.sample("k3", dist.Normal(0.5, 0.3)), "c0": numpyro.sample("c0", dist.Normal(12.0, 5.0)), "c1": numpyro.sample("c1", dist.Normal(1.0, 0.1)), "amax": numpyro.sample("amax", dist.Normal(0.9, 0.15)),
           "qa": numpyro.sample("qa", dist.Normal(8.0, 5.0)), "wa": numpyro.sample("wa", dist.LogNormal(np.log(3.0), 0.5)), "qw": numpyro.sample("qw", dist.Normal(18.0, 6.0)), "ww": numpyro.sample("ww", dist.LogNormal(np.log(4.0), 0.5)),
           "b0": numpyro.sample("b0", dist.Normal(0.9, 0.15)), "s1": numpyro.sample("s1", dist.LogNormal(np.log(0.15), 0.7)), "s2": numpyro.sample("s2", dist.LogNormal(np.log(0.3), 0.7)), "qb": numpyro.sample("qb", dist.Normal(40.0, 8.0))}
    ph, L = pcurve(par, index(par, d), d["gas"], d["eua"]); s0 = numpyro.sample("s0", dist.LogNormal(np.log(5.0), 0.7)); sr = numpyro.sample("sr", dist.LogNormal(np.log(0.08), 0.7))
    numpyro.factor("ll", 0.1 * dist.StudentT(4.0, ph, s0 + sr * L).log_prob(d["p"]).sum())
def fitx(h1, steps=2500):
    sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows(sl); ok = np.isfinite(d["p"]) & np.isfinite(d["sav"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]) & np.isfinite(d["frn"])
    d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}; A = run(modelX, ({k: jnp.asarray(v) for k, v in d.items()},), steps); return A, d
def pred(A, d): return pcurve(A, index(A, d, np), d["gas"], d["eua"], np)[0]
hx = lambda t_: int(np.searchsorted(S.idx.values, np.datetime64(pd.Timestamp(t_, tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)))) // 4
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        t0 = time.time(); A, d = fitx(hx(sys.argv[2])); ph = pred(A, d); n = len(ph); blk = (np.arange(n) * 9) // n; rl0 = d["rl"][:, 0]; np.savez(f"{SP}/out/ocx_{sys.argv[2]}.npz", **A)
        print(f"fit to the 540 days before {sys.argv[2]} in {time.time() - t0:.0f}s: in sample MAE {np.mean(np.abs(d['p'] - ph)) / 10:.2f} ct, bias {np.mean(d['p'] - ph) / 10:+.2f}")
        print("  index: neighbours' weights " + " ".join(f"{z} {v:.2f}" for z, v in zip(NB, A["wn"])) + f"; solid availability {float(A['ks']):.2f}; heat {float(A['kh']):.2f} GW per degree below {float(A['tstar']):.1f} C; French nuclear {float(A['kf']):.2f}; realised-minus-published {float(A['k3']):.2f}")
        print("  curve: " + ", ".join(f"{k} {float(A[k]):.2f}" for k in ("c0", "c1", "amax", "qa", "wa", "qw", "ww", "b0", "s1", "s2", "qb", "s0", "sr")))
        print("  error (actual - model) by ninth of the window, ct: " + " ".join(f"{np.mean((d['p'] - ph)[blk == k]) / 10:+.2f}" for k in range(9)) + " | by residual load bin: " + "  ".join(f"[{a},{b}) {np.mean((d['p'] - ph)[(rl0 >= a) & (rl0 < b)]) / 10:+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90))))
    if mode == "fwd":
        h1, h2 = hx(sys.argv[2]), hx(sys.argv[3]); A, _ = fitx(h1); sl = slice(h1, h2); d = rows(sl, drl=np.zeros(h2 - h1)); ok = np.isfinite(d["p"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["sav"]) & np.isfinite(d["gas"]); d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}
        err = (d["p"] - pred(A, d)) / 10; mo = np.arange(len(err)) // (30 * 24); rl0 = d["rl"][:, 0]
        print(f"fit to the 540 days before {sys.argv[2]}, applied unchanged to {sys.argv[3]}: MAE {np.mean(np.abs(err)):.2f} ct, bias {np.mean(err):+.2f}; by month: " + " ".join(f"{np.mean(err[mo == k]):+.2f}" for k in range(int(mo.max()) + 1) if (mo == k).sum() > 100) + " | by residual load bin: " + "  ".join(f"[{a},{b}) {np.mean(err[(rl0 >= a) & (rl0 < b)]):+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90)) if ((rl0 >= a) & (rl0 < b)).sum() > 30))
    if mode == "fc":
        period = sys.argv[2]; EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 7; I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
        res = {a: np.full((n, 153), np.nan) for a in ("known", "ts")}; t0 = time.time()
        for j, t in enumerate(tags):
            h0 = int(I["cuts"][j]) // 4
            if j % EVERY == 0: A, _ = fitx(h0 - 48)
            sl = slice(h0, h0 + 39); nh = 39
            for arm in ("known", "ts"):
                rl = RL[sl].copy()
                if arm == "ts":
                    zf = f"{NBD}/{t}.npz"
                    if not os.path.exists(zf): continue
                    zz = np.load(zf); zl = [str(x) for x in zz["zones"]]; q15 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(z)][:152] for z in NB]).astype(float)
                    rl = np.vstack([q15.reshape(38, 4, 9).mean(1), np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(z)][152:153] for z in NB]).astype(float)])
                d = rows(sl, rl=np.nan_to_num(rl), drl=np.zeros(nh)); d["gas"] = np.full(nh, float(I["gas"][j])); d["eua"] = np.full(nh, EUA[h0 - 48]); d["sav"] = np.full(nh, SAV[h0 - 24]); d["frn"] = np.full(nh, FRN[h0 - 24])
                res[arm][j] = np.repeat(pred(A, d) / 10.0, 4)[:153]
        a_ = I["actual"].astype(float)
        for arm, v in res.items():
            np.savez_compressed(f"{SP}/out/ocxfc_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); e = np.abs(v - a_); print(f"{period} {arm:6s}: MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a_ - v)[:, :32]):+.2f}, first 14 h {np.nanmean((a_ - v)[:, :56]):+.2f}   ({time.time() - t0:.0f}s)")
