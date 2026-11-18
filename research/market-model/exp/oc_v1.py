"""Offer curve married to a demand proxy, both learned (research harness). Hourly, Germany.
A. Offer curve: German price as a function of measured thermal output q (lignite + hard coal + gas), in the units in which it is stable:
     x  = q - mustrun(T)                                   thermal above heat-led must-run;  mustrun = m0 + m1 * heating degrees(T; threshold)
     w  = sigmoid((x - cw * S(t) - qw) / ww)               weight moving from solid-fuel pricing to gas pricing; S = available lignite + hard coal
     p  = (1 - w) * a(x) * Cs(t) + w * b(x) * L(t)         Cs = c0 + c1 * carbon (lignite's cost), L = 1.85 gas + 0.37 carbon (gas-fired cost)
     a(x) = amax * sigmoid((x - qa) / wa)                  rises from 0 (surplus) to about 1
     b(x) = b0 + s1 * softplus(x - cw * S) / 10 + s2 * softplus((x - cw * S - gk * G(t)) / 2) * 2 / 10     market heat rate, convex; G = available gas
   No fitted level, no state. Learned from (price, thermal output) pairs.
B. Demand proxy: thermal output as a function of what is known or forecast:
     q  = mustrun(T) + softplus(lin - mustrun) ,  lin = k0 + k1 * RL + k2 * (RL - day mean RL) + sum_z g_z * (RL_z - mean) + hour/day-type shift + wind, sun terms
C. Married: price forecast = mean of the curve over the proxy's error distribution (the curve is convex).
usage: oc.py fit TAGDATE | oc.py fc PERIOD [EVERY]"""
import sys, os, time, numpy as np, pandas as pd
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import star as S
SP = S.lc.SP; D = S.D; n15 = len(D); NB = S.ZN[1:]
def hourly(v): return np.nanmean(np.asarray(v, float)[: n15 // 4 * 4].reshape(-1, 4), 1)
P = hourly(D["p_de"]); Q = hourly(D["g_de_lignite"]) + hourly(D["g_de_coal"]) + hourly(D["g_de_gas"]); GAS = hourly(S.gas_all); EUA = hourly(S.eua_all) + 70.0; TEFF = hourly(S.teff_all[:, 0])
SAV = hourly(S.AV[("de", 1)]) + hourly(S.AV[("de", 2)]); GAV = hourly(S.AV[("de", 3)]); RL = np.column_stack([hourly(D[f"rl_{z}"]) for z in S.ZN]); DRL = hourly(S.drl_all); WX = np.column_stack([hourly(S.wx_all[:, 0]), hourly(S.wx_all[:, 1])])
HOUR = S.hour_all[: n15 // 4 * 4: 4]; DT = S.dtype_all[: n15 // 4 * 4: 4]; DAY = S.day_all[: n15 // 4 * 4: 4]; NH = len(P); TRAIN_H = int(os.environ.get("TRAIN_DAYS", 540)) * 24
sp = lambda x: jnp.logaddexp(0.0, x); nsp = lambda x: np.logaddexp(0.0, x); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
def mustrun(par, teff, xp=jnp): return par["m0"] + par["m1"] * 2.0 * xp.logaddexp(0.0, (par["tstar"] - teff) / 2.0)
def curve(par, q, teff, sav, gav, gas, eua, xp=jnp):
    sgm = jax.nn.sigmoid if xp is jnp else sg; sp_ = sp if xp is jnp else nsp
    x = q - mustrun(par, teff, xp); L = 1.85 * gas + 0.37 * eua; Cs = par["c0"] + par["c1"] * eua; knee = par["cw"] * sav
    w = sgm((x - knee - par["qw"]) / par["ww"]); a = par["amax"] * sgm((x - par["qa"]) / par["wa"])
    b = par["b0"] + par["s1"] * sp_(x - knee) / 10.0 + par["s2"] * 2.0 * sp_((x - knee - par["gk"] * gav) / 2.0) / 10.0
    return (1 - w) * a * Cs + w * b * L, w, L
def modelA(d):
    par = {"m0": numpyro.sample("m0", dist.LogNormal(np.log(4.5), 0.3)), "m1": numpyro.sample("m1", dist.LogNormal(np.log(0.4), 0.5)), "tstar": numpyro.sample("tstar", dist.Normal(13.0, 2.5)),
           "c0": numpyro.sample("c0", dist.Normal(12.0, 5.0)), "c1": numpyro.sample("c1", dist.Normal(1.0, 0.1)), "cw": numpyro.sample("cw", dist.LogNormal(np.log(0.5), 0.4)), "qw": numpyro.sample("qw", dist.Normal(0.0, 2.0)),
           "ww": numpyro.sample("ww", dist.LogNormal(np.log(2.5), 0.5)), "amax": numpyro.sample("amax", dist.Normal(0.9, 0.1)), "qa": numpyro.sample("qa", dist.Normal(2.0, 2.0)), "wa": numpyro.sample("wa", dist.LogNormal(np.log(1.5), 0.5)),
           "b0": numpyro.sample("b0", dist.Normal(0.9, 0.1)), "s1": numpyro.sample("s1", dist.LogNormal(np.log(0.15), 0.7)), "s2": numpyro.sample("s2", dist.LogNormal(np.log(0.5), 0.7)), "gk": numpyro.sample("gk", dist.LogNormal(np.log(0.6), 0.4))}
    ph, w, L = curve(par, d["q"], d["teff"], d["sav"], d["gav"], d["gas"], d["eua"]); s0 = numpyro.sample("s0", dist.LogNormal(np.log(5.0), 0.7)); s1_ = numpyro.sample("sr", dist.LogNormal(np.log(0.08), 0.7))
    numpyro.factor("ll", 0.1 * dist.StudentT(4.0, ph, s0 + s1_ * L).log_prob(d["p"]).sum())
def lin_of(par, d, xp=jnp):
    mu_h = xp.concatenate([par["h"], -par["h"].sum(keepdims=True)]); cal = mu_h[d["hour"]] + xp.concatenate([xp.zeros(1), par["dd"]])[d["dt"]]
    return par["k0"] + par["k1"] * d["rl"][:, 0] + par["k2"] * d["rldev"] + (d["rlz"] * par["g"][None]).sum(1) + cal + (d["wx"] * par["kx"][None]).sum(1) + par["k3"] * d["drl"]
def modelB(d, mr):
    par = {"k0": numpyro.sample("k0", dist.Normal(-8.0, 5.0)), "k1": numpyro.sample("k1", dist.Normal(0.8, 0.2)), "k2": numpyro.sample("k2", dist.Normal(-0.2, 0.2)), "g": numpyro.sample("g", dist.Normal(0.05, 0.1).expand([8])),
           "h": numpyro.sample("h", dist.Normal(0, 2).expand([23])), "dd": numpyro.sample("dd", dist.Normal(0, 2).expand([2])), "kx": numpyro.sample("kx", dist.Normal(0, 1).expand([2])), "k3": numpyro.sample("k3", dist.Normal(0.5, 0.3))}
    qh = mr + sp(lin_of(par, d) - mr); sq = numpyro.sample("sq", dist.LogNormal(np.log(1.5), 0.5)); numpyro.factor("ll", 0.1 * dist.StudentT(4.0, qh, sq).log_prob(d["q"]).sum())
def run(model, args, steps, seed=0):
    guide = AutoDelta(model, init_loc_fn=numpyro.infer.init_to_median(num_samples=10)); opt = numpyro.optim.optax_to_numpyro(__import__("optax").adam(__import__("optax").cosine_decay_schedule(0.03, steps, 0.05)))
    r = SVI(model, guide, opt, Trace_ELBO()).run(jax.random.PRNGKey(seed), steps, *args, progress_bar=False); return {k: np.asarray(v, float) for k, v in guide.median(r.params).items()}
def rows(sl, rl=None, drl=None):
    r = RL[sl] if rl is None else rl; dm = S.day_mean(r[:, 0], DAY[sl])
    return {"q": Q[sl], "p": P[sl], "teff": TEFF[sl], "sav": SAV[sl], "gav": GAV[sl], "gas": GAS[sl], "eua": EUA[sl], "rl": r, "rldev": r[:, 0] - dm, "rlz": r[:, 1:] - RLZM[None], "hour": HOUR[sl], "dt": DT[sl], "wx": WX[sl], "drl": (DRL[sl] if drl is None else drl)}
RLZM = np.nanmean(RL[:, 1:], 0)
def fit(h1, steps=1500):
    """train on hours [h1 - TRAIN_H, h1)"""
    sl = slice(max(h1 - TRAIN_H, 24 * 20), h1); d = rows(sl); ok = np.isfinite(d["q"]) & np.isfinite(d["p"]) & np.isfinite(d["sav"]) & np.isfinite(d["gav"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"])
    d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}; dj = {k: jnp.asarray(v) for k, v in d.items()}
    A = run(modelA, (dj,), steps); mr = mustrun(A, d["teff"], np); B = run(modelB, (dj, jnp.asarray(mr)), steps)
    qh = mr + nsp(lin_of(B, d, np) - mr); return A, B, d, qh
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9); GH_W = GH_W / GH_W.sum()
def married(A, B, d, noise=True):
    mr = mustrun(A, d["teff"], np); qh = mr + nsp(lin_of(B, d, np) - mr); sq = float(B["sq"]) * 1.3
    if not noise: return curve(A, qh, d["teff"], d["sav"], d["gav"], d["gas"], d["eua"], np)[0], qh
    return sum(w_ * curve(A, np.maximum(qh + x_ * sq, 0.0), d["teff"], d["sav"], d["gav"], d["gas"], d["eua"], np)[0] for x_, w_ in zip(GH_X, GH_W)), qh
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        tag = sys.argv[2]; h1 = int(np.searchsorted(S.idx.values, np.datetime64(pd.Timestamp(tag, tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)))) // 4; t0 = time.time(); A, B, d, qh = fit(h1)
        ph = curve(A, d["q"], d["teff"], d["sav"], d["gav"], d["gas"], d["eua"], np)[0]; pm, _ = married(A, B, d); np.savez(f"{SP}/out/oc_{tag}.npz", **{f"A_{k}": v for k, v in A.items()}, **{f"B_{k}": v for k, v in B.items()})
        print(f"fit to the 540 days before {tag} in {time.time() - t0:.0f}s")
        print("  offer curve: " + ", ".join(f"{k} {float(v):.2f}" for k, v in A.items())); print("  demand proxy: k0 %.1f, RL %.2f, RL minus day mean %.2f, realised-minus-published %.2f, neighbours %s, wind/sun %s, noise scale %.2f GW" % (B["k0"], B["k1"], B["k2"], B["k3"], np.round(B["g"], 2), np.round(B["kx"], 2), B["sq"]))
        print(f"  A. price given measured thermal output: MAE {np.mean(np.abs(d['p'] - ph)) / 10:.2f} ct | B. thermal output given drivers: MAE {np.mean(np.abs(d['q'] - qh)):.2f} GW, correlation {np.corrcoef(d['q'], qh)[0, 1]:.3f} | married (price from drivers): MAE {np.mean(np.abs(d['p'] - pm)) / 10:.2f} ct, bias {np.mean(d['p'] - pm) / 10:+.2f}")
        n = len(d["p"]); blk = (np.arange(n) * 9) // n; print("  married, mean error (actual - model) by ninth of the window, ct: " + " ".join(f"{np.mean((d['p'] - pm)[blk == k]) / 10:+.2f}" for k in range(9)) + " | curve alone: " + " ".join(f"{np.mean((d['p'] - ph)[blk == k]) / 10:+.2f}" for k in range(9)))
        rl0 = d["rl"][:, 0]; print("  married, by residual load bin: " + "  ".join(f"[{a},{b}) {np.mean((d['p'] - pm)[(rl0 >= a) & (rl0 < b)]) / 10:+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90))))
    if mode == "fc":
        period = sys.argv[2]; EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 7; I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
        res = {a: np.full((n, 153), np.nan) for a in ("known", "ts", "known_q")}; qerr = []; t0 = time.time()
        for j, t in enumerate(tags):
            d0 = int(I["cuts"][j]); h0 = d0 // 4
            if j % EVERY == 0: A, B, _, _ = fit(h0 - 48)                                                     # data to the end of D-1
            sl = slice(h0, h0 + 39); nh = 39
            for arm in ("known", "ts"):
                rl = RL[sl].copy()
                if arm == "ts":
                    zf = f"{NBD}/{t}.npz"
                    if not os.path.exists(zf): continue
                    zz = np.load(zf); zl = [str(x) for x in zz["zones"]]; q15 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(z)][:152] for z in NB]).astype(float)
                    rl = np.vstack([q15.reshape(38, 4, 9).mean(1), np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(z)][152:153] for z in NB]).astype(float)])
                d = rows(sl, rl=np.nan_to_num(rl), drl=np.zeros(nh)); d["gas"] = np.full(nh, float(I["gas"][j])); d["eua"] = np.full(nh, EUA[h0 - 48]); d["sav"] = np.full(nh, SAV[h0 - 24]); d["gav"] = np.full(nh, GAV[h0 - 24])
                pm, qh = married(A, B, d); res[arm][j] = np.repeat(pm / 10.0, 4)[:153]
                if arm == "known": qerr.append(Q[sl] - qh); res["known_q"][j] = np.repeat(curve(A, np.nan_to_num(Q[sl], nan=10.0), d["teff"], d["sav"], d["gav"], d["gas"], d["eua"], np)[0] / 10.0, 4)[:153]
        a_ = I["actual"].astype(float); qe = np.concatenate(qerr)
        for arm, v in res.items():
            np.savez_compressed(f"{SP}/out/ocfc_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); e = np.abs(v - a_); print(f"{period} {arm:8s}: MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a_ - v)[:, :32]):+.2f}, first 14 h {np.nanmean((a_ - v)[:, :56]):+.2f}")
        print(f"thermal output forecast with published residual loads: MAE {np.nanmean(np.abs(qe)):.2f} GW, bias {np.nanmean(qe):+.2f}; {time.time() - t0:.0f}s")
