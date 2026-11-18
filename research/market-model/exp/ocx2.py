"""Index model, second version (specification fixed from the screening on data before 2026-03-31; exp/screen.py):
  index  x = German residual load + wn * neighbours' residual loads - ks * solid availability - kg * gas availability - kn * pooled nuclear availability of
             the neighbours - kh * heating degrees + calendar
         all three availabilities are built from days <= d-3, i.e. exactly what is known at 14:00 two days ahead (the first version used a one-day lag in
         training and a two-day lag in forecasting); solid: 14-day max of German lignite + hard coal; gas: the last known day's max; nuclear: 14-day max of
         FR + BE + CH + CZ + NL.
  price  as in ocx.py (blend of lignite cost and gas-fired cost, learned shapes; no state).
  Online level correction, reported separately: the model's mean relative error on D+1 (prices known at the origin, loads forecast) times a persistence
  factor estimated in the training window.
usage: ocx2.py fit DATE | ocx2.py fwd FITDATE ENDDATE | ocx2.py fc PERIOD [EVERY]"""
import sys, os, time, numpy as np, pandas as pd
import oc, ocx
from oc import *
from ocx import pcurve, hx
N = pd.read_parquet(f"{SP}/out/nb_supply.parquet"); HT = pd.DatetimeIndex(S.idx[: oc.n15 // 4 * 4: 4]); N = N.reindex(HT if HT.tz is not None else HT.tz_localize("UTC"))
def lag3(hourly_vals, days, fn):
    """per hour: statistic over the `days` days ending three days before that hour's day"""
    s = pd.Series(hourly_vals); dm = s.groupby(oc.DAY).max() if fn == "max" else s.groupby(oc.DAY).mean(); dm = dm.reindex(range(int(oc.DAY.min()), int(oc.DAY.max()) + 1))
    r = (dm.rolling(days, min_periods=max(1, days // 2)).max() if fn == "max" else dm.rolling(days, min_periods=max(1, days // 2)).mean()).shift(3); return r.reindex(oc.DAY).to_numpy(float)
SOL = lag3(oc.hourly(S.D["g_de_lignite"]) + oc.hourly(S.D["g_de_coal"]), 14, "max"); GASY = lag3(oc.hourly(S.D["g_de_gas"]), 1, "max")
NUC = sum(np.nan_to_num(lag3(N[c].to_numpy(float), 14, "max"), nan=float(np.nanmean(N[c]))) for c in ("nuc_fr", "nuc_be", "nuc_ch", "nuc_cz", "nuc_nl"))
if os.environ.get("SOLMODE", "max") == "inmoney":
    # version 3: lignite enters by what it delivered when clearly in the money (margin 30..120 EUR/MWh over carbon + 12), 14-day mean ending three days earlier,
    # instead of its 14-day maximum; hard coal stays at its 14-day maximum. (Both passed the pre-summer screening about equally; they part ways from August 2026.)
    _lig = oc.hourly(S.D["g_de_lignite"]); _m = oc.P - (12.0 + oc.EUA); _in = (_m >= 30) & (_m < 120) & np.isfinite(_lig); _d = pd.Series(np.where(_in, _lig, 0.0)).groupby(oc.DAY).sum(); _n = pd.Series(_in.astype(float)).groupby(oc.DAY).sum()
    _rng = range(int(oc.DAY.min()), int(oc.DAY.max()) + 1); _num = _d.reindex(_rng).fillna(0).rolling(14, min_periods=3).sum(); _den = _n.reindex(_rng).fillna(0).rolling(14, min_periods=3).sum(); _ligin = (_num / _den.where(_den >= 12)).ffill().shift(3).reindex(oc.DAY).to_numpy(float)
    SOL = _ligin + lag3(oc.hourly(S.D["g_de_coal"]), 14, "max")
OWNER = os.environ.get("OWNER", "0") == "1"; SOL2 = np.zeros(len(SOL))
if OWNER:
    # version 4: solid fuel by owner class instead of one fleet term. Per class: output of its blocks in the hours when the fuel was clearly in the money, 14-day mean
    # built from days <= d-7 (block data are published five days late). Class 1: RWE + EnBW blocks (lignite and hard coal); class 2: LEAG, Saale and the other hard-coal owners.
    _O = pd.read_parquet(f"{SP}/out/owner_states.parquet"); _g = lambda ks: sum(_O[f"zin_{k}"].to_numpy(float) for k in ks)[: len(SOL)]
    SOL = _g(("lig_rwe", "lig_enbw", "coal_enbw")); SOL2 = _g(("lig_leag", "lig_saale", "coal_other"))
    if os.environ.get("OWNERSUM", "0") == "1": SOL = SOL + SOL2; SOL2 = np.zeros(len(SOL)); OWNER = False          # control: the same blocks, lag and statistic as one term
SOLM, GASYM, NUCM = float(np.nanmean(SOL)), float(np.nanmean(GASY)), float(np.nanmean(NUC)); SOL2M = float(np.nanmean(SOL2)) if OWNER else 0.0; FCTAG = os.environ.get("FCTAG", "ocx2fc")
def rows2(sl, rl=None, drl=None, state_at=None):
    d = rows(sl, rl=rl, drl=drl); k = sl if state_at is None else slice(state_at, state_at + 1)
    d["sol"] = SOL[sl] if state_at is None else np.full(len(d["p"]), SOL[state_at]); d["gasy"] = GASY[sl] if state_at is None else np.full(len(d["p"]), GASY[state_at]); d["nuc"] = NUC[sl] if state_at is None else np.full(len(d["p"]), NUC[state_at]); d["sol2"] = SOL2[sl] if state_at is None else np.full(len(d["p"]), SOL2[state_at]); return d
def index2(par, d, xp=jnp):
    mu_h = xp.concatenate([par["h"], -par["h"].sum(keepdims=True)]); cal = mu_h[d["hour"]] + xp.concatenate([xp.zeros(1), par["dd"]])[d["dt"]]; hdd = 2.0 * xp.logaddexp(0.0, (par["tstar"] - d["teff"]) / 2.0)
    return d["rl"][:, 0] + (d["rlz"] * par["wn"][None]).sum(1) - par["ks"] * (d["sol"] - SOLM) - (par["ks2"] * (d["sol2"] - SOL2M) if "ks2" in par else 0.0) - par["kg"] * (d["gasy"] - GASYM) - par["kn"] * (d["nuc"] - NUCM) - par["kh"] * hdd + cal + par["k3"] * d["drl"]
def modelX2(d):
    par = {"wn": numpyro.sample("wn", dist.LogNormal(np.log(0.3), 0.5).expand([8])), "ks": numpyro.sample("ks", dist.LogNormal(np.log(0.5), 0.5)), "kg": numpyro.sample("kg", dist.LogNormal(np.log(0.3), 0.7)),
           "kn": numpyro.sample("kn", dist.LogNormal(np.log(0.3), 0.5)), "kh": numpyro.sample("kh", dist.LogNormal(np.log(0.4), 0.5)), "tstar": numpyro.sample("tstar", dist.Normal(13.0, 2.5)), "h": numpyro.sample("h", dist.Normal(0, 2).expand([23])),
           "dd": numpyro.sample("dd", dist.Normal(0, 2).expand([2])), "k3": numpyro.sample("k3", dist.Normal(0.5, 0.3)), "c0": numpyro.sample("c0", dist.Normal(12.0, 5.0)), "c1": numpyro.sample("c1", dist.Normal(1.0, 0.1)),
           "amax": numpyro.sample("amax", dist.Normal(0.9, 0.15)), "qa": numpyro.sample("qa", dist.Normal(8.0, 5.0)), "wa": numpyro.sample("wa", dist.LogNormal(np.log(3.0), 0.5)), "qw": numpyro.sample("qw", dist.Normal(18.0, 6.0)),
           "ww": numpyro.sample("ww", dist.LogNormal(np.log(4.0), 0.5)), "b0": numpyro.sample("b0", dist.Normal(0.9, 0.15)), "s1": numpyro.sample("s1", dist.LogNormal(np.log(0.15), 0.7)), "s2": numpyro.sample("s2", dist.LogNormal(np.log(0.3), 0.7)),
           "qb": numpyro.sample("qb", dist.Normal(40.0, 8.0))}
    if OWNER: par["ks2"] = numpyro.sample("ks2", dist.LogNormal(np.log(0.5), 0.5))
    for k_, v_ in FIXED.items(): par[k_] = v_                                                                # curve-shape parameters held at given values (oracle or prior tests)
    ph, L = pcurve(par, index2(par, d), d["gas"], d["eua"]); s0 = numpyro.sample("s0", dist.LogNormal(np.log(5.0), 0.7)); sr = numpyro.sample("sr", dist.LogNormal(np.log(0.08), 0.7))
    numpyro.factor("ll", 0.1 * dist.StudentT(4.0, ph, s0 + sr * L).log_prob(d["p"]).sum())
FIXED = {}
if os.environ.get("FIXCURVE"):
    _z = np.load(os.environ["FIXCURVE"]); FIXED = {k: float(_z[k]) for k in os.environ.get("FIXKEYS", "qw,ww,b0,s1,s2,qb,amax,qa,wa").split(",")}
def pred2(A, d): return pcurve(A, index2(A, d, np), d["gas"], d["eua"], np)[0]
def fitx2(h1, steps=2500):
    sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows2(sl); ok = np.isfinite(d["p"]) & np.isfinite(d["sol"]) & np.isfinite(d["sol2"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]); dayk = oc.DAY[sl][ok]
    d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}; A = run(modelX2, ({k: jnp.asarray(v) for k, v in d.items()},), steps)
    for k_, v_ in FIXED.items(): A[k_] = np.array(v_)
    rel = (d["p"] - pred2(A, d)) / (1.85 * d["gas"] + 0.37 * d["eua"]); dm = pd.Series(rel).groupby(dayk).mean(); a_, b_ = dm.shift(1).to_numpy()[1:], dm.to_numpy()[1:]; m = np.isfinite(a_) & np.isfinite(b_)
    A["rho"] = np.array(float(np.sum(a_[m] * b_[m]) / np.sum(a_[m] ** 2))); return A, d                    # persistence of the daily mean relative error
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        t0 = time.time(); A, d = fitx2(hx(sys.argv[2])); ph = pred2(A, d); n = len(ph); blk = (np.arange(n) * 9) // n; rl0 = d["rl"][:, 0]; np.savez(f"{SP}/out/ocx2_{sys.argv[2]}.npz", **A)
        print(f"fit to the 540 days before {sys.argv[2]} in {time.time() - t0:.0f}s: in sample MAE {np.mean(np.abs(d['p'] - ph)) / 10:.2f} ct, bias {np.mean(d['p'] - ph) / 10:+.2f}; persistence of the daily relative error {float(A['rho']):.2f}")
        print("  index: neighbours' weights " + " ".join(f"{z} {v:.2f}" for z, v in zip(NB, A["wn"])) + f"; solid {'(large portfolios)' if OWNER else 'availability'} {float(A['ks']):.2f}" + (f", solid (other owners) {float(A['ks2']):.2f}" if 'ks2' in A else "") + f"; gas availability {float(A['kg']):.2f}; neighbours' nuclear {float(A['kn']):.2f}; heat {float(A['kh']):.2f} GW per degree below {float(A['tstar']):.1f} C")
        print("  error (actual - model) by ninth of the window, ct: " + " ".join(f"{np.mean((d['p'] - ph)[blk == k]) / 10:+.2f}" for k in range(9)) + " | by residual load bin: " + "  ".join(f"[{a},{b}) {np.mean((d['p'] - ph)[(rl0 >= a) & (rl0 < b)]) / 10:+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90))))
    if mode == "fwd":
        h1, h2 = hx(sys.argv[2]), hx(sys.argv[3]); A, _ = fitx2(h1); sl = slice(h1, h2); d = rows2(sl, drl=np.zeros(h2 - h1)); ok = np.isfinite(d["p"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["sol"]) & np.isfinite(d["gas"]) & np.isfinite(d["nuc"]); d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}
        err = (d["p"] - pred2(A, d)) / 10; mo = np.arange(len(err)) // (30 * 24); rl0 = d["rl"][:, 0]
        print(f"fit to the 540 days before {sys.argv[2]}, applied unchanged to {sys.argv[3]}: MAE {np.mean(np.abs(err)):.2f} ct, bias {np.mean(err):+.2f}; by month: " + " ".join(f"{np.mean(err[mo == k]):+.2f}" for k in range(int(mo.max()) + 1) if (mo == k).sum() > 100) + " | by residual load bin: " + "  ".join(f"[{a},{b}) {np.mean(err[(rl0 >= a) & (rl0 < b)]):+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90)) if ((rl0 >= a) & (rl0 < b)).sum() > 30))
    if mode == "fc":
        period = sys.argv[2]; EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 7; I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
        res = {a: np.full((n, 153), np.nan) for a in ("known", "ts", "known_oc", "ts_oc", "known_oc2", "ts_oc2", "known_oc3", "ts_oc3")}; t0 = time.time(); rhos = []; MEM, HW = 2.0, 1.0   # the settings the earlier structural model's correction layer settled on
        for j, t in enumerate(tags):
            h0 = int(I["cuts"][j]) // 4
            if j % EVERY == 0: A, _ = fitx2(h0 - 48); rhos.append(float(A["rho"]))
            for arm in ("known", "ts"):
                rl2 = RL[h0:h0 + 39].copy(); rl1 = RL[h0 - 24:h0].copy()
                if arm == "ts":
                    zf = f"{NBD}/{t}.npz"
                    if not os.path.exists(zf): continue
                    zz = np.load(zf); zl = [str(x) for x in zz["zones"]]; q15 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(z)][:152] for z in NB]).astype(float)
                    rl2 = np.vstack([q15.reshape(38, 4, 9).mean(1), np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(z)][152:153] for z in NB]).astype(float)])
                    rl1 = np.column_stack([D1["rl_d1"][j]] + [zz["d1_z"][zl.index(z)] for z in NB]).astype(float).reshape(24, 4, 9).mean(1)
                g_, e_ = float(I["gas"][j]), EUA[h0 - 48]
                d2 = rows2(slice(h0, h0 + 39), rl=np.nan_to_num(rl2), drl=np.zeros(39), state_at=h0); d2["gas"] = np.full(39, g_); d2["eua"] = np.full(39, e_); p2 = pred2(A, d2)
                d1 = rows2(slice(h0 - 24, h0), rl=np.nan_to_num(rl1), drl=np.zeros(24), state_at=h0 - 24); d1["gas"] = np.full(24, g_); d1["eua"] = np.full(24, e_); L = 1.85 * g_ + 0.37 * e_
                rel = np.nanmean((P[h0 - 24:h0] - pred2(A, d1)) / L); corr = float(A["rho"]) * rel * L * np.where(np.arange(39) < 24, 1.0, float(A["rho"]))
                res[arm][j] = np.repeat(p2 / 10.0, 4)[:153]; res[arm + "_oc"][j] = np.repeat((p2 + corr) / 10.0, 4)[:153]
                # hour-specific correction from the 72 known hours (D-1 and D with published loads and their own gas price, D+1 as above)
                dp = rows2(slice(h0 - 72, h0 - 24), drl=np.zeros(48)); dp["eua"] = np.nan_to_num(dp["eua"], nan=e_); dp["gas"] = np.nan_to_num(dp["gas"], nan=g_); dp = {k: np.nan_to_num(v) for k, v in dp.items()}
                r72 = np.concatenate([(P[h0 - 72:h0 - 24] - pred2(A, dp)) / (1.85 * dp["gas"] + 0.37 * dp["eua"]), (P[h0 - 24:h0] - pred2(A, d1)) / L]); age = np.repeat([2.0, 1.0, 0.0], 24); hk = HOUR[h0 - 72:h0]
                hf = HOUR[h0:h0 + 39]; dist_ = np.abs(hf[:, None] - hk[None]); dist_ = np.minimum(dist_, 24 - dist_); w_ = np.exp(-age[None] / MEM) * np.exp(-0.5 * (dist_ / HW) ** 2) * np.isfinite(r72)[None]
                c2 = (w_ * np.nan_to_num(r72)[None]).sum(1) / np.maximum(w_.sum(1), 1e-9) * L; res[arm + "_oc2"][j] = np.repeat((p2 + c2) / 10.0, 4)[:153]
                # the same correction in index space: per known hour, the shift of the load index [GW] at which the curve gives the observed price (bisection, +-15 GW),
                # averaged with the same weights and passed through the curve for the forecast hours
                x72 = np.concatenate([index2(A, dp, np), index2(A, d1, np)]); g72 = np.concatenate([dp["gas"], d1["gas"]]); e72 = np.concatenate([dp["eua"], d1["eua"]]); a72 = P[h0 - 72:h0]; lo_ = np.full(72, -15.0); hi_ = np.full(72, 15.0)
                for _ in range(22):
                    mid_ = 0.5 * (lo_ + hi_); f_ = pcurve(A, x72 + mid_, g72, e72, np)[0] - a72; lo_ = np.where(f_ < 0, mid_, lo_); hi_ = np.where(f_ < 0, hi_, mid_)
                dx = 0.5 * (lo_ + hi_); okx = np.isfinite(a72); wx_ = np.exp(-age[None] / MEM) * np.exp(-0.5 * (dist_ / HW) ** 2) * okx[None]; sh = (wx_ * np.nan_to_num(dx)[None]).sum(1) / np.maximum(wx_.sum(1), 1e-9)
                res[arm + "_oc3"][j] = np.repeat(pcurve(A, index2(A, d2, np) + sh, d2["gas"], d2["eua"], np)[0] / 10.0, 4)[:153]
        a_ = I["actual"].astype(float); print(f"{period}: persistence factors used {np.round(rhos, 2)}")
        for arm, v in res.items():
            np.savez_compressed(f"{SP}/out/{FCTAG}_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); e = np.abs(v - a_); print(f"{period} {arm:9s}: MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a_ - v)[:, :32]):+.2f}, first 14 h {np.nanmean((a_ - v)[:, :56]):+.2f}   ({time.time() - t0:.0f}s)")
