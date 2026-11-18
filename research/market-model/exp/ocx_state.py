"""Level of the index model as a latent state with missing data (notes section 15; specification fixed there before scoring).
Evidence per day = median over the day's hours of the implied index shift (the shift [GW] at which the fitted curve gives the observed price).
State: z_d = m_d + u_d + noise, m random walk (slow level), u AR(1) (reverts to zero). Days without evidence are predicted through.
usage: ocx_state.py checks FITDATE | ev | learn | sim | fig | fc PERIOD"""
import sys, os, time, numpy as np, pandas as pd
import oc, ocx2
from oc import *
from ocx2 import rows2, index2, pred2, pcurve, hx
from ocx_fits import fit_dates, get_fit, V, SFX
if V == "t":
    import ocxt; rows2 = ocxt.rows_t; index2 = ocxt.index_t; pred2 = ocxt.pred_t
if V in ("p1", "p2", "p3"):
    import ocxp; index2 = ocxp.index_p; pred2 = ocxp.pred_p
from scipy.optimize import minimize
WARM, OOS = 90, 7; FRZ = ("_frz" + os.environ["FREEZE"][5:7] + os.environ["FREEZE"][8:10]) if os.environ.get("FREEZE") else ""
PSTART = {"winter": "2025-11-30", "summer": "2026-03-30", "autumn": "2026-09-01"}       # day D of each period's first origin
def implied(A, d):
    """hourly implied index shift for the rows in d. Returns (shift with NaN where unusable, code): 0 usable, 1 price or input missing, 2 at the lower bound, 3 at the upper bound"""
    a = d["p"]; bad = ~np.isfinite(a) | ~np.isfinite(d["rl"]).all(1) | ~np.isfinite(d["sol"]) | ~np.isfinite(d["gasy"]) | ~np.isfinite(d["nuc"]) | ~np.isfinite(d["gas"]) | ~np.isfinite(d["eua"])
    dd = {k: np.nan_to_num(v) for k, v in d.items()}; dd["okx"] = ~(bad & np.isfinite(a)) if np.isfinite(a).any() else ~bad; x = index2(A, dd, np); n = len(a); lo = np.full(n, -15.0); hi = np.full(n, 15.0); a0 = np.nan_to_num(a)
    for _ in range(24):
        mid = 0.5 * (lo + hi); f = pcurve(A, x + mid, dd["gas"], dd["eua"], np)[0] - a0; lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    dx = 0.5 * (lo + hi); code = np.where(bad, 1, np.where(dx < -14.8, 2, np.where(dx > 14.8, 3, 0))); return np.where(code == 0, dx, np.nan), code
def daily(dx, days, hours=None, sel=None, minn=12):
    """median per day id over the selected hours; days with fewer than minn usable hours are missing. Returns a Series over consecutive day ids."""
    s = pd.Series(dx if sel is None else np.where(sel(hours), dx, np.nan)); g = s.groupby(np.asarray(days)); z = g.median().where(g.count() >= minn)
    return z.reindex(range(int(np.min(days)), int(np.max(days)) + 1))
def kf2(z, th, radd=None, cap=None):
    """two-state filter (slow level m, fast part u) over a daily series with NaN for missing days.
    Returns per day: one-day-ahead predicted state mean and variance (before seeing the day), predictive variance of the evidence, log predictive density (0 if missing); and the final state."""
    phi, qu, qm, r = th; m = np.zeros(2); Pm = np.diag([25.0, qu / max(1 - phi * phi, 1e-3)]); F = np.array([1.0, phi]); Q = np.diag([qm, qu]); n = len(z)
    zp, S0, S, ll, comp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 2))
    for i in range(n):
        m = F * m; Pm = F[:, None] * Pm * F[None] + Q; zp[i] = m.sum(); S0[i] = Pm.sum(); S[i] = S0[i] + r + (0.0 if radd is None else radd[i]); comp[i] = m
        if np.isfinite(z[i]):
            v = z[i] - zp[i]; ll[i] = -0.5 * (np.log(2 * np.pi * S[i]) + v * v / S[i])
            if cap is not None: v = float(np.clip(v, -cap * np.sqrt(S[i]), cap * np.sqrt(S[i])))
            K = Pm.sum(1) / S[i]; m = m + K * v; Pm = Pm - np.outer(K, Pm.sum(0))
    return dict(zp=zp, S0=S0, S=S, ll=ll, comp=comp, m=m, P=Pm, F=F, Q=Q)
def ahead(k, th, steps):
    """state mean (m + u) and variance `steps` days after the last filtered day"""
    m, Pm, out = k["m"].copy(), k["P"].copy(), []
    for _ in range(steps): m = k["F"] * m; Pm = k["F"][:, None] * Pm * k["F"][None] + k["Q"]; out.append((m.sum(), Pm.sum()))
    return out
def unpack(x): return 1 / (1 + np.exp(-x[0])), np.exp(x[1]), np.exp(x[2]), np.exp(x[3])
def nlp(x, segs, cap=None):
    """minus log posterior of the hyperparameters: out-of-sample one-day-ahead likelihood over the segments + weak priors (none truncated)"""
    th = unpack(x); ll = sum(kf2(z, th, cap=cap)["ll"][w:].sum() for z, w in segs)
    lp = 3 * np.log(th[0]) + 1 * np.log(1 - th[0]) - 0.5 * ((x[1] - np.log(2.0)) / 2.0) ** 2 - 0.5 * ((x[2] - np.log(0.05)) / 2.5) ** 2 - 0.5 * ((x[3] - np.log(4.0)) / 2.0) ** 2
    return -(ll + lp)
def learn(segs, cap=None):
    best = None
    for x0 in ([1.4, 0.7, -3.0, 1.4], [0.0, 1.0, -1.0, 0.5], [2.5, -0.5, -2.0, 1.5], [0.5, 0.0, 0.0, 1.0]):
        o = minimize(nlp, np.array(x0), args=(segs, cap), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=3000))
        if best is None or o.fun < best.fun: best = o
    return unpack(best.x), best.fun
def evidence_for_fit(date):
    """evidence under the weekly fit dated `date`: 90 days before the fit date (in sample) and the 7 days after it (out of sample); loads as published, no realised-load term"""
    A = get_fit(date); h = hx(date); d0 = int(oc.DAY[h]); sl = slice(h - (WARM + 1) * 24, min(h + (OOS + 1) * 24, oc.NH)); n = sl.stop - sl.start; d = rows2(sl, drl=np.zeros(n)); dx, code = implied(A, d); days, hours = oc.DAY[sl], oc.HOUR[sl]
    rng = range(d0 - WARM, d0 + OOS); z = daily(dx, days).reindex(rng).to_numpy(); zn = daily(dx, days, hours, lambda h_: h_ < 8, 4).reindex(rng).to_numpy(); dx1, _ = implied(A, rows2(sl)); z1 = daily(dx1, days).reindex(rng).to_numpy()
    kd = float(A["k3"]) * pd.Series(np.asarray(oc.DRL[sl], float)).groupby(np.asarray(days)).median().reindex(rng).to_numpy(); return z, zn, z1, kd, code, hours
def load_ev():
    E = np.load(f"{SP}/out/ocxw{SFX}_ev.npz", allow_pickle=True); return [str(x) for x in E["dates"]], E["Z"], E["ZN"], E["Z1"], E["KD"]
def segs_before(dates, Z, day0):
    """segments (series, warm-up length) of the fits whose 7 out-of-sample days end by day0"""
    return [(Z[i], WARM) for i, t in enumerate(dates) if pd.Timestamp(t) + pd.Timedelta(days=OOS) <= pd.Timestamp(day0)]
def hist_evidence(A, h0, g_, e_):
    """daily evidence under fit A for the 90 days to D-1 and for D (h0 = first hour of D+2): published loads, no realised-load term"""
    d0 = int(oc.DAY[h0 - 48]); sl = slice(h0 - 24 - (WARM + 2) * 24, h0 - 24); dh = rows2(sl, drl=np.zeros(sl.stop - sl.start)); dh["gas"] = np.where(np.isfinite(dh["gas"]), dh["gas"], g_); dh["eua"] = np.where(np.isfinite(dh["eua"]), dh["eua"], e_)
    dxh, _ = implied(A, dh); return daily(dxh, oc.DAY[sl]).reindex(range(d0 - WARM, d0 + 1)).to_numpy()
def later_evidence(A, h0, g_, e_):
    """what D+2 turned out to be (for checking only): median implied shift over the day and over its hours 0-7, published loads"""
    d = rows2(slice(h0, h0 + 24), drl=np.zeros(24), state_at=h0); d["gas"] = np.full(24, g_); d["eua"] = np.full(24, e_); dx, _ = implied(A, d)
    return (float(np.nanmedian(dx)) if np.isfinite(dx).sum() >= 12 else np.nan), (float(np.nanmedian(dx[:8])) if np.isfinite(dx[:8]).sum() >= 4 else np.nan)
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "checks":
        A = get_fit(sys.argv[2]); h = hx(sys.argv[2]); sl = slice(h - 540 * 24, h); n = sl.stop - sl.start; d = rows2(sl, drl=np.zeros(n)); dx, code = implied(A, d); hours = oc.HOUR[sl]
        print(f"fit {sys.argv[2]}, the 540 days before it: hours usable {100 * np.mean(code == 0):.1f}%, price or input missing {100 * np.mean(code == 1):.1f}%, at the lower bound {100 * np.mean(code == 2):.1f}%, at the upper bound {100 * np.mean(code == 3):.1f}%")
        print("  share at the lower bound by hour of day, %: " + " ".join(f"{100 * np.mean(code[hours == k] == 2):.0f}" for k in range(24)))
        p = d["p"]; print(f"  price in the hours at the lower bound: median {np.nanmedian(p[code == 2]):.1f}, 90th percentile {np.nanpercentile(p[code == 2], 90):.1f} EUR/MWh; share of them with price <= 5: {100 * np.mean(p[code == 2] <= 5):.0f}%; at the upper bound: median price {np.nanmedian(p[code == 3]) if (code == 3).any() else np.nan:.0f}")
        xs = np.linspace(-30, 90, 481); worst = 0.0
        for g in np.nanpercentile(d["gas"], [1, 25, 50, 75, 99]):
            for e in np.nanpercentile(d["eua"], [1, 50, 99]):
                dp = np.diff(pcurve(A, xs, np.full(len(xs), g), np.full(len(xs), e), np)[0]); worst = min(worst, dp.min())
        print(f"  monotone in the index over gas {np.nanpercentile(d['gas'], 1):.0f}..{np.nanpercentile(d['gas'], 99):.0f} and carbon {np.nanpercentile(d['eua'], 1):.0f}..{np.nanpercentile(d['eua'], 99):.0f}: smallest step of the curve over 0.25 GW = {worst:+.4f} EUR/MWh ({'monotone' if worst >= -1e-9 else 'NOT monotone'})")
        days = oc.DAY[sl]; z = daily(dx, days); kd = float(A["k3"]) * pd.Series(np.asarray(oc.DRL[sl], float)).groupby(np.asarray(days)).median()
        print(f"  realised-load term: k3 {float(A['k3']):.2f}; variance of k3 x daily-median (realised - published) load {np.nanvar(kd):.2f} GW^2, against the variance of the daily evidence {np.nanvar(z):.1f} GW^2; its autocorrelation at 1, 2, 3 days: " + " ".join(f"{kd.autocorr(k):.2f}" for k in (1, 2, 3)))
    if mode == "ev":
        dates = [t for t in fit_dates() if os.path.exists(f"{SP}/out/ocxw{SFX}/{t}.npz")]; Z, ZN, Z1, KD = [], [], [], []
        for t in dates: z, zn, z1, kd, _, _ = evidence_for_fit(t); Z.append(z); ZN.append(zn); Z1.append(z1); KD.append(kd)
        np.savez(f"{SP}/out/ocxw{SFX}_ev.npz", dates=np.array(dates), Z=np.array(Z), ZN=np.array(ZN), Z1=np.array(Z1), KD=np.array(KD)); Z = np.array(Z)
        print(f"{len(dates)} weekly fits {dates[0]} .. {dates[-1]}; evidence days per fit: in sample {np.isfinite(Z[:, :WARM]).sum(1).mean():.1f} of {WARM}, out of sample {np.isfinite(Z[:, WARM:]).sum(1).mean():.1f} of {OOS}")
    if mode == "learn":
        dates, Z, ZN, Z1, KD = load_ev(); CAP = float(os.environ["CAP"]) if os.environ.get("CAP") else None; out = {}
        for per, day0 in PSTART.items():
            segs = segs_before(dates, Z, day0); idx = [i for i, t in enumerate(dates) if pd.Timestamp(t) + pd.Timedelta(days=OOS) <= pd.Timestamp(day0)]; th, f = learn(segs, CAP); out[per] = th
            ks = [kf2(z, th, cap=CAP) for z, _ in segs]; inn = np.concatenate([(z[WARM:] - k["zp"][WARM:]) / np.sqrt(k["S"][WARM:]) for (z, _), k in zip(segs, ks)]); zo = np.concatenate([z[WARM:] for z, _ in segs]); zp = np.concatenate([k["zp"][WARM:] for k in ks]); zl = np.concatenate([z[WARM - 1:-1] for z, _ in segs])
            st = kf2(np.full(400, 0.0), th); gain = st["S0"][-1] / st["S"][-1]; ok = np.isfinite(inn); s_ = pd.Series(inn)
            print(f"\n{per}: learned on the {len(segs)} weekly fits whose out-of-sample week ends before {day0} ({ok.sum()} out-of-sample days)")
            print(f"  fast part: persistence {th[0]:.2f}, innovation variance {th[1]:.2f} (stationary sd {np.sqrt(th[1] / (1 - th[0] ** 2)):.2f} GW); slow level: innovation variance {th[2]:.3f} per day (sd of its change over 30 days {np.sqrt(30 * th[2]):.2f} GW); evidence noise variance {th[3]:.2f}; weight on a new day's evidence {gain:.2f}")
            print(f"  standardised one-day-ahead innovations, out of sample: sd {np.nanstd(inn):.2f}, excess kurtosis {s_.kurt():.2f}, beyond 2.5 sd {100 * np.nanmean(np.abs(inn[ok]) > 2.5):.1f}% (normal 1.2%), beyond 3.5 sd {100 * np.nanmean(np.abs(inn[ok]) > 3.5):.1f}%; autocorrelation at 1-7 days: " + " ".join(f"{s_.autocorr(k):.2f}" for k in range(1, 8)))
            m_ = np.isfinite(zo) & np.isfinite(zl); rho = np.sum(zo[m_] * zl[m_]) / np.sum(zl[m_] ** 2)
            print(f"  one-day-ahead error of the daily evidence, out of sample, mean absolute GW: no layer {np.nanmean(np.abs(zo[m_])):.2f} | yesterday's value x {rho:.2f} (fitted here, in hindsight) {np.nanmean(np.abs(zo[m_] - rho * zl[m_])):.2f} | filter {np.nanmean(np.abs(zo[m_] - zp[m_])):.2f}; mean of the evidence {np.nanmean(zo):+.2f}, of the filter's error {np.nanmean(zo[m_] - zp[m_]):+.2f}")
            for lab, th2 in (("fast part only (no slow level)", (th[0], th[1], 1e-8, th[3])), ("slow level only (random walk + noise)", (0.0, 1e-8, max(th[2], 0.3), th[3] + th[1]))):
                k2 = [kf2(z, th2) for z, _ in segs]; zp2 = np.concatenate([k["zp"][WARM:] for k in k2]); print(f"    for comparison, {lab}: {np.nanmean(np.abs(zo[m_] - zp2[m_])):.2f}")
            zn = np.concatenate([ZN[i][WARM:] for i in idx]); m2 = np.isfinite(zn) & np.isfinite(zp) & np.isfinite(zl); b = np.polyfit(zp[m2], zn[m2], 1)
            print(f"  night block (hours 0-7) of the same day against the filter's prediction: correlation {np.corrcoef(zp[m2], zn[m2])[0, 1]:.2f}, slope {b[0]:.2f}, intercept {b[1]:+.2f} GW; night evidence mean absolute: no layer {np.mean(np.abs(zn[m2])):.2f}, with the filter's prediction {np.mean(np.abs(zn[m2] - zp[m2])):.2f}, with yesterday's all-hours value x {rho:.2f}: {np.mean(np.abs(zn[m2] - rho * zl[m2])):.2f}")
        np.savez(f"{SP}/out/ocxw{SFX}_th{'' if CAP is None else '_cap'}.npz", **{k: np.array(v) for k, v in out.items()})
    if mode == "fc":
        period = sys.argv[2]; I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
        _thf = f"{SP}/out/ocxw{SFX}_th.npz"; TH = tuple(np.load(_thf if os.path.exists(_thf) else f"{SP}/out/ocxw_th.npz")[period]); dates = fit_dates(); fh = [hx(t) for t in dates]; V0, N0 = 2.0, 5.0; MEM, HW = 2.0, 1.0; t0 = time.time()
        res = {a + s: np.full((n, 153), np.nan) for a in ("known", "ts") for s in ("", "_st", "_m90", "_oc", "_oc2", "_oc3")}; diag = {k: np.full(n, np.nan) for k in ("s2_known", "s2_ts", "sd2_known", "sd2_ts", "z1_known", "z1_ts", "zD", "e_load", "v_load", "z2", "z2n")}; e_hist = []
        for j, t in enumerate(tags):
            h0 = int(I["cuts"][j]) // 4; A = get_fit(min(dates[max(i for i in range(len(dates)) if fh[i] <= h0 - 48)], os.environ.get("FREEZE", "9999"))); g_, e_ = float(I["gas"][j]), EUA[h0 - 48]; L = 1.85 * g_ + 0.37 * e_; d0 = int(oc.DAY[h0 - 48])
            # evidence under this fit for the 90 days to D-1 and for D: published loads, no realised-load term
            zh = hist_evidence(A, h0, g_, e_); diag["zD"][j] = zh[-1]; v_load = (N0 * V0 + float(np.sum(np.square(e_hist)))) / (N0 + len(e_hist)); diag["v_load"][j] = v_load
            dpub = rows2(slice(h0 - 24, h0), drl=np.zeros(24), state_at=h0 - 24); dpub["gas"] = np.full(24, g_); dpub["eua"] = np.full(24, e_); xpub = index2(A, {k: np.nan_to_num(v) for k, v in dpub.items()}, np); e_now = None
            for arm in ("known", "ts"):
                rl2 = RL[h0:h0 + 39].copy(); rl1 = RL[h0 - 24:h0].copy()
                if arm == "ts":
                    zf = f"{NBD}/{t}.npz"
                    if not os.path.exists(zf): continue
                    zz = np.load(zf); zl = [str(x) for x in zz["zones"]]; q15 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(z)][:152] for z in NB]).astype(float)
                    rl2 = np.vstack([q15.reshape(38, 4, 9).mean(1), np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(z)][152:153] for z in NB]).astype(float)])
                    rl1 = np.column_stack([D1["rl_d1"][j]] + [zz["d1_z"][zl.index(z)] for z in NB]).astype(float).reshape(24, 4, 9).mean(1)
                d2 = rows2(slice(h0, h0 + 39), rl=np.nan_to_num(rl2), drl=np.zeros(39), state_at=h0); d2["gas"] = np.full(39, g_); d2["eua"] = np.full(39, e_); x2 = index2(A, d2, np); p2 = pcurve(A, x2, d2["gas"], d2["eua"], np)[0]
                d1 = rows2(slice(h0 - 24, h0), rl=np.nan_to_num(rl1), drl=np.zeros(24), state_at=h0 - 24); d1["gas"] = np.full(24, g_); d1["eua"] = np.full(24, e_); dx1, _ = implied(A, d1); z1 = float(np.nanmedian(dx1)) if np.isfinite(dx1).sum() >= 12 else np.nan
                if arm == "ts": e_now = float(np.nanmedian(index2(A, d1, np) - xpub))
                zz_ = np.r_[zh, z1]; radd = np.zeros(len(zz_)); radd[-1] = v_load if arm == "ts" else 0.0; k = kf2(zz_, TH, radd=radd); (s2, v2), (s3, v3) = ahead(k, TH, 2); sh = np.where(np.arange(39) < 24, s2, s3)
                diag["s2_" + arm][j] = s2; diag["sd2_" + arm][j] = np.sqrt(v2); diag["z1_" + arm][j] = z1
                res[arm + "_m90"][j] = np.repeat(pcurve(A, x2 + np.nanmean(zh), d2["gas"], d2["eua"], np)[0] / 10.0, 4)[:153]            # control: constant recalibration by the 90-day mean of the evidence, no dynamics
                res[arm][j] = np.repeat(p2 / 10.0, 4)[:153]; res[arm + "_st"][j] = np.repeat(pcurve(A, x2 + sh, d2["gas"], d2["eua"], np)[0] / 10.0, 4)[:153]
                # the three earlier layers on the same fit (copied from ocx2.py): persistence of the D+1 relative error; hour-specific in price; hour-specific in index space
                rel = np.nanmean((P[h0 - 24:h0] - pred2(A, d1)) / L); corr = float(A["rho"]) * rel * L * np.where(np.arange(39) < 24, 1.0, float(A["rho"])); res[arm + "_oc"][j] = np.repeat((p2 + corr) / 10.0, 4)[:153]
                dp = rows2(slice(h0 - 72, h0 - 24), drl=np.zeros(48)); dp["eua"] = np.nan_to_num(dp["eua"], nan=e_); dp["gas"] = np.nan_to_num(dp["gas"], nan=g_); dp = {k_: np.nan_to_num(v) for k_, v in dp.items()}
                r72 = np.concatenate([(P[h0 - 72:h0 - 24] - pred2(A, dp)) / (1.85 * dp["gas"] + 0.37 * dp["eua"]), (P[h0 - 24:h0] - pred2(A, d1)) / L]); age = np.repeat([2.0, 1.0, 0.0], 24); hk = HOUR[h0 - 72:h0]
                hf = HOUR[h0:h0 + 39]; dist_ = np.abs(hf[:, None] - hk[None]); dist_ = np.minimum(dist_, 24 - dist_); w_ = np.exp(-age[None] / MEM) * np.exp(-0.5 * (dist_ / HW) ** 2) * np.isfinite(r72)[None]
                c2 = (w_ * np.nan_to_num(r72)[None]).sum(1) / np.maximum(w_.sum(1), 1e-9) * L; res[arm + "_oc2"][j] = np.repeat((p2 + c2) / 10.0, 4)[:153]
                x72 = np.concatenate([index2(A, dp, np), index2(A, d1, np)]); g72 = np.concatenate([dp["gas"], d1["gas"]]); e72 = np.concatenate([dp["eua"], d1["eua"]]); a72 = P[h0 - 72:h0]; lo_ = np.full(72, -15.0); hi_ = np.full(72, 15.0)
                for _ in range(22):
                    mid_ = 0.5 * (lo_ + hi_); f_ = pcurve(A, x72 + mid_, g72, e72, np)[0] - a72; lo_ = np.where(f_ < 0, mid_, lo_); hi_ = np.where(f_ < 0, hi_, mid_)
                dx = 0.5 * (lo_ + hi_); okx = np.isfinite(a72); wx_ = np.exp(-age[None] / MEM) * np.exp(-0.5 * (dist_ / HW) ** 2) * okx[None]; sh3 = (wx_ * np.nan_to_num(dx)[None]).sum(1) / np.maximum(wx_.sum(1), 1e-9)
                res[arm + "_oc3"][j] = np.repeat(pcurve(A, x2 + sh3, d2["gas"], d2["eua"], np)[0] / 10.0, 4)[:153]
            if e_now is not None and np.isfinite(e_now): e_hist.append(e_now); diag["e_load"][j] = e_now
            diag["z2"][j], diag["z2n"][j] = later_evidence(A, h0, g_, e_)
        a_ = I["actual"].astype(float); np.savez(f"{SP}/out/ocxs{SFX}{FRZ}_{period}_diag.npz", cutoffs=I["cutoffs"], **diag)
        print(f"{period}: state dynamics used: persistence {TH[0]:.2f}, fast variance {TH[1]:.2f}, level variance {TH[2]:.3f}, noise {TH[3]:.2f}; load-forecast index error of D+1 (daily median): sd {np.nanstd(diag['e_load']):.2f} GW, mean {np.nanmean(diag['e_load']):+.2f}; extra noise variance used, first/last origin: {diag['v_load'][0]:.2f} / {diag['v_load'][-1]:.2f}")
        for arm, v in res.items():
            np.savez_compressed(f"{SP}/out/ocxs{SFX}{FRZ}_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); e = np.abs(v - a_); print(f"{period} {arm:10s}: MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a_ - v)[:, :32]):+.2f}, first 14 h {np.nanmean((a_ - v)[:, :56]):+.2f}   ({time.time() - t0:.0f}s)")
        r_ = TH[3]
        for arm in ("known", "ts"):
            sdp = np.sqrt(diag["sd2_" + arm] ** 2 + r_); zs = (diag["z2"] - diag["s2_" + arm]) / sdp; ok = np.isfinite(zs)
            print(f"{period} {arm}: realised evidence of D+2 against the predicted state: {ok.sum()} days, inside the 50% predictive interval {100 * np.mean(np.abs(zs[ok]) < 0.6745):.0f}%, inside the 90% interval {100 * np.mean(np.abs(zs[ok]) < 1.6449):.0f}%; mean error {np.nanmean(diag['z2'] - diag['s2_' + arm]):+.2f} GW; mean absolute: no layer {np.nanmean(np.abs(diag['z2'])):.2f}, with the state {np.nanmean(np.abs(diag['z2'] - diag['s2_' + arm])):.2f}; night block: no layer {np.nanmean(np.abs(diag['z2n'])):.2f}, with the state {np.nanmean(np.abs(diag['z2n'] - diag['s2_' + arm])):.2f}, mean night error {np.nanmean(diag['z2n'] - diag['s2_' + arm]):+.2f}")
    if mode == "sim":
        # simulated-data check: known dynamics, gaps inserted, the same segment structure and learner as on the real evidence (43 weekly segments of 90 + 7 days)
        TRUE = (0.70, 5.0, 0.03, 5.0); rng = np.random.default_rng(1); NS = 43; est = []; cov = {0.5: [], 0.9: []}; covg = {0.5: [], 0.9: []}
        for rep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 8):
            nd = NS * OOS + WARM; m = np.cumsum(rng.normal(0, np.sqrt(TRUE[2]), nd)) + rng.normal(0, 2.0); u = np.zeros(nd); u[0] = rng.normal(0, np.sqrt(TRUE[1] / (1 - TRUE[0] ** 2)))
            for i in range(1, nd): u[i] = TRUE[0] * u[i - 1] + rng.normal(0, np.sqrt(TRUE[1]))
            s = m + u; z = s + rng.normal(0, np.sqrt(TRUE[3]), nd); miss = rng.random(nd) < 0.04
            for g0 in rng.integers(WARM, nd - 6, 6): miss[g0:g0 + rng.integers(2, 6)] = True
            z = np.where(miss, np.nan, z); segs = [(z[OOS * k: OOS * k + WARM + OOS], WARM) for k in range(NS)]; th, _ = learn(segs); est.append(th)
            for k_ in range(NS):
                kk = kf2(segs[k_][0], th); sl_ = slice(OOS * k_ + WARM, OOS * k_ + WARM + OOS); zs = (s[sl_] - kk["zp"][WARM:]) / np.sqrt(kk["S0"][WARM:]); g_ = miss[sl_]
                for c, q_ in ((0.5, 0.6745), (0.9, 1.6449)): cov[c] += list(np.abs(zs) < q_); covg[c] += list(np.abs(zs[g_]) < q_)
        est = np.array(est); print(f"simulated recovery, {len(est)} replications, {NS} segments each, 4% of days missing at random plus six gaps of 2-5 days")
        for i, nm in enumerate(("persistence of the fast part", "fast innovation variance", "level innovation variance", "evidence noise variance")): print(f"  {nm:30s}: true {TRUE[i]:.2f}, recovered median {np.median(est[:, i]):.2f}, range {est[:, i].min():.2f}..{est[:, i].max():.2f}")
        print(f"  coverage of the one-day-ahead state interval, all out-of-sample days: 50% interval {100 * np.mean(cov[0.5]):.0f}%, 90% interval {100 * np.mean(cov[0.9]):.0f}%; on days whose own evidence is missing ({len(covg[0.5])} days): {100 * np.mean(covg[0.5]):.0f}%, {100 * np.mean(covg[0.9]):.0f}%")
