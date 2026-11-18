"""M1 with the lignite block replaced by committed capacity x utilisation (notes 21.44). KMODE=oracle: the state of the target day is given;
KMODE=hazard: the state is known up to LAGK days before the target and carried forward with the fitted switching rates and the model's own price.
Layer in GW as m1_layer.py, midnight load. usage: [KMODE=oracle|hazard|carry LAGK=2] m2_layer.py RUN"""
import sys, os, subprocess, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
from scipy.stats import norm
from scipy.optimize import least_squares
SP = M.SP; RUN = sys.argv[1]; KMODE = os.environ.get("KMODE", "oracle"); LAGK = int(os.environ.get("LAGK", 2)); NH = 200; th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}
Z = np.load(f"{SP}/out/commit_lignite.npz"); Kd = pd.Series(Z["K"], index=pd.to_datetime(Z["days"])).asfreq("D").interpolate(); b_on, b_off, capu = Z["b_on"], Z["b_off"], Z["cap"]; CAPL = float(capu.sum())
R0 = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); R0 = R0[R0.tr]; R0["K"] = Kd.reindex(R0.index.normalize().tz_localize(None)).values; R0 = R0.dropna(subset=["K"]); uf = lambda q, x: q[0] + (q[1] - q[0]) * norm.cdf((x - q[2]) / q[3]); QU = least_squares(lambda q: uf(q, R0.rc_l.values) - R0.lig.values / R0.K.values, [0.46, 0.93, 0.95, 0.15], loss="soft_l1").x; print("utilisation of committed lignite: from", np.round(QU[0], 2), "to", np.round(QU[1], 2), "centred at price / cost", np.round(QU[2], 2), "width", np.round(QU[3], 2))
def supply(p, L, co2, K): cl = th["a_l"] + co2; f = M.fuels(th, p, L, co2); gX, gS = M.other_curves(th, p, L); return K * uf(QU, p / cl) + f["c"] + f["g"] + gX + gS + th["B_g"]
def clear(D, L, co2, K, eps=0.0):
    ex = lambda p, wd: supply(p, L, co2, K) + eps - wd - D; up = ex(np.full(np.shape(D), M.EPS), 0.0) < 0; dn = ex(np.full(np.shape(D), -M.EPS), th["K_s"]) > 0; lo = np.where(up, M.EPS, -3000.0); hi = np.where(up, 20000.0, -M.EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(55): mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
sig = lambda z: 1 / (1 + np.exp(-z))
def khaz(K0, rc, mn, we, steps):   # expected committed capacity after `steps` two-day steps from a fleet with K0 GW on (fleet-level use of the unit hazards)
    x = np.array([1.0, np.clip(rc, 0, 2.5), np.clip(mn, -1, 2), we]); po, pf = sig(x @ b_on), sig(x @ b_off); K = K0
    for _ in range(steps): K = K * (1 - pf) + (CAPL - K) * po
    return K
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["rl_de"]]; S.index = S.index.tz_convert("UTC"); R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; R["co2"] = np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1); R["L"] = 1.85 * R.gas + 0.37 * R.co2
full = pd.date_range(R.index.min().floor("D"), R.index.max().ceil("D"), freq="1h", tz="UTC"); G = R.reindex(full); co2f = G.co2.ffill(); G["K"] = Kd.reindex(G.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)).values; os.makedirs(f"{SP}/out/struct_{RUN}", exist_ok=True); CM = {"winter": "cmw", "summer": "cms", "autumn": "cma"}
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); GEN = np.load(f"{SP}/out/gen_mid_{per}.npz"); a = I["actual"].astype(float); meta = []; m1 = np.load(f"{SP}/out/structfc_m1m_{per}_price.npz")["vint_ts"].astype(float)
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); tag = str(c)[:10]; cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; tt = t0 + pd.Timedelta(minutes=15) * np.arange(153); hh = pd.date_range(t0 - pd.Timedelta(hours=24 * NH), t0 - pd.Timedelta(hours=1), freq="1h"); g = G.reindex(hh); P = g.p.values.reshape(NH, 24); oth = np.nanmean(g.other.values.reshape(NH, 24)[-9:-2], 0); X = (g.rl.values - g.other.values).reshape(NH, 24).copy(); X[-2] = g.rl.values.reshape(NH, 24)[-2] - oth; X[-1] = g.rl.values.reshape(NH, 24)[-1] - oth
        Lh = g.L.values.reshape(NH, 24).copy(); Ch = g.co2.values.reshape(NH, 24).copy(); Lh[-1] = Lj; Ch[-1] = cj; Kh = g.K.values.reshape(NH, 24).copy(); tday = (t0.tz_convert("Europe/Berlin") + pd.Timedelta(hours=1)).normalize().tz_localize(None); Ktrue = float(Kd.get(tday, np.nan)); Klast = float(Kd.get(tday - pd.Timedelta(days=LAGK), np.nan))
        if KMODE != "oracle":
            Kh[-LAGK + 1:] = Klast if LAGK > 1 else Kh[-1:]   # history rows whose state is not yet known at issue carry the last known state
            if KMODE == "carry": Kt = Klast
            else: pf_ = m1[j, :96] * 10; clj = th["a_l"] + cj; Kt = khaz(Klast, float(np.mean(pf_) / clj), float(np.min(pf_) / clj), float(tday.dayofweek >= 5), max(1, LAGK // 2))
        else: Kt = Ktrue
        ok = ~np.isnan(P + X + Lh + Ch + Kh); wd = th["K_s"] * (P < -M.EPS) + 0.5 * th["K_s"] * (np.abs(P) <= M.EPS); u = np.full((NH, 24), np.nan); u[ok] = (X - (supply(np.where(ok, P, 50.0), np.where(ok, Lh, 90.0), np.where(ok, Ch, 70.0), np.where(ok, Kh, 9.0)) - wd))[ok]
        hod = (np.arange(153) // 4) % 24; xk = S.rl_de.reindex(tt).ffill().bfill().values - oth[hod]; xt = GEN["rl_pes"][j].astype(float) - oth[hod]; meta.append((Lj, cj, xk, xt, Kt, Ktrue)); np.savez_compressed(f"{SP}/out/struct_{RUN}/{tag}.npz", R1=(10 * u).astype(np.float32), XH1=X.astype(np.float32), x_known=xk.astype(np.float32), known_shape=np.zeros(153, np.float32), x_ts=xt.astype(np.float32), ts_shape=np.zeros(153, np.float32))
    subprocess.run([sys.executable, "dyn2.py", per, RUN, "known,ts", "x+xh"], capture_output=True); U = np.load(f"{SP}/out/structfc_{RUN}_ts_dyn2xxh.npz")["vint"].astype(float); out = np.stack([clear(m_[3], np.full(153, m_[0]), np.full(153, m_[1]), m_[4], U[j]) / 10 for j, m_ in enumerate(meta)]); bare = np.stack([clear(m_[2], np.full(153, m_[0]), np.full(153, m_[1]), m_[4]) / 10 for m_ in meta]); np.savez_compressed(f"{SP}/out/structfc_{RUN}_{per}_price.npz", vint_ts=out.astype(np.float32), bare=bare.astype(np.float32), Kt=np.array([m_[4] for m_ in meta]), Ktrue=np.array([m_[5] for m_ in meta]))
    good = ~np.isnan([m_[4] for m_ in meta]); e = lambda v: np.abs(v - a)[good][:, :24].mean(); cc = np.load(f"{SP}/out/structfc_{CM[per]}_ts_dyn2xxh.npz")["vint"].astype(float); b1 = np.load(f"{SP}/out/structfc_m1m_{per}_price.npz")["bare"].astype(float); kerr = np.nanmean(np.abs(np.array([m_[4] - m_[5] for m_ in meta])))
    print(f"{per} [{KMODE}]: KEY METRIC ct ({int(good.sum())} nights) | M1 + layer {e(m1):.2f} -> lignite as state x utilisation {e(out):.2f} | candidate's curve {e(cc):.2f} | bare, published load: M1 {e(b1):.2f} -> {e(bare):.2f} | error of the state used: {kerr:.2f} GW")
