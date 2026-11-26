"""Candidate 4th member 'SD': yesterday's price curve + a residual-load-delta correction, fitted walk-forward.
p_D(s) = p_ref(s) + a_h[daypair] + b_h * (RL_D(s) - RL_ref(s)) [+ c_h * (RL_D - RL_ref)^2 ...], per local hour h.
Training rows: every past day t <= D-1 (from 2025-06-01), RL = stored TSO day-ahead figures (gate-known for t <= D-1).
Prediction: RL_D = the champion's rl_mixed (what Timestral/market read); RL_ref = TSO figures of the reference day.
Reference: D-1 (prices of D-1 are public before the gate), and D-7 as a second variant."""
import gzip, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import HuberRegressor
from panel import champion, day_mae, SP
def store(name, key):
    z = json.load(gzip.open(f"{SP}/faithful/stores_fresh/{name}.json.gz"))
    return pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in z[key].items()}).sort_index()
pr = store("prices_v3_DE_LU", "price")   # ct/kWh
G = {k: store("generation_v1_DE_LU", k) for k in ("tso_load", "tso_wind", "tso_solar")}
rl = (G["tso_load"] - G["tso_wind"] - G["tso_solar"])            # GW

TZ = "Europe/Berlin"
def grid(series, day):
    t = pd.date_range(pd.Timestamp(day, tz=TZ), pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1), freq="15min", inclusive="left").tz_convert("UTC")
    return series.reindex(t).to_numpy(float), t

def daytype(d):  # 0 Mon, 1 Tue-Fri, 2 Sat, 3 Sun
    w = pd.Timestamp(d).weekday(); return 0 if w == 0 else (1 if w < 5 else w - 3)

def build(first="2025-06-01", last="2026-11-26"):
    days = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d"); Pd, Rd, H = {}, {}, {}
    for d in days:
        p, t = grid(pr, d); r, _ = grid(rl, d)
        if len(p) != 96: continue      # skip clock-change days in training
        Pd[d], Rd[d] = p, r; H[d] = t.tz_convert(TZ).hour.to_numpy()
    return Pd, Rd, H

def fit_predict(Pd, Rd, H, D, rlD, lag=1, train_days=365, quad=False):
    """rows: day t in [D-train_days, D-1] with ref t-lag; per hour Huber on [pair dummies, dRL, (dRL^2)]"""
    Dt = pd.Timestamp(D); ref = (Dt - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
    if ref not in Pd or len(rlD) != 96: return None
    X, y, h = [], [], []
    for k in range(1, train_days + 1):
        t = (Dt - pd.Timedelta(days=k)).strftime("%Y-%m-%d"); r = (Dt - pd.Timedelta(days=k + lag)).strftime("%Y-%m-%d")
        if t not in Pd or r not in Pd: continue
        dr = Rd[t] - Rd[r]; dp = Pd[t] - Pd[r]; pair = np.zeros((96, 4)); pair[:, daytype(t)] = 1  # intercept by target daytype
        pair2 = np.zeros((96, 4)); pair2[:, daytype(r)] = 1
        f = [pair, pair2[:, 1:], dr[:, None]] + ([dr[:, None] ** 2 / 10] if quad else [])
        X.append(np.hstack(f)); y.append(dp); h.append(H[t])
    X, y, h = np.vstack(X), np.concatenate(y), np.concatenate(h); ok = np.isfinite(X).all(1) & np.isfinite(y)
    pair = np.zeros((96, 4)); pair[:, daytype(D)] = 1; pair2 = np.zeros((96, 4)); pair2[:, daytype(ref)] = 1
    dr = rlD - Rd[ref]; XD = np.hstack([pair, pair2[:, 1:], dr[:, None]] + ([dr[:, None] ** 2 / 10] if quad else []))
    hD = pd.date_range(pd.Timestamp(D, tz=TZ), periods=96, freq="15min").hour.to_numpy(); out = np.full(96, np.nan)
    for hh in range(24):
        m = ok & (h == hh); md = hD == hh
        reg = HuberRegressor(alpha=1e-3, fit_intercept=False, max_iter=300).fit(X[m], y[m])
        out[md] = Pd[ref][md] + XD[md] @ reg.coef_
    return out

if __name__ == "__main__":
    import sys
    from panel import pad
    P = champion(); Pd, Rd, H = build(); a = P["actual"]; X = P["X"]
    res = {}
    for name, lag, quad in (("sd1", 1, False), ("sd7", 7, False), ("sd1q", 1, True)):
        F = np.full_like(a, np.nan)
        for i, D in enumerate(P["days"]):
            rlD = P["rl_mixed"][i][np.isfinite(P["rl_mixed"][i])]
            o = fit_predict(Pd, Rd, H, D, rlD, lag=lag, quad=quad)
            if o is not None: F[i, :96] = o
        res[name] = F
        np.save(f"{SP}/C/{name}.npy", F)
        ok = np.isfinite(F).all(1)
        S61 = np.arange(61)[ok[:61]]; S18 = np.arange(62, 80)[ok[62:]]
        dm = day_mae(F, a); eq = X.mean(-1)
        e4 = F - a; E = X - a[..., None]; mm = np.isfinite(e4)
        cor = [np.corrcoef(e4[mm], E[..., k][mm])[0, 1] for k in range(3)]
        print(f"{name}: MAE 61d {np.nanmean(dm[S61]):.3f} ({len(S61)} d) 18d {np.nanmean(dm[S18]):.3f}; err corr with trees/ts/market {np.round(cor,2)}")
        for wt in (0.25, 0.2):
            B = eq.copy(); B[ok] = (1 - wt) * eq[ok] + wt * F[ok]
            print(f"   blend thirds*(1-{wt}) + {wt}*{name}: 61d {np.nanmean(day_mae(B,a)[S61]):.3f} vs {np.nanmean(day_mae(eq,a)[S61]):.3f};  18d {np.nanmean(day_mae(B,a)[S18]):.3f} vs {np.nanmean(day_mae(eq,a)[S18]):.3f}")
    print("naive D-7 MAE:", end=" ")
    N = np.full_like(a, np.nan)
    for i, D in enumerate(P["days"]):
        r = (pd.Timestamp(D) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        if r in Pd and np.isfinite(a[i]).sum() == 96: N[i, :96] = Pd[r]
    print(np.nanmean(day_mae(N, a)[:61]).round(3), np.nanmean(day_mae(N, a)[62:]).round(3))
