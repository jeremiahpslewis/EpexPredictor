"""Walk-forward stacking: residual (actual - equal thirds) regressed on gate-known context, refit each origin on days < d.
Features lagged >= 1 day where they use actuals; gas taken as the last THE value at least 1 day before the target day."""
import gzip, json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import QuantileRegressor, HuberRegressor, Ridge
from panel import champion, day_mae, SP
P = champion(); X, a, qh, wd = P["X"], P["actual"], P["qh"], P["wd"]; D = len(a); eq = X.mean(-1)
gz = {}
for st in ("stores", "stores_fresh"):
    gz.update(json.load(gzip.open(f"{SP}/faithful/{st}/gasprices_the_v1_DE_LU.json.gz"))["gasprice"])
gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in gz.items()}).sort_index()
days = pd.DatetimeIndex(P["days"]).tz_localize("Europe/Berlin")
g1 = np.array([gas[:d - pd.Timedelta(days=1)].iloc[-1] for d in days]); g8 = np.array([gas[:d - pd.Timedelta(days=8)].iloc[-1] for d in days])
de = np.nanmean(eq - a, 1); de_lag = np.r_[0, de[:-1]]
def feats(name):
    t = qh / 96 * 2 * np.pi; base = [np.ones_like(eq)]
    cols = {"hod": [np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)],
            "dev": [X[..., 0] - eq, X[..., 1] - eq, X[..., 2] - eq],
            "spread": [X.max(-1) - X.min(-1)],
            "level": [(eq - 10) / 5],
            "rl": [(P["rl_mixed"] - 40) / 10],
            "wkend": [(wd >= 5).astype(float)],
            "lagerr": [np.repeat(de_lag[:, None], 100, 1)],
            "gas": [np.repeat(((g1 - 40) / 10)[:, None], 100, 1), np.repeat(((g1 - g8) / 5)[:, None], 100, 1)]}
    for n in name.split("+"): base += cols[n]
    return np.stack(base, -1)
def run(fs, model, alpha, burn=14, sub=4):
    Z = feats(fs); y = a - eq; F = eq.copy()
    for d in range(burn, D):
        Zt = Z[:d, ::sub].reshape(-1, Z.shape[-1]); yt = y[:d, ::sub].ravel(); m = np.isfinite(yt) & np.isfinite(Zt).all(1)
        if model == "lad": r = QuantileRegressor(quantile=0.5, alpha=alpha, fit_intercept=False, solver="highs").fit(Zt[m], yt[m])
        elif model == "huber": r = HuberRegressor(alpha=alpha, fit_intercept=False, max_iter=500).fit(Zt[m], yt[m])
        else: r = Ridge(alpha=alpha, fit_intercept=False).fit(Zt[m], yt[m])
        zd = Z[d]; ok = np.isfinite(zd).all(1); F[d, ok] = eq[d, ok] + zd[ok] @ r.coef_
    return F, r.coef_
if __name__ == "__main__":
    S61, S18 = np.arange(61), np.arange(62, 80)
    print("equal thirds 61d %.3f 18d %.3f" % (day_mae(eq, a)[S61].mean(), day_mae(eq, a)[S18].mean()))
    for fs in ["hod", "dev", "hod+dev", "hod+dev+spread+level", "hod+dev+spread+level+rl+wkend", "hod+dev+spread+level+rl+wkend+lagerr+gas", "lagerr+gas", "level+rl"]:
        for model, alphas in (("huber", (1e-2, 1e1, 1e3)), ("lad", (1e-4, 1e-3))):
            for al in alphas:
                F, c = run(fs, model, al); dm = day_mae(F, a)
                print(f"{fs:45s} {model:5s} a={al:g}  61d {dm[S61].mean():.3f}  18d {dm[S18].mean():.3f}", flush=True)
