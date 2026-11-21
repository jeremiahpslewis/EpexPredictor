"""
Counterfactual (1 GW German gas outage, announced before the auction) and the external slope check, per posterior draw.
Usage: cf.py PANEL_DIR POST.npz OUT_PREFIX [NDRAWS]
Per draw: recover shocks u from the observed (p_DE, p_N, m); re-clear the baseline (must reproduce the data);
re-clear with the outage ('cheapest' = 1 GW of in-merit gas, primary; 'average' = available gas - 1 GW);
re-clear with +1 GW German demand (dp/dD, for the external check). Everything is model minus model baseline.
"""
import sys, json
import numpy as np, pandas as pd, jax, jax.numpy as jnp
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as Mo

PANEL, POST, OUTP = sys.argv[1], sys.argv[2], sys.argv[3]
ND = int(sys.argv[4]) if len(sys.argv) > 4 else 200
Pn = pd.read_parquet(f"{PANEL}/panel.parquet"); x, obs, brk = Mo.to_inputs(Pn)
S = dict(np.load(POST)); nS = len(S["h_G"]); pick = np.linspace(0, nS - 1, ND).astype(int)

@jax.jit
def one(th):
    u, _, _ = Mo.invert(th, x, obs["pDE"], obs["pN"], obs["m"])
    b = Mo.clear(th, x, u); c1 = Mo.clear(th, x, u, outage=1.0, kind="cheapest")
    c2 = Mo.clear(th, x, u, outage=1.0, kind="average"); d = Mo.clear(th, x, u, dD=1.0)
    Sb, dSb = Mo.fuel_supply(th, x, b["pDE"]); _, dM = Mo.exch(th, b["pDE"] - b["pN"]); _, dSN = Mo.n_supply(th, x, b["pN"])
    gas_share = dSb["G"] / (dSb["L"] + dSb["C"] + dSb["G"] + 1e-12)       # gas share of the German supply slope at baseline
    return b, c1, c2, d, gas_share

keys = ["pDE", "pN", "m", "L", "C", "G"]
res = {k: [] for k in ["base_err_p", "base_err_m", "acct1", "acct2"]}
D1 = {k: [] for k in keys}; D2 = {k: [] for k in keys}; SL = []; GS = []
for i in pick:
    th = {k: jnp.asarray(S[k][i]) for k in Mo.THETA_KEYS}
    b, c1, c2, d, gs = one(th)
    res["base_err_p"].append(float(jnp.abs(b["pDE"] - obs["pDE"]).max())); res["base_err_m"].append(float(jnp.abs(b["m"] - obs["m"]).max()))
    for k in keys: D1[k].append(np.asarray(c1[k] - b[k])); D2[k].append(np.asarray(c2[k] - b[k]))
    res["acct1"].append(float(jnp.abs((c1["L"] + c1["C"] + c1["G"] + c1["m"]) - (b["L"] + b["C"] + b["G"] + b["m"])).max()))
    res["acct2"].append(float(jnp.abs((c2["L"] + c2["C"] + c2["G"] + c2["m"]) - (b["L"] + b["C"] + b["G"] + b["m"])).max()))
    SL.append(np.asarray(d["pDE"] - b["pDE"])); GS.append(np.asarray(gs))
D1 = {k: np.array(v) for k, v in D1.items()}; D2 = {k: np.array(v) for k, v in D2.items()}; SL = np.array(SL); GS = np.array(GS)
np.savez_compressed(OUTP + "_draws.npz", **{f"cheap_{k}": v.astype(np.float32) for k, v in D1.items()},
                    **{f"avg_{k}": v.astype(np.float32) for k, v in D2.items()}, slope=SL.astype(np.float32), gas_share=GS.astype(np.float32), pick=pick)
p = Pn.p.to_numpy(); pn_obs = Pn.pN.to_numpy()
# hours where the counterfactual has no feasible clearing below the price cap (supply exhausted, no demand response)
cap1 = (p + D1["pDE"] >= Mo.PMAX - 1) | (pn_obs + D1["pN"] >= Mo.PMAX - 1); cap2 = (p + D2["pDE"] >= Mo.PMAX - 1) | (pn_obs + D2["pN"] >= Mo.PMAX - 1)
acc1 = np.abs(D1["L"] + D1["C"] + D1["G"] + D1["m"]); acc2 = np.abs(D2["L"] + D2["C"] + D2["G"] + D2["m"])
res["acct1"] = [float(acc1[~cap1].max())]; res["acct2"] = [float(acc2[~cap2].max())]
out = {"checks": {k: float(np.max(v)) for k, v in res.items()}, "ndraws": ND, "nhours": int(len(Pn)),
       "cap_hours_cheapest": dict(mean_per_draw=float(cap1.sum(1).mean()), q05=float(np.quantile(cap1.sum(1), .05)), q95=float(np.quantile(cap1.sum(1), .95))),
       "cap_hours_average": dict(mean_per_draw=float(cap2.sum(1).mean()))}
for k in keys: D1[k] = np.where(cap1, np.nan, D1[k]); D2[k] = np.where(cap2, np.nan, D2[k])
print("checks (max over draws and hours; accounting outside cap hours):", out["checks"], out["cap_hours_cheapest"])

def summ(D, mask=None):
    r = {}
    for k in keys:
        v = D[k] if mask is None else np.where(mask, D[k], np.nan)
        per_draw = np.nanmean(v, axis=1)                                   # across-hour mean, one number per draw
        med_draw = int(np.argsort(per_draw)[len(per_draw) // 2]); hrs = v[med_draw][~np.isnan(v[med_draw])]
        r[k] = dict(mean=float(per_draw.mean()), draw_q05=float(np.quantile(per_draw, 0.05)), draw_q95=float(np.quantile(per_draw, 0.95)),
                    hour_q10=float(np.quantile(hrs, 0.10)), hour_q50=float(np.quantile(hrs, 0.50)), hour_q90=float(np.quantile(hrs, 0.90)))
    return r
gm = GS > 0.5                                           # gas carries most of the German slope at the margin
out["cheapest"] = {"all": summ(D1), "gas_marginal": summ(D1, gm), "gas_not_marginal": summ(D1, ~gm),
                   "share_hours_gas_marginal": float(gm.mean())}
out["average"] = {"all": summ(D2), "gas_marginal": summ(D2, gm)}
for band, (lo, hi) in {"p<50": (-1e9, 50), "50-100": (50, 100), "100-150": (100, 150), ">150": (150, 1e9)}.items():
    mk = np.broadcast_to((p >= lo) & (p < hi), D1["pDE"].shape)
    out["cheapest"][f"band {band} hours"] = int(((p >= lo) & (p < hi)).sum())
    out["cheapest"][f"band {band}"] = {k: float(np.nanmean(np.where(mk, D1[k], np.nan))) for k in keys}
# example days (posterior median per hour and 90% band across draws)
ex = {}
for dname in ["2026-02-11", "2026-08-12"]:
    sel = np.flatnonzero(Pn.day.astype(str).str[:10].to_numpy() == dname)
    ex[dname] = {"hour": Pn.hour.to_numpy()[sel].tolist(), "p_obs": p[sel].round(1).tolist(),
                 **{f"d{k}_{q}": np.nanquantile(D1[k][:, sel], qq, axis=0).round(3).tolist() for k in ["pDE", "m", "G", "C", "L", "pN"] for q, qq in (("q05", .05), ("q50", .5), ("q95", .95))}}
out["example_days"] = ex

# ---- external check: model dp/dD against the empirical net slope (21.144 recipe, same hours)
dfe = pd.DataFrame({"p": p, "D": Pn.DDE.to_numpy(), "DN": Pn.DN.to_numpy(), "day": Pn.day.to_numpy(), "hour": Pn.hour.to_numpy(),
                    "month": Pn.index.month, "wk": (Pn.day.dt.dayofweek >= 5).astype(int)}, index=Pn.index)
cell = dfe.hour.astype(str) + "_" + dfe.month.astype(str) + "_" + dfe.wk.astype(str)
def resid(s):
    r = s - s.groupby(dfe.day).transform("mean"); return r - r.groupby(cell).transform("mean")
pr, Dr, DNr = resid(dfe.p), resid(dfe.D), resid(dfe.DN)
def ols(y, X): return np.linalg.lstsq(X, y, rcond=None)[0]
capS = p + SL >= Mo.PMAX - 1; out["cap_hours_plus1GW"] = float(capS.sum(1).mean()); SL = np.where(capS, np.nan, SL)
lev = pd.qcut(dfe.p.groupby(dfe.day).transform("mean"), 3, labels=["low", "middle", "high"])
ext = {}
for name, mk in [("all", np.ones(len(dfe), bool))] + [(l, (lev == l).to_numpy()) for l in ["low", "middle", "high"]]:
    X = np.c_[Dr[mk], DNr[mk]]; b_emp = ols(pr[mk].to_numpy(), X)[0]
    # model analogue: the same regression weights (residual demand variance) applied to the model's local slopes
    w = np.broadcast_to(Dr[mk].to_numpy() ** 2, SL[:, mk].shape); ok = ~np.isnan(SL[:, mk])
    per_draw = np.where(ok, SL[:, mk] * w, 0).sum(1) / np.where(ok, w, 0).sum(1)
    ext_med = float(np.nanmedian(SL[:, mk]))
    # bootstrap by day for the empirical slope
    days = dfe.day[mk].to_numpy(); ud = np.unique(days); rng = np.random.default_rng(0); bs = []
    idx_by_day = pd.Series(np.arange(mk.sum())).groupby(days).apply(np.array).to_dict()
    for _ in range(300):
        ii = np.concatenate([idx_by_day[d] for d in rng.choice(ud, len(ud))]); bs.append(ols(pr[mk].to_numpy()[ii], X[ii])[0])
    ext[name] = dict(empirical=float(b_emp), emp_q05=float(np.quantile(bs, .05)), emp_q95=float(np.quantile(bs, .95)),
                     model=float(per_draw.mean()), model_q05=float(np.quantile(per_draw, .05)), model_q95=float(np.quantile(per_draw, .95)),
                     ratio_emp_over_model=float(b_emp / per_draw.mean()), model_hour_median=ext_med, hours=int(mk.sum()))
out["external_slope"] = ext
json.dump(out, open(OUTP + ".json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "example_days"}, indent=1))
