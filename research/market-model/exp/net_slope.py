"""Price response to weather-driven residual-load deviations: observed against the curves fitted once (notes 21.144)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1n as M
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}
Z = np.load(f"{SP}/out/shift15.npz"); S = pd.read_parquet(f"{SP}/out/star_data.parquet"); idx = S.index.tz_convert("UTC")
q = pd.DataFrame({"p": Z["p"], "rl": Z["rl"], "other": Z["other"], "nb": Z["nbterm"]}, index=idx); h = q.resample("1h").mean()
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; co2 = pd.Series(E[:len(idx)], index=idx[:len(E)]).resample("1h").mean().reindex(h.index).ffill()
h["L"] = (1.85 * R.gas.reindex(h.index).ffill() + 0.37 * co2).shift(48).ffill().bfill(); h["co2"] = co2.shift(48).ffill().bfill(); h["D"] = h.rl - h.other
h = h.dropna(); loc = h.index.tz_convert("Europe/Berlin"); h["day"] = loc.normalize(); h["hm"] = loc.hour.astype(str) + "_" + loc.month.astype(str) + "_" + (loc.weekday >= 5).astype(str); h["half"] = loc.year.astype(str) + np.where(loc.month <= 6, "H1", "H2")
# the shift at the observed price, and its day mean
sup = M.supply(th, h.p.to_numpy(), h.L.to_numpy(), h.co2.to_numpy()) - th["K_s"] * ((h.p.to_numpy() < -M.EPS) + 0.5 * (np.abs(h.p.to_numpy()) <= M.EPS)); h["u"] = h.D - sup - h.nb
h["ubar"] = h.groupby("day").u.transform("mean"); h["pm"] = np.asarray(M.clear(th, h.D.to_numpy(), h.L.to_numpy(), h.co2.to_numpy(), eps=(h.ubar + h.nb).to_numpy()))
def resid(col):  # remove day and hour x month x day-type means (two-way, a few alternating passes)
    x = h[col].copy()
    for _ in range(6): x = x - x.groupby(h.day).transform("mean"); x = x - x.groupby(h.hm).transform("mean")
    return x
for c in ("p", "pm", "D"): h[c + "_r"] = resid(c)
h["lvl"] = pd.cut(h.groupby("day").p.transform("mean") / h.groupby("day").L.transform("mean"), [-9, 0.7, 0.95, 1.2, 9], labels=["low (<0.7)", "middle (0.7-0.95)", "high (0.95-1.2)", "very high (>1.2)"])
sl = lambda g, y: float((g[y] * g.D_r).sum() / (g.D_r ** 2).sum())
rows = []
for (hf, lv), g in h.groupby(["half", "lvl"]):
    if len(g) < 300: continue
    rows.append(dict(half=hf, level=lv, hours=len(g), observed=sl(g, "p_r"), model=sl(g, "pm_r")))
T = pd.DataFrame(rows); T["ratio"] = T.observed / T.model
print("slope of price on demand, EUR/MWh per GW (weather-driven deviations; day and hour-of-day x month x day-type removed)")
print(T.pivot(index="level", columns="half", values="ratio").round(2).to_string()); print()
print(T.pivot(index="level", columns="half", values="observed").round(1).to_string()); print()
print(T.pivot(index="level", columns="half", values="model").round(1).to_string()); print()
print(T.pivot(index="level", columns="half", values="hours").to_string())
A = h.groupby("half").apply(lambda g: pd.Series({"observed": sl(g, "p_r"), "model": sl(g, "pm_r"), "sd_D_resid_GW": g.D_r.std()})); A["ratio"] = A.observed / A.model; print(); print(A.round(2).to_string())
T.to_csv(f"{SP}/out/net_slope.csv"); A.to_csv(f"{SP}/out/net_slope_half.csv")

# counterfactual (notes 21.145): imports and storage respond to the price in EUR at the fit window's cost level, not to p / L
fitw = np.asarray(h.index < pd.Timestamp("2025-12-01", tz="UTC")); Lref = float(h.L[fitw].mean()); print("\nL_ref", round(Lref, 1), "| mean L by half year", h.groupby("half").L.mean().round(1).to_dict())
def supply_cf(p, L, co2):
    fu = M.fuels(th, p, L, co2); gX, gS = M.other_curves(th, p, np.full_like(p, Lref)); return sum(fu.values()) + gX + gS + th["B_g"]
def clear_cf(D, L, co2, eps):
    lo, hi = np.full(len(D), -500.0), np.full(len(D), 3000.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); ex = supply_cf(mid, L, co2) + eps - th["K_s"] * (mid < -M.EPS) - D; lo = np.where(ex < 0, mid, lo); hi = np.where(ex < 0, hi, mid)
    return 0.5 * (lo + hi)
p_, L_, c_ = h.p.to_numpy(), h.L.to_numpy(), h.co2.to_numpy(); sup2 = supply_cf(p_, L_, c_) - th["K_s"] * ((p_ < -M.EPS) + 0.5 * (np.abs(p_) <= M.EPS)); h["u2"] = h.D - sup2 - h.nb
h["u2bar"] = h.groupby("day").u2.transform("mean"); h["pm2"] = clear_cf(h.D.to_numpy(), L_, c_, (h.u2bar + h.nb).to_numpy()); h["pm2_r"] = resid("pm2")
B = h.groupby("half").apply(lambda g: pd.Series({"observed": sl(g, "p_r"), "model": sl(g, "pm_r"), "model, imports in EUR": sl(g, "pm2_r")})); B["ratio"] = B.observed / B.model; B["ratio, imports in EUR"] = B.observed / B["model, imports in EUR"]
print(B.round(2).to_string())
rows = []
for (hf, lv), g in h.groupby(["half", "lvl"]):
    if len(g) >= 300: rows.append(dict(half=hf, level=lv, r1=sl(g, "p_r") / sl(g, "pm_r"), r2=sl(g, "p_r") / sl(g, "pm2_r")))
T2 = pd.DataFrame(rows); print(T2.pivot(index="level", columns="half", values="r2").round(2).to_string()); B.to_csv(f"{SP}/out/net_slope_cf.csv"); T2.to_csv(f"{SP}/out/net_slope_cf_levels.csv")
