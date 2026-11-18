"""Quarter-hour pieces of the supply shift for M1n (notes 21.51): S(p) at every known quarter-hour price, the neighbour term, published residual load,
other plant. The shift itself, u = (rl - other) - S(p) - neighbour term, is assembled per origin in ts_shift.py with what is known at the issue.
Fuel costs lagged 48 h everywhere (the day before the target has not happened at a midnight issue)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, os
NONB = os.environ.get("NONB", "0") == "1"   # ablation 21.54: M1, no neighbour term
MOD = os.environ.get("MOD", ""); MD = MOD == "d"; MB = MOD == "b"
SP = M.SP; P = np.load(f"{SP}/out/market1{'' if NONB else 'n'}_post.npz"); PN_ = np.load(f"{SP}/out/market1{'d' if MD else 'b' if MB else 'n'}_post.npz"); th = {k: float(np.median(P[k])) for k in M.PN}; SUP = M.supply
if MD:
    import market1d as Dm; th = {k: float(np.median(PN_[k])) for k in Dm.PN}; SUP = Dm.supply
if MB:
    import market1b as Mb; th = {k: float(np.median(PN_[k])) for k in Mb.PN}; SUP = Mb.supply
B = np.median(PN_["kn"], 0) / PN_["NSD"] * (0.0 if NONB else 1.0)
nb = pd.read_parquet(f"{SP}/out/nb_rl.parquet"); idx = nb.index; NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]
S = pd.read_parquet(f"{SP}/out/star_data.parquet"); S.index = S.index.tz_convert("UTC"); S = S.reindex(idx); R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC")
E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; R["co2"] = np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1); R["L"] = 1.85 * R.gas + 0.37 * R.co2
H = R[["p", "other", "L", "co2"]].reindex(idx, method="ffill"); p = S.p_de.values.astype(float); print("quarter-hour price vs hourly price: mean abs difference EUR/MWh by half-year", {str(k): round(float(v), 2) for k, v in (pd.Series(np.abs(p - H.p.values), index=idx).resample("6MS").mean()).items()})
L = H.L.shift(192).ffill().bfill().values; co2 = H.co2.shift(192).ffill().bfill().values; ok = ~np.isnan(p); wd = th["K_s"] * (p < -M.EPS) + 0.5 * th["K_s"] * (np.abs(p) <= M.EPS); Sp = np.full(len(idx), np.nan); Sp[ok] = np.asarray(SUP(th, p[ok], L[ok], co2[ok])) - wd[ok]
nbterm = (nb[NBC].ffill().values - PN_["NMU"]) @ B
np.savez_compressed(f"{SP}/out/shift15{'_m1nonb' if NONB else '_d' if MD else '_b' if MB else ''}.npz", Sp=Sp, rl=S.rl_de.values.astype(float), other=H.other.values.astype(float), nbterm=nbterm, p=p, NMU=PN_["NMU"], B=B)
u = S.rl_de.values - H.other.values - Sp - nbterm; print("shift u GW: mean", round(float(np.nanmean(u)), 2), "sd", round(float(np.nanstd(u)), 2), "| nan share", round(float(np.isnan(u).mean()), 4), "| share of slots at or below zero price", round(float((p[ok] <= M.EPS).mean()), 3))
loc = idx.tz_convert("Europe/Berlin"); night = np.asarray(loc.hour < 6)
for nm, a, b in (("winter", "2025-12-01", "2026-03-01"), ("summer", "2026-03-31", "2026-08-28"), ("autumn", "2026-09-02", "2026-11-08")):
    m = np.asarray((idx >= a) & (idx < b)) & night & ok; print(nm, "night slots at or below zero price:", round(float((p[m] <= M.EPS).mean()), 3), "| sd of u at night", round(float(np.nanstd(u[m])), 2), "| mean abs quarter-to-quarter change of u", round(float(np.nanmean(np.abs(np.diff(u)))), 2))
