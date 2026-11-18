"""Quarter-hour slots are what is scored. How much of the slot price is within-hour, since when, and what does the hourly market model lose there?
Uses the M0c curves with the 7-day state carried two days ahead (as market0_horizon.py). Realised load and fuel prices: isolates the model, not a forecast test."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.argv = ["x"]; import market0_innov as Q
M = Q.M; SP = M.SP; S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["p_de", "rlreal_de", "rl_de"]]; S.index = S.index.tz_convert("UTC")
hr = S.index.floor("1h"); S["ph"] = S.p_de.groupby(hr).transform("mean"); S["rlh"] = S.rlreal_de.groupby(hr).transform("mean"); S["dev"] = S.p_de - S.ph; S["drl"] = S.rlreal_de - S.rlh
mo = S.dev.abs().groupby(S.index.tz_convert("Europe/Berlin").strftime("%Y-%m")).mean() / 10; print("mean |slot price - hourly mean|, ct, by month:"); print(mo.round(3).to_string())
G = Q.G; st = Q.S; n15 = len(S); gi = G.index.get_indexer(hr); ok = gi >= 0; f = lambda v: np.where(ok, np.asarray(v)[np.clip(gi, 0, len(G) - 1)], np.nan)
L15, sT, sX, sD, pqh = f(G.L.values), f(st["eT"]), f(st["eX"]), f(st["d"]), f(Q.pq); m = ~np.isnan(L15 + sT + sX + sD + S.rlreal_de.values + S.p_de.values + pqh); pq15 = np.full(n15, np.nan); pq15[m] = M.clear(Q.th, S.rlreal_de.values[m], L15[m], d=sD[m], eT=sT[m], eX=sX[m])[0]
loc = S.index.tz_convert("Europe/Berlin"); day = np.asarray(loc.normalize().tz_localize(None)); slot = np.asarray(loc.hour * 4 + loc.minute // 15); p = S.p_de.values
PER = {"winter": pd.date_range("2025-12-01", periods=30, freq="3D"), "summer": pd.date_range("2026-03-31", periods=50, freq="3D"), "autumn": pd.date_range("2026-09-02", periods=66, freq="1D")}
print("\nD+2 from the candidate's origins, scored on quarter-hour slots (ct): hourly model repeated over the hour | same model cleared each quarter | floor of any hourly forecast (the true hourly mean)")
for nm, org in PER.items():
    tg = np.isin(day, (org + pd.Timedelta(days=2)).values)
    for lab, ns in (("night", 32), ("first 14 h", 56)):
        k = m & tg & (slot < ns); print(f"  {nm:7s} {lab:10s} {np.mean(np.abs(pqh[k] - p[k])) / 10:5.2f} | {np.mean(np.abs(pq15[k] - p[k])) / 10:5.2f} | {np.mean(np.abs(S.ph.values[k] - p[k])) / 10:5.2f}   ({int(k.sum())} slots)")
# does clearing each quarter on the same curve give the within-hour shape? slope of observed within-hour deviation on the model's
k = m & (S.index >= pd.Timestamp("2025-12-01", tz="UTC")); dm = pq15 - pqh; dv = S.dev.values
for lab, kk in (("all hours", k), ("slots 0-31", k & (slot < 32)), ("slots 24-55 (morning ramp)", k & (slot >= 24) & (slot < 56))):
    b = np.polyfit(dm[kk], dv[kk], 1)[0]; c = np.corrcoef(dm[kk], dv[kk])[0, 1]; print(f"within-hour deviation, observed on model's, {lab}: slope {b:.2f}, correlation {c:.2f}; sd observed {np.std(dv[kk]) / 10:.2f} ct, model {np.std(dm[kk]) / 10:.2f} ct")
