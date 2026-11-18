"""Horizon table for a market-model level (notes 21.10): price error 1-7 days ahead of a 14:00 origin with the state carried (mean of the last 7 days by hour
of day, data to 12:00 of the origin day), against two reduced-form baselines: level 1 (price = L k(RL)) with and without its own price error carried.
usage: market_horizon.py market0|market0d      Realised load and fuel prices of the target hour: isolates the price model, not a forecast test."""
import sys, importlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
MOD = sys.argv[1]; M = importlib.import_module(MOD); SP = M.SP; KW = {"market0": dict(d="d", eT="eT", eX="eX"), "market0d": dict(eps="eps", eT="eT", eX="eX", eS="eS")}[MOD]; DEM = {"market0": "rlreal", "market0d": "D"}[MOD]
R = pd.read_parquet(f"{SP}/out/{MOD}_rows.parquet").join(pd.read_parquet(f"{SP}/out/zoom1_rows.parquet")[["fit", "res"]], how="left"); th = {k: float(np.median(np.load(f"{SP}/out/{MOD}_post.npz")[k])) for k in M.PN}; R["po"] = (R.p - R.p0) / R.L
full = pd.date_range(R.index.min().tz_convert("UTC").floor("D"), R.index.max().tz_convert("UTC").ceil("D"), freq="1h", tz="UTC"); G = R.copy(); G.index = G.index.tz_convert("UTC"); G = G.reindex(full); loc = G.index.tz_convert("Europe/Berlin"); hod = np.asarray(loc.hour); n = len(G)
def state(v, h, N=7):
    lag0 = np.where(hod <= 12, 24 * h, 24 * (h + 1)); acc = np.zeros(n); cnt = np.zeros(n); idx = np.arange(n)
    for j in range(N):
        src = idx - lag0 - 24 * j; x = np.where(src >= 0, v[np.clip(src, 0, n - 1)], np.nan); m = ~np.isnan(x); acc[m] += x[m]; cnt[m] += 1
    return np.where(cnt >= 4, acc / np.maximum(cnt, 1), np.nan)
tr = (G.tr == True).values; ho = (G.tr == False).values; night = hod < 8; p = G.p.values; L = G.L.values; F = {}; NM = ("market model, no state", "market model + price error carried", "market model + shocks carried", "level 1, no state", "level 1 + price error carried")
for h in range(1, 8):
    st = {a: state(G[c].values, h) for a, c in KW.items()}; so = state(G.po.values, h); s1 = state(G.res.values, h); m = ~np.isnan(sum(st.values()) + so + s1 + p + G.fit.values); pq = np.full(n, np.nan); pq[m] = M.clear(th, G[DEM].values[m], L[m], **{a: v[m] for a, v in st.items()})[0]
    F[h] = (m, (G.p0.values, G.p0.values + so * L, pq, G.fit.values * L, (G.fit.values + s1) * L))
print(f"{MOD}: hourly price MAE in ct. Columns: " + " | ".join(NM))
for nm, msk in (("fitting period", tr), ("held-out year", ho)):
    for sub, ms in (("all hours", np.ones(n, bool)), ("night 00-08", night)):
        for h in (1, 2, 4, 7):
            m, V = F[h]; k = m & msk & ms; print(f"  {nm:15s} {sub:11s} +{h}d  " + "  ".join(f"{np.mean(np.abs(v[k] - p[k])) / 10:5.2f}" for v in V))
# the candidate's origins, D+2, hourly prices and quarter-hour slots
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["p_de"]]; S.index = S.index.tz_convert("UTC"); gi = G.index.get_indexer(S.index.floor("1h")); okq = gi >= 0; m, V = F[2]; l15 = S.index.tz_convert("Europe/Berlin"); day = np.asarray(l15.normalize().tz_localize(None)); slot = np.asarray(l15.hour * 4 + l15.minute // 15); p15 = S.p_de.values
q = lambda v: np.where(okq, np.asarray(v, float)[np.clip(gi, 0, n - 1)], np.nan); m15 = q(m.astype(float)) > 0.5; PER = {"winter": pd.date_range("2025-12-01", periods=30, freq="3D"), "summer": pd.date_range("2026-03-31", periods=50, freq="3D"), "autumn": pd.date_range("2026-09-02", periods=66, freq="1D")}; CAND = {"winter": (0.67, 0.88), "summer": (0.79, 1.28), "autumn": (1.31, 1.63)}
print("candidate's origins, D+2, scored on quarter-hour slots (hourly forecast repeated), ct; last column: the candidate itself (forecast load)")
for nm, org in PER.items():
    tg = np.isin(day, (org + pd.Timedelta(days=2)).values)
    for j, (lab, ns) in enumerate((("night", 32), ("first 14 h", 56))):
        k = m15 & tg & (slot < ns) & ~np.isnan(p15); print(f"  {nm:7s} {lab:10s} " + "  ".join(f"{np.mean(np.abs(q(v)[k] - p15[k])) / 10:5.2f}" for v in V) + f"   [{CAND[nm][j]:.2f}]   hourly-scored (= slots with the true within-hour shape known): " + "  ".join(f"{np.mean(np.abs(q(v)[k] - q(p)[k])) / 10:5.2f}" for v in V))
np.savez(f"{SP}/out/{MOD}_horizon.npz", **{f"h{h}_{i}": F[h][1][i] for h in F for i in range(5)}, **{f"m{h}": F[h][0] for h in F})
