"""Screening of candidate state variables on data before 2026-03-31 only (the summer forecast period starts there). Target: German price / gas-fired cost.
Baseline: piecewise-linear function of the regional load index (German residual load + 0.3 x neighbours) + hour of day + day type.
For each candidate (lagged so that it is known at a forecast origin: built from days <= d-3, D+1 price information from day d-1):
  coefficient per standard deviation in each half of the screening data, and the out-of-half test: fit on one half, change in mean absolute error on the other.
A candidate passes if the sign agrees in both halves and the error falls in both out-of-half tests."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); N = pd.read_parquet(f"{SP}/out/nb_supply.parquet").reindex(D.index); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); day = loc.normalize(); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); L = (1.85 * D["gas"] + 0.37 * ed.reindex(day.tz_localize(None)).values); y = (D["p_de"] / L)
NB = ["fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]; nb = sum(D[f"rl_{z}"] for z in NB); x0 = D["rl_de"] + 0.3 * (nb - nb.mean())
def daily(s, fn): return s.groupby(day).agg(fn)
def lagged(dly, lag): v = dly.shift(lag); return pd.Series(v.reindex(day).values, index=D.index)
def trail(s, days, fn, lag=3):                                                                              # statistic over the `days` days ending at d - lag
    dm = daily(s, "max") if fn == "max" else daily(s, "mean"); r = dm.rolling(days, min_periods=max(1, days // 2)).max() if fn == "max" else dm.rolling(days, min_periods=max(1, days // 2)).mean(); return lagged(r, lag)
C = {}
nz = [c for c in ("nuc_fr", "nuc_be", "nuc_ch", "nuc_cz", "nuc_nl") if c in N]
C["nuclear availability, all neighbours, 14-day max"] = sum(trail(N[c], 14, "max") for c in nz); C["nuclear availability, all neighbours, 3-day max"] = sum(trail(N[c], 3, "max") for c in nz)
C["nuclear availability, France only, 3-day max"] = trail(N["nuc_fr"], 3, "max"); C["nuclear availability, BE + CH + CZ + NL, 3-day max"] = sum(trail(N[c], 3, "max") for c in nz if c != "nuc_fr")
C["run-of-river AT + CH + FR + DE, 7-day mean"] = sum(trail(N[c], 7, "mean") for c in ("ror_at", "ror_ch", "ror_fr", "ror_de") if c in N)
sol = D["g_de_lignite"] + D["g_de_coal"]; C["German lignite + hard coal, 14-day max"] = trail(sol, 14, "max"); C["German lignite + hard coal, yesterday's max"] = trail(sol, 1, "max"); C["German gas, yesterday's max"] = trail(D["g_de_gas"], 1, "max")
try:
    O = pd.read_parquet(f"{SP}/out/de_outage_hourly.parquet").reindex(D.index); C["German outages lignite + coal + gas, hindsight (ceiling)"] = O["lignite"] + O["hard coal"] + O["gas"]
except Exception as e: print("no outage file", e)
PR = {z: D[f"p_{z}"] for z in NB}
for z, f in (("itn", "p_itn"), ("no2", "p_no2"), ("se4", "p_se4"), ("si", "p_si"), ("hu", "p_hu")):
    try: PR[z] = pd.read_parquet(f"{SP}/ext/smard/series/{f}_DE_hour.parquet")[f].reindex(D.index)
    except Exception as e: print("missing", f)
spread = {z: lagged(daily((PR[z] - D["p_de"]) / L, "mean"), 1) for z in PR}
for z in PR: C[f"D+1 spread to Germany / gas-fired cost: {z}"] = spread[z]
for nm, zs in (("south (itn at ch si hu)", ("itn", "at", "ch", "si", "hu")), ("west (fr be nl)", ("fr", "be", "nl")), ("north (no2 se4 dk)", ("no2", "se4", "dk")), ("east (pl cz)", ("pl", "cz"))):
    zz = [z for z in zs if z in spread]; C[f"D+1 spread, mean of {nm}"] = sum(spread[z] for z in zz) / len(zz)
C["D+1 spread, mean of all ring zones"] = sum(spread.values()) / len(spread)
# baseline design
kn = np.arange(-5, 61, 5.0); B = [np.ones(len(D)), x0.values] + [np.maximum(x0.values - k, 0.0) for k in kn]; hr = loc.hour.values; dt = np.where(loc.dayofweek == 6, 2, np.where(loc.dayofweek == 5, 1, 0))
B += [(hr == h).astype(float) for h in range(1, 24)] + [(dt == k).astype(float) for k in (1, 2)]; X0 = np.column_stack(B)
scr = (loc >= "2024-07-15") & (loc < "2026-03-31") & np.isfinite(y.values) & np.isfinite(X0).all(1); mid = pd.Timestamp("2025-05-22", tz="Europe/Berlin"); hA = scr & (loc < mid); hB = scr & (loc >= mid)
def fitpred(Xa, ya, Xb):
    b, *_ = np.linalg.lstsq(Xa, ya, rcond=None); return Xb @ b, b
# own lagged level: residual of the baseline (fitted on the screening data) averaged over day d-1
r0 = pd.Series(y.values - X0 @ np.linalg.lstsq(X0[scr], y.values[scr], rcond=None)[0], index=D.index); C["Germany's own D+1 residual (level)"] = lagged(daily(r0, "mean"), 1)
Lm = L.values; print(f"screening hours {scr.sum()} (first half {hA.sum()}, second {hB.sum()}); baseline in-sample MAE {np.mean(np.abs(r0.values[scr] * Lm[scr])) / 10:.2f} ct")
print(f"{'candidate':62s} {'coef/sd A':>9s} {'coef/sd B':>9s} | out-of-half MAE change, ct: fit A->B, fit B->A | {'verdict':8s}"); res = {}
for name, c in C.items():
    cv = c.values.astype(float); ok = np.isfinite(cv); sd = np.nanstd(cv[scr & ok]); z = (cv - np.nanmean(cv[scr & ok])) / sd; out = []
    for a_, b_ in ((hA, hB), (hB, hA)):
        ka, kb = a_ & ok, b_ & ok; p0, _ = fitpred(X0[ka], y.values[ka], X0[kb]); X1a = np.column_stack([X0[ka], z[ka]]); X1b = np.column_stack([X0[kb], z[kb]]); p1, bb = fitpred(X1a, y.values[ka], X1b)
        out.append((bb[-1], (np.mean(np.abs((y.values[kb] - p1) * Lm[kb])) - np.mean(np.abs((y.values[kb] - p0) * Lm[kb]))) / 10))
    passed = (np.sign(out[0][0]) == np.sign(out[1][0])) and out[0][1] < 0 and out[1][1] < 0; res[name] = (out, passed)
    print(f"{name:62s} {out[0][0]:+9.3f} {out[1][0]:+9.3f} | {out[0][1]:+.3f}  {out[1][1]:+.3f} | {'pass' if passed else '-'}")
