"""Screening of 'solid fuel held back' states on data before 2026-03-31 (same protocol as screen.py), and their course through 2026.
held back = what the fleet can deliver (14-day max of hourly output) minus what it delivered in the hours when it was clearly in the money
(lignite margin 30..120 EUR/MWh), both over the 14 days ending three days before the target day."""
import numpy as np, pandas as pd
exec(open("screen.py").read().split("C = {}")[0])                                      # data, helpers, x0, y, L, day, loc
eua_h = pd.Series(ed.reindex(day.tz_localize(None)).values, index=D.index); m_lig = D["p_de"] - (12.0 + eua_h); inm = (m_lig >= 30) & (m_lig < 120)
def held(series, days=14, lag=3):
    dmax = series.groupby(day).max().rolling(days, min_periods=7).max()
    num = (series.where(inm)).groupby(day).sum().rolling(days, min_periods=3).sum(); den = inm.groupby(day).sum().rolling(days, min_periods=3).sum(); dm = (num / den.where(den >= 12)).ffill()
    return lagged(dmax - dm, lag), lagged(dm, lag), lagged(dmax, lag)
C = {}; lig = D["g_de_lignite"]; sol = D["g_de_lignite"] + D["g_de_coal"]
hl, ml, xl = held(lig); hs, ms, xs = held(sol)
C["lignite held back (14-day max minus output when in the money)"] = hl; C["lignite + hard coal held back"] = hs; C["lignite output when in the money (14-day mean)"] = ml; C["lignite 14-day max"] = xl; C["lignite + hard coal 14-day max"] = xs
kn = np.arange(-5, 61, 5.0); B = [np.ones(len(D)), x0.values] + [np.maximum(x0.values - k, 0.0) for k in kn]; hr = loc.hour.values; dt = np.where(loc.dayofweek == 6, 2, np.where(loc.dayofweek == 5, 1, 0))
B += [(hr == h).astype(float) for h in range(1, 24)] + [(dt == k).astype(float) for k in (1, 2)]; X0 = np.column_stack(B); Lm = L.values
scr = (loc >= "2024-07-15") & (loc < "2026-03-31") & np.isfinite(y.values) & np.isfinite(X0).all(1); mid = pd.Timestamp("2025-05-22", tz="Europe/Berlin"); hA = scr & (loc < mid); hB = scr & (loc >= mid)
def fitpred(Xa, ya, Xb): b, *_ = np.linalg.lstsq(Xa, ya, rcond=None); return Xb @ b, b
print(f"{'candidate':66s} {'coef/sd A':>9s} {'coef/sd B':>9s} | out-of-half MAE change, ct: A->B, B->A | verdict")
for name, c in C.items():
    cv = c.values.astype(float); ok = np.isfinite(cv); z = (cv - np.nanmean(cv[scr & ok])) / np.nanstd(cv[scr & ok]); out = []
    for a_, b_ in ((hA, hB), (hB, hA)):
        ka, kb = a_ & ok, b_ & ok; p0, _ = fitpred(X0[ka], y.values[ka], X0[kb]); p1, bb = fitpred(np.column_stack([X0[ka], z[ka]]), y.values[ka], np.column_stack([X0[kb], z[kb]]))
        out.append((bb[-1], (np.mean(np.abs((y.values[kb] - p1) * Lm[kb])) - np.mean(np.abs((y.values[kb] - p0) * Lm[kb]))) / 10))
    ok_ = (np.sign(out[0][0]) == np.sign(out[1][0])) and out[0][1] < 0 and out[1][1] < 0; print(f"{name:66s} {out[0][0]:+9.3f} {out[1][0]:+9.3f} | {out[0][1]:+.3f}  {out[1][1]:+.3f} | {'pass' if ok_ else '-'}")
mth = loc.strftime("%Y-%m"); print("\nmonthly means, GW: lignite held back | lignite 14-day max | lignite when in the money || same for lignite + hard coal")
for mm in sorted(set(mth)):
    if mm >= "2025-01": k = mth == mm; print(f"  {mm}: {np.nanmean(hl[k]):4.1f} | {np.nanmean(xl[k]):5.1f} | {np.nanmean(ml[k]):5.1f} || {np.nanmean(hs[k]):4.1f} | {np.nanmean(xs[k]):5.1f} | {np.nanmean(ms[k]):5.1f}")
