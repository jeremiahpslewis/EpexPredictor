"""Screening of owner-level supply states on data before 2026-03-31 (two-half stability test as in screen.py)."""
import numpy as np, pandas as pd
exec(open("screen.py").read().split("C = {}")[0])
O = pd.read_parquet(f"{SP}/out/owner_states.parquet").reindex(D.index); C = {}
for k in ("lig_rwe", "lig_leag", "lig_enbw", "lig_saale", "coal_enbw", "coal_other", "gas_rwe", "gas_other"): C[f"in-the-money output, {k}"] = O[f"zin_{k}"]
C["in-the-money output, large portfolios (RWE + EnBW, lignite + coal)"] = O["zin_lig_rwe"] + O["zin_lig_enbw"] + O["zin_coal_enbw"]; C["in-the-money output, other owners (LEAG, Saale, other coal)"] = O["zin_lig_leag"] + O["zin_lig_saale"] + O["zin_coal_other"]
C["in-the-money output, all solid-fuel blocks"] = sum(O[f"zin_{k}"] for k in ("lig_rwe", "lig_leag", "lig_enbw", "lig_saale", "coal_enbw", "coal_other")); C["14-day max, all solid-fuel blocks (lag 7)"] = sum(O[f"max_{k}"] for k in ("lig_rwe", "lig_leag", "lig_enbw", "lig_saale", "coal_enbw", "coal_other"))
kn = np.arange(-5, 61, 5.0); B = [np.ones(len(D)), x0.values] + [np.maximum(x0.values - k, 0.0) for k in kn]; hr = loc.hour.values; dt = np.where(loc.dayofweek == 6, 2, np.where(loc.dayofweek == 5, 1, 0))
B += [(hr == h).astype(float) for h in range(1, 24)] + [(dt == k).astype(float) for k in (1, 2)]; X0 = np.column_stack(B); Lm = L.values
scr = (loc >= "2024-08-01") & (loc < "2026-03-31") & np.isfinite(y.values) & np.isfinite(X0).all(1); mid = pd.Timestamp("2025-05-30", tz="Europe/Berlin"); hA = scr & (loc < mid); hB = scr & (loc >= mid)
def fitpred(Xa, ya, Xb): b, *_ = np.linalg.lstsq(Xa, ya, rcond=None); return Xb @ b, b
print(f"{'candidate (GW; built from days <= d-7)':70s} {'coef/GW A':>9s} {'coef/GW B':>9s} | out-of-half MAE change, ct: A->B, B->A | verdict")
for name, c in C.items():
    cv = c.values.astype(float); ok = np.isfinite(cv); z = cv - np.nanmean(cv[scr & ok]); out = []
    for a_, b_ in ((hA, hB), (hB, hA)):
        ka, kb = a_ & ok, b_ & ok; p0, _ = fitpred(X0[ka], y.values[ka], X0[kb]); p1, bb = fitpred(np.column_stack([X0[ka], z[ka]]), y.values[ka], np.column_stack([X0[kb], z[kb]]))
        out.append((bb[-1], (np.mean(np.abs((y.values[kb] - p1) * Lm[kb])) - np.mean(np.abs((y.values[kb] - p0) * Lm[kb]))) / 10))
    ok_ = (np.sign(out[0][0]) == np.sign(out[1][0])) and out[0][1] < 0 and out[1][1] < 0; print(f"{name:70s} {out[0][0]:+9.3f} {out[1][0]:+9.3f} | {out[0][1]:+.3f}  {out[1][1]:+.3f} | {'pass' if ok_ else '-'}")
# joint regression with the two owner classes side by side: is a GW from a large portfolio worth the same as a GW from the others?
for lab, k_ in (("first half", hA), ("second half", hB), ("all screening data", scr)):
    a1 = C["in-the-money output, large portfolios (RWE + EnBW, lignite + coal)"].values; a2 = C["in-the-money output, other owners (LEAG, Saale, other coal)"].values; ok = k_ & np.isfinite(a1) & np.isfinite(a2)
    b = np.linalg.lstsq(np.column_stack([X0[ok], a1[ok] - np.nanmean(a1[scr]), a2[ok] - np.nanmean(a2[scr])]), y.values[ok], rcond=None)[0]; print(f"  joint, {lab}: effect on price / gas-fired cost per GW: large portfolios {b[-2]:+.3f}, other owners {b[-1]:+.3f}")
