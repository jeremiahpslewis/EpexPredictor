"""Hourly shape of the trough-day effect: mean implied index shift by (how deep the day's trough is) x (where the hour sits relative to the day's mean index),
and by hour of day. Out-of-sample hours only (each hour scored by the latest weekly fit before it)."""
import sys, numpy as np, pandas as pd
import oc, ocx2
from ocx_state import *
R = []
for t in fit_dates():
    A = get_fit(t); h = hx(t); sl = slice(h, min(h + OOS * 24, oc.NH)); n = sl.stop - sl.start
    if n < 24: continue
    d = rows2(sl, drl=np.zeros(n)); dx, code = implied(A, d); dd = {k: np.nan_to_num(v) for k, v in d.items()}; x = index2(A, dd, np); ph = pcurve(A, x, dd["gas"], dd["eua"], np)[0]; days = oc.DAY[sl]
    f = pd.DataFrame(dict(day=days, hour=oc.HOUR[sl], dx=dx, code=code, x=x, ph=ph, p=d["p"], qa=float(A["qa"]), date=pd.Timestamp(t))); g = f.groupby("day")["x"]; f["xmean"] = g.transform("mean"); f["xmin"] = g.transform("min"); f["xmax"] = g.transform("max"); f["nsurp"] = (f["ph"] < 10).groupby(f["day"]).transform("sum")
    f["xprev"] = f["x"].shift(1); f["xmax6"] = f["x"].rolling(6, min_periods=1).max().shift(1); f["xmin6"] = f["x"].rolling(6, min_periods=1).min().shift(1); R.append(f)
H = pd.concat(R, ignore_index=True); H["depth"] = np.logaddexp(0, (H["qa"] - H["xmin"]) / 3) * 3; H["dev"] = H["x"] - H["xmean"]; H["above"] = H["x"] - H["xmin"]; H.to_parquet(f"{SP}/out/trough_look2.parquet")
H["err"] = (H["p"] - H["ph"]) / 10
def tab(D, lab):
    print(f"\n{lab}: mean implied shift [GW] (and mean price error actual - model, ct) of usable hours; rows = day's trough depth, columns = hour's index minus the day's lowest index")
    cb = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 80)]; print("   depth \\ above min   " + "  ".join(f"[{a},{b})".center(15) for a, b in cb))
    for a, b in ((0, 0.5), (0.5, 3), (3, 8), (8, 15), (15, 60)):
        k = (D["depth"] >= a) & (D["depth"] < b); row = []
        for c, e in cb:
            kk = k & (D["above"] >= c) & (D["above"] < e); u = kk & (D["code"] == 0); row.append(f"{D['dx'][u].mean():+5.1f} ({D['err'][kk].mean():+.2f}) n{kk.sum():4d}" if kk.sum() >= 25 else " " * 20)
        print(f"   [{a},{b})".ljust(20) + " ".join(row))
pre = H[H["date"] + pd.Timedelta(days=7) <= "2026-03-30"]; post = H[H["date"] >= "2026-03-30"]; tab(pre, "before 2026-03-30"); tab(post, "from 2026-03-30")
for lab, D in (("before 2026-03-30", pre), ("from 2026-03-30", post)):
    print(f"\n{lab}: mean price error (actual - model, ct) by hour of day; trough days (>= 1 hour of model surplus) | other days")
    tr = D["nsurp"] >= 1; print("   trough days: " + " ".join(f"{D['err'][tr & (D['hour'] == k)].mean():+.1f}" for k in range(24))); print("   other days:  " + " ".join(f"{D['err'][~tr & (D['hour'] == k)].mean():+.1f}" for k in range(24)))
    print("   implied shift, trough days: " + " ".join(f"{D['dx'][tr & (D['hour'] == k) & (D['code'] == 0)].mean():+.0f}" for k in range(24))); print("   implied shift, other days:  " + " ".join(f"{D['dx'][~tr & (D['hour'] == k) & (D['code'] == 0)].mean():+.0f}" for k in range(24)))
