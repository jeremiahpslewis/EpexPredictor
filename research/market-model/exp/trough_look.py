"""Is the seasonal level evidence explained by a day-level driver known in advance? Per out-of-sample day (scored by the latest weekly fit before it):
daily evidence (median implied index shift, all usable hours / night) against the day's index profile: minimum, mean, range, hours in model surplus."""
import sys, numpy as np, pandas as pd
import oc, ocx2
from ocx_state import *
dates = [t for t in fit_dates()]; rows_ = []
for t in dates:
    A = get_fit(t); h = hx(t); sl = slice(h, min(h + OOS * 24, oc.NH)); n = sl.stop - sl.start
    if n < 24: continue
    d = rows2(sl, drl=np.zeros(n)); dx, code = implied(A, d); dd = {k: np.nan_to_num(v) for k, v in d.items()}; x = index2(A, dd, np); ph = pcurve(A, x, dd["gas"], dd["eua"], np)[0]; days = oc.DAY[sl]; hours = oc.HOUR[sl]
    for dy in np.unique(days):
        k = days == dy
        if k.sum() < 23: continue
        v = dx[k]; hv = hours[k]; xx = x[k]; rl = d["rl"][k, 0]
        rows_.append(dict(fit=t, day=dy, date=pd.Timestamp(t) + pd.Timedelta(days=int(dy - oc.DAY[h])), z=np.nanmedian(v) if np.isfinite(v).sum() >= 12 else np.nan, zn=np.nanmedian(v[hv < 8]) if np.isfinite(v[hv < 8]).sum() >= 4 else np.nan,
                          ze=np.nanmedian(v[hv >= 17]) if np.isfinite(v[hv >= 17]).sum() >= 4 else np.nan, xmin=xx.min(), xmean=xx.mean(), xmax=xx.max(), xnight=xx[hv < 8].mean(), rlmin=np.nanmin(rl), rlmean=np.nanmean(rl), nlow=(code[k] == 2).sum(), nsurp=(ph[k] < 10).sum(),
                          qa=float(A["qa"]), qw=float(A["qw"]), teff=np.nanmean(d["teff"][k]), sol=np.nanmean(d["sol"][k]), gasy=np.nanmean(d["gasy"][k]), nuc=np.nanmean(d["nuc"][k]), dt=int(d["dt"][k][0])))
T = pd.DataFrame(rows_); T["depth"] = np.logaddexp(0, (T["qa"] - T["xmin"]) / 3) * 3; T["range"] = T["xmax"] - T["xmin"]; T["nightgap"] = T["xnight"] - T["xmin"]; T.to_parquet(f"{SP}/out/trough_look.parquet")
pre = T[T["date"] < "2026-03-30"]; post = T[T["date"] >= "2026-03-30"]; print(f"{len(T)} out-of-sample days, {len(pre)} before 2026-03-30")
def rep(D, lab):
    print(f"\n{lab}: correlation of the daily evidence with day-level drivers (all hours | night | evening)")
    for c in ("xmin", "depth", "xmean", "range", "nightgap", "nsurp", "nlow", "rlmin", "rlmean", "teff", "sol", "nuc", "gasy"):
        r = [D[[zc, c]].dropna().corr().iloc[0, 1] for zc in ("z", "zn", "ze")]; print(f"  {c:9s} {r[0]:+.2f} | {r[1]:+.2f} | {r[2]:+.2f}")
rep(pre, "before 2026-03-30"); rep(pre[pre["date"] < "2025-11-01"], "  first half (Jun-Oct 2025)"); rep(pre[pre["date"] >= "2025-11-01"], "  second half (Nov 2025-Mar 2026)"); rep(post, "from 2026-03-30 (not used for any choice)")
print("\nmean evidence by trough depth (GW the day's lowest index falls below the surplus point of the curve), before 2026-03-30 | after:")
for a, b in ((0, 0.5), (0.5, 3), (3, 8), (8, 15), (15, 60)):
    k1 = (pre["depth"] >= a) & (pre["depth"] < b); k2 = (post["depth"] >= a) & (post["depth"] < b)
    print(f"  depth [{a},{b}): {k1.sum():3d} days, all hours {pre['z'][k1].mean():+.2f}, night {pre['zn'][k1].mean():+.2f}, evening {pre['ze'][k1].mean():+.2f} | {k2.sum():3d} days, all hours {post['z'][k2].mean():+.2f}, night {post['zn'][k2].mean():+.2f}, evening {post['ze'][k2].mean():+.2f}")
print("\nmean evidence by hours of model surplus in the day (model price < 10 EUR/MWh), before | after:")
for a, b in ((0, 1), (1, 4), (4, 7), (7, 10), (10, 25)):
    k1 = (pre["nsurp"] >= a) & (pre["nsurp"] < b); k2 = (post["nsurp"] >= a) & (post["nsurp"] < b)
    print(f"  hours [{a},{b}): {k1.sum():3d} days, all hours {pre['z'][k1].mean():+.2f}, night {pre['zn'][k1].mean():+.2f} | {k2.sum():3d} days, all hours {post['z'][k2].mean():+.2f}, night {post['zn'][k2].mean():+.2f}")
