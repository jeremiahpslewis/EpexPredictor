"""
Link audit and loss factors (PREREG_OUTAGE.md, link convention and addendum). Usage: link_audit.py SP OUT.json
1. Zero-schedule audit per Core-Nordic link: share of MTUs with an exact-zero day-ahead schedule; for NordLink also
   against physical flows and outage records.
2. Loss factor per link, estimated on October-November 2025 only, by the pre-registered estimator:
   median of 1 - p_from/p_to over MTUs with |p| > 20 at both ends, 0 < f < 0.9 * max flow in that direction, and
   |1 - p_from/p_to| < 0.05. Done for each flow direction; links within 0.002 of zero are lossless.
"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
SP = Path(sys.argv[1]); E = SP / "entsoe/eleven_all"
LINKS = [("DE_LU", "DK_1"), ("DE_LU", "DK_2"), ("DE_LU", "SE_4"), ("DE_LU", "NO_2"), ("NL", "NO_2"), ("NL", "DK_1"), ("PL", "SE_4")]
def ser(tag):
    fs = sorted(glob.glob(str(E / f"{tag}_2*.parquet")))
    s = pd.concat([pd.read_parquet(f) for f in fs]).iloc[:, 0]; s.index = s.index.tz_convert("UTC"); return s[~s.index.duplicated()].sort_index()
def price(z):
    if z == "DE_LU": return ser("price_DE_LU")
    s = pd.read_parquet(SP / f"entsoe/neighbours/derived/{z}_price_15min.parquet").iloc[:, 0]; s.index = s.index.tz_convert("UTC"); return s
out = {}
for a, b in LINKS:
    f = (ser(f"sx_{a}_{b}") - ser(f"sx_{b}_{a}")).dropna()
    pa, pb = price(a).reindex(f.index), price(b).reindex(f.index)
    r = {"mtus": int(len(f)), "share_zero_schedule": round(float((f.abs() < 0.5).mean()), 4),
         "share_zero_schedule_with_price_gap_gt1": round(float(((f.abs() < 0.5) & ((pa - pb).abs() > 1)).mean()), 4)}
    win = (f.index >= pd.Timestamp("2025-10-01", tz="UTC")) & (f.index < pd.Timestamp("2025-12-01", tz="UTC"))
    for d, (pf, pt, ff) in {"a_to_b": (pa, pb, f), "b_to_a": (pb, pa, -f)}.items():
        mx = ff[win].max()
        ok = win & (pf.abs() > 20) & (pt.abs() > 20) & (ff > 0) & (ff < 0.9 * mx)
        ratio = 1 - pf[ok] / pt[ok]; ratio = ratio[ratio.abs() < 0.05]
        ell = float(ratio.median()) if len(ratio) else float("nan")
        r[f"loss_{d}"] = dict(n=int(len(ratio)), median=round(ell, 4), iqr=[round(float(ratio.quantile(.25)), 4), round(float(ratio.quantile(.75)), 4)] if len(ratio) else None,
                              lossless=bool(abs(ell) <= 0.002) if len(ratio) else None)
    out[f"{a}-{b}"] = r
# every link against physical flow and outage records (missing-schedule rule of the addendum)
def outages(a, b):          # this link's own transmission-unavailability records, both directions
    fs = sorted(glob.glob(str(E / f"ut_{a}_{b}_*.parquet"))) + sorted(glob.glob(str(E / f"ut_{b}_{a}_*.parquet")))
    if not fs: return pd.DataFrame(columns=["start", "end"])
    U = pd.concat([pd.read_parquet(f) for f in fs]).reset_index()
    U = U[U.docstatus.astype(str) != "Cancelled"]; U["start"] = pd.to_datetime(U.start, utc=True); U["end"] = pd.to_datetime(U.end, utc=True)
    return U
miss_all = []
for a, b in LINKS:
    sx = (ser(f"sx_{a}_{b}") - ser(f"sx_{b}_{a}")).dropna()
    pf = (ser(f"pf_{a}_{b}") - ser(f"pf_{b}_{a}")).resample("15min").ffill().reindex(sx.index)
    cov = pd.Series(False, index=sx.index)
    U = outages(a, b)
    for _, row in U.iterrows(): cov |= (sx.index >= row.start) & (sx.index < row.end)
    out[f"{a}-{b}"]["outage_records"] = int(len(U)); out[f"{a}-{b}"]["share_zero_schedule_under_outage"] = round(float(((sx.abs() < 0.5) & cov).mean()), 4)
    missing = (sx.abs() < 0.5) & (pf.abs() > 100) & ~cov
    zero_phys = (sx.abs() < 0.5) & (pf.abs() <= 100)
    out[f"{a}-{b}"]["audit"] = dict(corr_sched_phys=round(float(sx.corr(pf)), 3), share_missing_by_rule=round(float(missing.mean()), 4),
                                    share_zero_and_phys_zero=round(float(zero_phys.mean()), 4),
                                    missing_by_month={str(k.date()): round(float(v), 3) for k, v in missing.resample("MS").mean().items()})
    miss_all.append(missing[missing].index.to_series().rename(f"{a}-{b}"))
pd.concat(miss_all, axis=1).notna().reset_index().to_csv(Path(sys.argv[2]).with_suffix(".missing.csv"), index=False)
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[2], "w"), indent=1)
