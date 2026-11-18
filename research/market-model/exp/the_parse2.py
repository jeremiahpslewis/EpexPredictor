"""Parse THE MarketAreaMonitor / AggregatedConsumptionData XML into daily frames (regex only)."""
import re
import sys

import pandas as pd

def parse(path, tag):
    txt = open(path, encoding="utf-8-sig").read()
    rows = [dict(re.findall(r"<(\w+)>([^<]*)</\1>", b)) for b in re.findall(rf"<{tag}>(.*?)</{tag}>", txt, flags=re.S)]
    df = pd.DataFrame([r for r in rows if "Gasday" in r])
    df["Gasday"] = pd.to_datetime(df["Gasday"].str[:10], format="%Y-%m-%d", utc=True)
    return df.drop_duplicates("Gasday").set_index("Gasday").sort_index()

mon = parse(sys.argv[1], "MarketAreaMonitor")
con = parse(sys.argv[2], "AggregatedConsumptionData")
print("monitor:", mon.shape, mon.index.min().date(), mon.index.max().date(), "units", mon["Unit"].unique().tolist())
q = pd.to_datetime(mon["TimestampDataQueried"], errors="coerce", utc=True)
lag_h = (q - (mon.index + pd.Timedelta(hours=5))).dt.total_seconds() / 3600   # hours after the gas day's ~06:00 local start
print("publication delay after gas-day start (hours): median %.0f, 90th pct %.0f, max %.0f" % (lag_h.median(), lag_h.quantile(0.9), lag_h.max()))
num = [c for c in mon.columns if c not in ("Unit", "TimestampDataQueried")]
mon[num] = mon[num].apply(pd.to_numeric, errors="coerce")
print(mon[num].tail(4).to_string())
print("missing days:", len(pd.date_range(mon.index.min(), mon.index.max(), freq="D").difference(mon.index)))
print("\nconsumption:", con.shape, con.index.min().date(), con.index.max().date(), "status counts:", con["Status"].value_counts().to_dict())
print(con.tail(3).to_string())
mon[num].to_parquet(sys.argv[3])
