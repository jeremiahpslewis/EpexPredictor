"""Parse THE PricesEnergyImbalance XML (regex only) and compare with the BNetzA series already in the frames."""
import re
import sys

import numpy as np
import pandas as pd

xml_path, frame_path, bna_html, out_path = sys.argv[1:5]
txt = open(xml_path, encoding="utf-8-sig").read()
rows = []
for block in re.findall(r"<PricesEnergyImbalance>(.*?)</PricesEnergyImbalance>", txt, flags=re.S):
    rec = dict(re.findall(r"<(\w+)>([^<]*)</\1>", block))
    if "Gasday" in rec:
        rows.append(rec)
df = pd.DataFrame(rows)
df["Gasday"] = pd.to_datetime(df["Gasday"], format="%Y-%m-%d", utc=True)
num = [c for c in df.columns if c not in ("Gasday", "Unit")]
df[num] = df[num].apply(pd.to_numeric, errors="coerce")
df = df.drop_duplicates("Gasday").set_index("Gasday").sort_index()
print("rows", len(df), "| first", df.index.min().date(), "| last", df.index.max().date(), "| units", df["Unit"].unique().tolist())
full = pd.date_range(df.index.min(), df.index.max(), freq="D")
print("missing gas days:", len(full.difference(df.index)), "| NaN VTP average:", int(df["VTPDailyAveragePrice"].isna().sum()),
      "| weekend rows present:", int((df.index.dayofweek >= 5).sum()))
print(df[["VTPDailyAveragePrice", "PositiveEnergyImbalancePrice", "NegativeEnergyImbalancePrice"]].tail(5).round(2).to_string())

# BNetzA series: month-ahead from the frame, day-ahead from the saved page
f = pd.read_parquet(frame_path, columns=["gasprice"])
m1 = f["gasprice"][f.index.hour.isin([0]) & (f.index.minute == 0)]
m1.index = m1.index.normalize()
html = open(bna_html, encoding="utf-8", errors="replace").read()
blk = html[re.search(r"data_myChartId_\w+?_export\s*=", html).start():]
labels = re.findall(r"'(\d\d\.\d\d\.\d{4})'", re.search(r"labels: (\[.*?\])", blk).group(1))
da_raw = re.search(r"label: 'THE Day Ahead'[\w\W]*?data: (\[.*?\])", blk).group(1)
vals = []
for x in da_raw.strip()[1:-1].split(","):
    try:
        vals.append(float(x.strip()))
    except ValueError:
        vals.append(np.nan)
da = pd.Series(vals, index=pd.to_datetime(labels, format="%d.%m.%Y", utc=True)).dropna()
da = da[~da.index.duplicated(keep="last")].sort_index()
vtp = df["VTPDailyAveragePrice"]
j = pd.concat({"the_vtp": vtp, "bna_da": da, "bna_m1": m1}, axis=1).loc["2024-03-01":]
biz = j.dropna()
print("\nbusiness days with all three series:", len(biz))
print("correlation:\n", biz.corr().round(4).to_string())
print("mean abs difference: vtp vs BNetzA day-ahead %.2f, vtp vs month-ahead %.2f (EUR/MWh); mean level %.1f" % (
    (biz.the_vtp - biz.bna_da).abs().mean(), (biz.the_vtp - biz.bna_m1).abs().mean(), biz.the_vtp.mean()))
for sh in (-1, 0, 1):
    b = pd.concat({"vtp": vtp, "da": da.shift(sh, freq="D")}, axis=1).loc["2024-03-01":].dropna()
    print(f"  THE average for gas day D vs BNetzA day-ahead labelled D{-sh:+d}: mean abs diff {(b.vtp - b.da).abs().mean():.2f}, corr of daily changes {np.corrcoef(b.vtp.diff().dropna(), b.da.diff().dropna())[0, 1]:.3f}")
df.to_parquet(out_path)
