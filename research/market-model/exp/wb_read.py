import sys, pandas as pd, numpy as np
path, out = sys.argv[1], sys.argv[2]
x = pd.ExcelFile(path); print(x.sheet_names)
raw = x.parse("Monthly Prices", header=None); hdr = None
for i in range(12):
    row = [str(c) for c in raw.iloc[i].tolist()]
    if any("Coal" in c for c in row): hdr = i; break
names = [str(c) for c in raw.iloc[hdr].tolist()]; units = [str(c) for c in raw.iloc[hdr + 1].tolist()]
cols = {j: n for j, n in enumerate(names) if any(k in n for k in ("Coal", "Natural gas, Europe", "Crude oil, Brent"))}
print({n: units[j] for j, n in cols.items()})
d = raw.iloc[hdr + 2:, [0] + list(cols)].copy(); d.columns = ["m"] + list(cols.values()); d = d[d["m"].astype(str).str.match(r"^\d{4}M\d{2}$")]
d.index = pd.to_datetime(d["m"].str.replace("M", "-") + "-01"); d = d.drop(columns="m").apply(pd.to_numeric, errors="coerce")
print(d.loc["2024-01":].round(1).to_string()); d.to_csv(out)
