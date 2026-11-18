import sys, glob, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]; rows = []
for f in sorted(glob.glob(src + "/*202[123]*")):
    x = pd.ExcelFile(f)
    for sh in x.sheet_names[:1]:
        raw = x.parse(sh, header=None); hdr = None
        for i in range(12):
            r = [str(c) for c in raw.iloc[i].tolist()]
            if any("Auction Price" in c for c in r) and any(c.strip() == "Date" for c in r): hdr = i; break
        if hdr is None: print(f, "no header", [str(c) for c in raw.iloc[3].tolist()][:8]); continue
        names = [str(c).strip() for c in raw.iloc[hdr].tolist()]; d = raw.iloc[hdr + 1:].copy(); d.columns = names
        pc = [c for c in names if c.startswith("Auction Price")][0]; cc = [c for c in names if c in ("Contract", "Auction Name")]
        d = d[pd.to_numeric(d[pc], errors="coerce").notna()]
        if "Contract" in names: print("   contracts:", d["Contract"].astype(str).value_counts().to_dict()); d = d[~d["Contract"].astype(str).str.contains("EAA|EUAA", regex=True)]
        rows.append(pd.DataFrame({"date": pd.to_datetime(d["Date"], errors="coerce"), "eua": pd.to_numeric(d[pc], errors="coerce")})); print(f.split("/")[-1], len(d), pc, cc)
E = pd.concat(rows).dropna().groupby("date")["eua"].mean().sort_index(); print(E.resample("QS").mean().round(1).to_string()); E.to_frame().to_parquet(out)
