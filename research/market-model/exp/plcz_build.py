import sys, json, glob, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]; frames = {}
for c in ("pl", "cz"):
    parts = []
    for f in sorted(glob.glob(f"{src}/pp_{c}_*.json")):
        j = json.load(open(f)); t = pd.to_datetime(j["unix_seconds"], unit="s", utc=True); d = {}
        for pt in j["production_types"]:
            n = pt["name"].lower()
            if "brown coal" in n: d["lignite"] = np.array(pt["data"], float)
            elif n == "fossil hard coal": d["hcoal"] = np.array(pt["data"], float)
            elif n == "fossil gas": d["gas"] = np.array(pt["data"], float)
        parts.append(pd.DataFrame(d, index=t).resample("1h").mean())
        if f.endswith("_e.json"): print(c, [pt["name"] for pt in j["production_types"]])
    x = pd.concat(parts).sort_index(); x = x[~x.index.duplicated()] / 1000.0; x.columns = [f"{c}_{k}" for k in x.columns]; frames[c] = x
pd.concat(frames.values(), axis=1).to_parquet(out); print(pd.concat(frames.values(), axis=1).resample("QS").mean().round(1).to_string())
