"""Hourly nuclear and run-of-river output of Germany's neighbours (and German run-of-river) from the downloaded energy-charts files. Reads JSON only."""
import sys, json, glob, numpy as np, pandas as pd
SP, out = sys.argv[1], sys.argv[2]
def pp(pattern):
    fr = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f)); t = pd.to_datetime(d["unix_seconds"], unit="s", utc=True); fr.append(pd.DataFrame({p["name"]: pd.to_numeric(pd.Series(p["data"]), errors="coerce").to_numpy() for p in d["production_types"]}, index=t).resample("1h").mean())
    if not fr: return None
    M = pd.concat(fr).sort_index(); return M[~M.index.duplicated()] / 1000.0
SRC = {"be": f"{SP}/ext/ecnb/pp_be_*.json", "ch": f"{SP}/ext/ecnb/pp_ch_*.json", "at": f"{SP}/ext/ecnb/pp_at_*.json", "nl": f"{SP}/ext/ecnb/pp_nl_*.json", "dk": f"{SP}/ext/ecnb/pp_dk_*.json",
       "fr": f"{SP}/ext/ecpp/pp_f*.json", "cz": f"{SP}/ext/ecplcz/pp_cz_*.json", "pl": f"{SP}/ext/ecplcz/pp_pl_*.json", "de": f"{SP}/ext/ecpp/pp_[abcd]*.json"}
cols = {}
for z, pat in SRC.items():
    M = pp(pat)
    if M is None: print(z, "no files"); continue
    names = list(M.columns); print(z, M.index.min().date(), "..", M.index.max().date(), "| mean GW:", {n: round(float(M[n].mean()), 1) for n in names if n in ("Nuclear", "Hydro Run-of-River", "Hydro water reservoir", "Fossil gas", "Fossil hard coal", "Fossil brown coal / lignite", "Load", "Residual load")})
    for n, k in (("Nuclear", "nuc"), ("Hydro Run-of-River", "ror"), ("Hydro water reservoir", "res"), ("Fossil gas", "gas"), ("Residual load", "rlreal"), ("Cross border electricity trading", "trade")):
        if n in M: cols[f"{k}_{z}"] = M[n]
pd.DataFrame(cols).to_parquet(out); print("columns:", sorted(cols))
