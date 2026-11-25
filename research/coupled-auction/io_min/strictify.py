"""
Strictly pre-gate eleven-zone prediction records (prereg/ENSEMBLE_PROTOCOL.md addendum 1). Usage:
    strictify.py SP PANEL_DIR PRED_RECS.pkl OUT.pkl
From the holdout prediction records, replaces:
- Germany's residual load: yesterday's published value (the lag record's D) plus the change in the deployed
  lead-weighted residual-load forecast (SP/out/gen_lwA_autumn.npz rl_pes, GW) from its forecast for D-1 to its
  forecast for D, at the same clock time;
- each neighbour's residual load: today's ENTSO-E load forecast minus yesterday's published wind and solar forecast;
- heating degree days: from D-1.
MTUs whose day has no deployed forecast in the stored autumn set (2026-11-08) are dropped and counted.
"""
import sys, pickle
import numpy as np, pandas as pd
SP, PANEL, RECS, OUT = sys.argv[1:5]
sys.argv = ["predict11_build.py", SP, PANEL, "/dev/null", "2025-10-01", "2025-10-02"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import predict11_build as B, net11 as N
G = np.load(f"{SP}/out/gen_lwA_autumn.npz", allow_pickle=True); cut = pd.DatetimeIndex(G["cutoffs"]).tz_localize("UTC")
rl = {}                                                       # deployed forecast issued at cutoff j, by target time
for j, c in enumerate(cut):
    for k in range(G["rl_pes"].shape[1]): rl.setdefault(j, {})[c + pd.Timedelta(minutes=15 * k)] = float(G["rl_pes"][j, k])
day_j = {c.tz_convert("Europe/Berlin").date(): j for j, c in enumerate(cut)}
DAY = pd.Timedelta(days=1); recs = pickle.load(open(RECS, "rb"))["recs"]; out, drop = [], 0
for r in recs:
    t = r["t"]; j = day_j.get(t.tz_convert("Europe/Berlin").date())
    if j is None or j == 0 or t not in rl.get(j, {}) or (t - DAY) not in rl.get(j - 1, {}): drop += 1; continue
    rr = dict(r); D = np.array(r["D"], float)
    D[0] = r["lag"]["D"][0] + 1000 * (rl[j][t] - rl[j - 1][t - DAY])
    for i, z in enumerate(N.Z[1:], start=1):
        D[i] = float(B.load[z].asof(t)) - float(B.ws[z].asof(t - DAY))
    rr["D"] = D; rr["D_a"] = D.copy()
    de = dict(r["de"]); de["hdd"] = float(B.prow(t - DAY).hdd); rr["de"] = de
    out.append(rr)
pickle.dump(dict(recs=out, skipped={}), open(OUT, "wb"))
d0 = np.array([r["D"][0] for r in out]); dorig = np.array([r["D"][0] for r in recs if r["t"] in {x["t"] for x in out}])
print(f"strict records {len(out)}, dropped {drop}; German residual load change vs original, MW: median {np.median(d0 - dorig):+.0f}, mean |diff| {np.mean(np.abs(d0 - dorig)):.0f}")
