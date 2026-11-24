"""
Holdout prediction records for the out-of-sample test (prereg/OOS_PROTOCOL.md). Usage:
    predict11_build.py SP PANEL_DIR OUT.pkl START END [STRIDE_MTU]
Builds, for each MTU t of day D in [START, END), only what is known before the D-1 noon gate:
- the Core flow-based domain for t (JAO final computation, published before the gate);
- the published residual-load forecasts for t (ENTSO-E load and wind/solar forecasts);
- fuel and carbon prices and availability (panel row of t's hour; D-1 prices, notices before the gate);
- from the same MTU one day earlier (t - 1 day, whose auction ran on D-2): held exchanges, omitted-hub net positions,
  German other plant / storage output, and the full outcome needed to recover that MTU's shocks (variant b);
- link limits: the largest flow per direction over the seven days before D; link outage flags from transmission
  unavailability notices.
Day-D outcomes (prices, net positions, schedules, shadow prices, flows, fuel outputs) are NOT read into the records.
Scoring targets (hourly German price and SDAC10 net import) are written to OUT.targets.pkl by a separate function at
the end and are read only by the scorer.
"""
import sys, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); import net11 as N

SP, PANEL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
START, END = pd.Timestamp(sys.argv[4], tz="UTC"), pd.Timestamp(sys.argv[5], tz="UTC"); STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 1
E = SP / "entsoe/eleven_all"; JB = SP / "jao/bulk"; NBD = SP / "entsoe/neighbours/derived"; DAY = pd.Timedelta(days=1)
def ser(tag):
    fs = sorted(glob.glob(str(E / f"{tag}_2*.parquet")))
    if not fs: return None
    d = pd.concat([pd.read_parquet(f) for f in fs]); s = d.iloc[:, 0] if d.shape[1] == 1 else d
    s.index = s.index.tz_convert("UTC"); return s[~s.index.duplicated()].sort_index()
def sx(a, b):
    s1, s2 = ser(f"sx_{N.ENT.get(a, a)}_{N.ENT.get(b, b)}"), ser(f"sx_{N.ENT.get(b, b)}_{N.ENT.get(a, a)}")
    if s1 is None and s2 is None: return None
    idx = (s1 if s1 is not None else s2).index
    return (s1.reindex(idx).fillna(0) if s1 is not None else 0) - (s2.reindex(idx).fillna(0) if s2 is not None else 0)
def nbs(z, s):
    d = pd.read_parquet(NBD / f"{z}_{s}_15min.parquet"); d.index = d.index.tz_convert("UTC"); return d
grid = pd.date_range("2025-10-01 00:00", "2026-11-08 23:45", freq="15min", tz="UTC")
price = {"DE": ser("price_DE_LU")} | {z: nbs(z, "price").iloc[:, 0] for z in N.Z[1:]}
load = {"DE": ser("load_DE_LU")} | {z: nbs(z, "load").iloc[:, 0] for z in N.Z[1:]}
ws = {"DE": ser("ws_DE_LU").fillna(0).sum(axis=1)} | {z: nbs(z, "ws").fillna(0).sum(axis=1) for z in N.Z[1:]}
LINKF = {l: sx(l[0], l[1]) for l in N.LINKS}
HELD = {}
for z in N.Z:
    parts = [sx(z, b) for b in N.ROE[z]] + [sx(a, b) for a, b in N.NORD_INTERNAL if a == z] + [-sx(a, b) for a, b in N.NORD_INTERNAL if b == z]
    parts = [p_ for p_ in parts if p_ is not None]
    HELD[z] = sum(p_.reindex(grid).fillna(0) for p_ in parts) if parts else pd.Series(0.0, index=grid)
P = pd.read_parquet(PANEL / "panel.parquet")
def cov(a, b):
    fs = sorted(glob.glob(str(E / f"ut_{N.ENT[a]}_{N.ENT[b]}_*.parquet"))) + sorted(glob.glob(str(E / f"ut_{N.ENT[b]}_{N.ENT[a]}_*.parquet")))
    c = pd.Series(False, index=grid)
    if fs:
        U = pd.concat([pd.read_parquet(f) for f in fs]).reset_index(); U = U[U.docstatus.astype(str) != "Cancelled"]
        for _, r in U.iterrows(): c |= (grid >= pd.Timestamp(r.start).tz_convert("UTC")) & (grid < pd.Timestamp(r.end).tz_convert("UTC"))
    return c
COV = {l: cov(l[0], l[1]) for l in N.LINKS}
_cache = {}
def jao(ep, t):
    d0 = pd.Timestamp("2025-09-30", tz="UTC"); k = (t.normalize() - d0).days // 2
    if (ep, k) not in _cache:
        if len(_cache) > 12: _cache.clear()
        _cache[(ep, k)] = pd.read_parquet(JB / f"{ep}_{(d0 + pd.Timedelta(days=2 * k)):%Y%m%d}.parquet")
    return _cache[(ep, k)]
def prow(t):
    return P.loc[P.index[P.index.get_indexer([t.floor("h")], method="nearest")[0]]]

def outcome_record(t):
    """Everything needed to recover the shocks of a PAST MTU (used only for t - 1 day)."""
    npj = jao("netPos", t); npj = npj[npj.dateTimeUtc == f"{t:%Y-%m-%dT%H:%M}:00Z"]
    if len(npj) == 0: return None
    npr = npj.iloc[0]; g = lambda h: float(npr.get(f"hub_{h}")) if pd.notna(npr.get(f"hub_{h}")) else 0.0
    fc = jao("finalComputation", t); fc = fc[fc.dateTimeUtc == f"{t.floor('h'):%Y-%m-%dT%H:%M}:00Z"]
    active_vh = [h for _, _, h in N.LINKS if f"ptdf_{h}" in fc and fc[f"ptdf_{h}"].notna().any()]
    H = N.build_H(active_vh); y = np.zeros(N.NY); y[:N.IA] = [g(z) for z in N.CORE]; y[N.IA] = g("ALDE")
    for l in N.LINKS:
        f = float(LINKF[l].asof(t)) if LINKF[l] is not None else 0.0
        y[N.IF[l]], y[N.IF[l] + 1] = max(f, 0.0), max(-f, 0.0)
        if l[2] in active_vh and pd.notna(npr.get(f"hub_{l[2]}")):
            v = g(l[2]); y[N.IF[l]], y[N.IF[l] + 1] = max(-v, 0.0) * (1 - N.LOSS.get((l[0], l[1]), 0.0)), max(v, 0.0)
    row = prow(t)
    D = np.array([float(load[z].asof(t)) - float(ws[z].asof(t)) for z in N.Z]); D[0] -= 1000 * float(row.other + row.psgen + row.pump)
    return dict(p=np.array([float(price[z].asof(t)) for z in N.Z]), n=H @ y, X=np.array([float(HELD[z].asof(t)) for z in N.Z]), D=D,
                de=dict(Pg=float(row.Pg), tau=float(row.tau), hdd=float(row.hdd), K_L=float(row.K_lignite), K_C=float(row.K_coal), K_G=float(row.K_gas)),
                Lc=1.85 * float(row.Pg) + 0.37 * float(row.tau), omit=(npr, fc.columns))

recs, skipped = [], {}
ts = [t for t in grid if START <= t < END][::STRIDE]
for t in ts:
    try:
        tl = t - DAY; lag = outcome_record(tl)
        if lag is None: skipped[str(t)] = "no lag outcome"; continue
        fc = jao("finalComputation", t); fc = fc[fc.dateTimeUtc == f"{t.floor('h'):%Y-%m-%dT%H:%M}:00Z"]
        if len(fc) == 0: skipped[str(t)] = "no domain"; continue
        hubs = [c[5:] for c in fc.columns if c.startswith("ptdf_")]
        active_vh = [h for _, _, h in N.LINKS if f"ptdf_{h}" in fc and fc[f"ptdf_{h}"].notna().any()]
        H = N.build_H(active_vh); e = N.core_balance(active_vh)
        npl = lag["omit"][0]; gl = lambda h: float(npl.get(f"hub_{h}")) if pd.notna(npl.get(f"hub_{h}")) else 0.0
        omit_np = np.array([gl(o) for o in N.OMIT])
        extra = [h for h in hubs if h not in N.CORE + N.OMIT + ["ALBE", "ALDE"] + [l[2] for l in N.LINKS] and pd.notna(npl.get(f"hub_{h}"))]
        e0 = -omit_np.sum() - sum(gl(h) for h in extra)
        Pm = fc[[f"ptdf_{h}" for h in hubs]].astype(float).fillna(0.0).to_numpy(); hc = {h: i for i, h in enumerate(hubs)}
        A = np.array([N.ptdf_to_y(lambda h: Pm[k, hc[h]] if h in hc else 0.0, active_vh) for k in range(len(fc))])
        b = fc.ram.astype(float).to_numpy() - Pm[:, [hc[o] for o in N.OMIT + extra]] @ np.r_[omit_np, [gl(h) for h in extra]]
        lo = np.full(N.NY, -np.inf); hi = np.full(N.NY, np.inf); lo[N.IA], hi[N.IA] = -1000.0, 1000.0
        d0 = t.normalize()
        for l in N.LINKS:
            f = LINKF[l].reindex(grid).fillna(0); f = f[(f.index >= d0 - 7 * DAY) & (f.index < d0)]
            for jj, w in ((N.IF[l], f.clip(lower=0).max()), (N.IF[l] + 1, (-f).clip(lower=0).max())):
                lo[jj] = 0.0; hi[jj] = 0.0 if bool(COV[l].asof(t)) else float(w)
        row = prow(t); rowl = prow(tl)
        D = np.array([float(load[z].asof(t)) - float(ws[z].asof(t)) for z in N.Z]); D[0] -= 1000 * float(rowl.other + rowl.psgen + rowl.pump)
        X = lag["X"].copy()
        rec = dict(t=t, H=H, e=e, e0=e0, A=A, b=b, lo=lo, hi=hi, active_vh=active_vh, D=D, X=X,
                   de=dict(Pg=float(row.Pg), tau=float(row.tau), hdd=float(row.hdd), K_L=float(row.K_lignite), K_C=float(row.K_coal), K_G=float(row.K_gas)),
                   Lc=1.85 * float(row.Pg) + 0.37 * float(row.tau), lag={k: v for k, v in lag.items() if k != "omit"})
        if not np.all(np.isfinite(np.r_[D, X, b])): skipped[str(t)] = "non-finite inputs"; continue
        recs.append(rec)
    except Exception as ex:
        skipped[str(t)] = f"{type(ex).__name__}: {ex}"[:200]
pickle.dump(dict(recs=recs, skipped=skipped), open(OUT, "wb"))
print("prediction records", len(recs), "skipped", len(skipped), list(skipped.items())[:3])

def write_targets():
    """Scoring targets, hourly: German price and SDAC10 net import (panel m, GW). Read only by the scorer."""
    hrs = sorted({r["t"].floor("h") for r in recs})
    tg = pd.DataFrame({"p": P.p.reindex(hrs), "m": P.m.reindex(hrs)}); tg.to_pickle(OUT + ".targets.pkl")
    print("targets", len(tg), "hours")
write_targets()
