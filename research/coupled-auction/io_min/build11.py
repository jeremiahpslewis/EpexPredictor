"""
Per-MTU records for the eleven-zone fit and the outage study. Usage: build11.py SP PANEL_DIR OUT.pkl [NFIT]
- fit sample: NFIT MTUs drawn with seed 7 from 2025-10-01 .. 2026-11-08 at quarter-hour resolution, outage MTUs excluded;
- outage sample: prereg/outage_mtus.txt.
Each record holds:
- prices; the observed y; zone net positions n; held exports X; residual loads D (MW);
- German inputs and fuel outputs;
- the network objects: presolved rows and published RAM, shadow rows with their own PTDFs, e, e0, H;
- the regime: AB rows and the price-subspace basis Q;
- link limits;
- checks and flags.
"""
import sys, glob, json, pickle, random
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); import net11 as N

SP, PANEL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); NFIT = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
E = SP / "entsoe/eleven_all"; JB = SP / "jao/bulk"; HERE = Path(__file__).parent
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
NB = SP / "entsoe/neighbours/derived"
def nb(z, s):
    d = pd.read_parquet(NB / f"{z}_{s}_15min.parquet"); d.index = d.index.tz_convert("UTC"); return d

# ---- MTU samples
grid = pd.date_range("2025-10-01 00:00", "2026-11-08 23:45", freq="15min", tz="UTC")
OUTAGE = [pd.Timestamp(t.strip().replace("Z", ""), tz="UTC") for t in open(HERE / "prereg/outage_mtus.txt") if t.strip()]
rest = sorted(set(grid) - set(OUTAGE)); rng = random.Random(7); FIT = sorted(rng.sample(rest, NFIT))
MTUS = [(t, "fit") for t in FIT] + [(t, "outage") for t in OUTAGE]

# ---- series
price = {"DE": ser("price_DE_LU")}
for z in N.Z[1:]: price[z] = nb(z, "price").iloc[:, 0]
load = {"DE": ser("load_DE_LU")}; ws = {"DE": ser("ws_DE_LU").fillna(0).sum(axis=1)}
for z in N.Z[1:]: load[z] = nb(z, "load").iloc[:, 0]; ws[z] = nb(z, "ws").fillna(0).sum(axis=1)
LINKF = {l: sx(l[0], l[1]) for l in N.LINKS}
HELD = {}
for z in N.Z:
    parts = [sx(z, b) for b in N.ROE[z]] + [sx(a, b) for a, b in N.NORD_INTERNAL if a == z] + [-sx(a, b) for a, b in N.NORD_INTERNAL if b == z]
    parts = [p_ for p_ in parts if p_ is not None]
    HELD[z] = sum(p_.reindex(grid).fillna(0) for p_ in parts) if parts else pd.Series(0.0, index=grid)
P = pd.read_parquet(PANEL / "panel.parquet")                                   # hourly German inputs
def outage_cov(a, b):
    fs = sorted(glob.glob(str(E / f"ut_{N.ENT[a]}_{N.ENT[b]}_*.parquet"))) + sorted(glob.glob(str(E / f"ut_{N.ENT[b]}_{N.ENT[a]}_*.parquet")))
    cov = pd.Series(False, index=grid)
    if fs:
        U = pd.concat([pd.read_parquet(f) for f in fs]).reset_index(); U = U[U.docstatus.astype(str) != "Cancelled"]
        for _, r in U.iterrows(): cov |= (grid >= pd.Timestamp(r.start).tz_convert("UTC")) & (grid < pd.Timestamp(r.end).tz_convert("UTC"))
    return cov
COV = {l: outage_cov(l[0], l[1]) for l in N.LINKS}
MISS = pd.read_csv(HERE / "prereg/link_audit.missing.csv", parse_dates=["index"]).set_index("index"); MISS.index = MISS.index.tz_convert("UTC")
WIN = {}                                                                         # +-14 day max flow per direction
for l, f in LINKF.items():
    fr = f.reindex(grid).fillna(0)
    WIN[l] = (fr.clip(lower=0).rolling("28D", center=True).max(), (-fr).clip(lower=0).rolling("28D", center=True).max())

def jao_day(ep, t):
    d0 = pd.Timestamp("2025-09-30", tz="UTC"); k = (t.normalize() - d0).days // 2; f = JB / f"{ep}_{(d0 + pd.Timedelta(days=2 * k)):%Y%m%d}.parquet"
    return pd.read_parquet(f)
_cache = {}
def jao(ep, t):
    key = (ep, (t.normalize() - pd.Timestamp("2025-09-30", tz="UTC")).days // 2)
    if key not in _cache:
        if len(_cache) > 12: _cache.clear()
        _cache[key] = jao_day(ep, t)
    return _cache[key]

recs = []; skipped = {}
for t, kind in MTUS:
    try:
        stamp = f"{t:%Y-%m-%dT%H:%M}:00Z"; hstamp = f"{t.floor('h'):%Y-%m-%dT%H:%M}:00Z"
        fc = jao("finalComputation", t); fc = fc[fc.dateTimeUtc == hstamp]
        npj = jao("netPos", t); npj = npj[npj.dateTimeUtc == stamp]
        spx = jao("shadowPrices", t); spx = spx[(spx.dateTimeUtc == stamp) & (spx.shadowPrice.fillna(0) > 1e-6)]
        if len(fc) == 0 or len(npj) == 0: skipped[str(t)] = "no JAO domain or net positions"; continue
        npr = npj.iloc[0]; hubs = [c[5:] for c in fc.columns if c.startswith("ptdf_")]
        g = lambda h: float(npr.get(f"hub_{h}")) if pd.notna(npr.get(f"hub_{h}")) else 0.0
        active_vh = [h for _, _, h in N.LINKS if f"ptdf_{h}" in fc and fc[f"ptdf_{h}"].notna().any()]
        H = N.build_H(active_vh); e = N.core_balance(active_vh)
        p = np.array([float(price[z].asof(t)) for z in N.Z])
        # observed y
        y = np.zeros(N.NY); y[:N.IA] = [g(z) for z in N.CORE]; y[N.IA] = g("ALDE")
        lflags = {}
        for l in N.LINKS:
            f = float(LINKF[l].asof(t)) if LINKF[l] is not None else 0.0
            y[N.IF[l]], y[N.IF[l] + 1] = max(f, 0.0), max(-f, 0.0)
            # under advanced hybrid coupling the market result for the link is the JAO virtual-hub net position
            # (NP_VH = -f+/(1 - l_ab) + f-); schedules can differ by tens of MW, which would break the Core balance
            if l[2] in active_vh and pd.notna(npr.get(f"hub_{l[2]}")):
                v = g(l[2]); y[N.IF[l]], y[N.IF[l] + 1] = max(-v, 0.0) * (1 - N.LOSS.get((l[0], l[1]), 0.0)), max(v, 0.0)
            lflags[l[2]] = dict(missing=bool(MISS.reindex([t]).fillna(False).get(f"{N.ENT[l[0]]}-{N.ENT[l[1]]}", pd.Series([False])).iloc[0]) if f"{N.ENT[l[0]]}-{N.ENT[l[1]]}" in MISS else False,
                                outage_zero=bool(abs(f) < 0.5 and COV[l].asof(t)), win_out=float(WIN[l][0].asof(t)), win_in=float(WIN[l][1].asof(t)))
        omit_np = np.array([g(o) for o in N.OMIT]); e0 = -omit_np.sum()
        # omitted hubs other than Core zones (e.g. PL_LT_BigHub, RO_BG_VH when active) are held at their published NPs
        extra_hubs = [h for h in hubs if h not in N.CORE + N.OMIT + ["ALBE", "ALDE"] + [l[2] for l in N.LINKS] and pd.notna(npr.get(f"hub_{h}"))]
        e0 -= sum(g(h) for h in extra_hubs)
        Pm = fc[[f"ptdf_{h}" for h in hubs]].astype(float).fillna(0.0).to_numpy(); hc = {h: i for i, h in enumerate(hubs)}
        A = np.array([N.ptdf_to_y(lambda h: Pm[k, hc[h]] if h in hc else 0.0, active_vh) for k in range(len(fc))])
        b_pub = fc.ram.astype(float).to_numpy() - Pm[:, [hc[o] for o in N.OMIT + extra_hubs]] @ np.r_[omit_np, [g(h) for h in extra_hubs]]
        A_sp = np.array([N.ptdf_to_y(lambda h, r=r: float(r.get(f"hub_{h}")) if pd.notna(r.get(f"hub_{h}")) else 0.0, active_vh) for _, r in spx.iterrows()]) if len(spx) else np.zeros((0, N.NY))
        b_sp = (spx.ram.astype(float).to_numpy() - np.array([sum((float(r.get(f"hub_{h}")) if pd.notna(r.get(f"hub_{h}")) else 0.0) * g(h) for h in N.OMIT + extra_hubs) for _, r in spx.iterrows()])) if len(spx) else np.zeros(0)
        # checks against published values: domain form with the PUBLISHED hub net positions (incl. virtual hubs)
        npall = np.array([g(h) for h in hubs]); viol_pub = Pm @ npall - fc.ram.astype(float).to_numpy()
        vh_cons = {h: (g(h), float(-y[N.IF[l]] / (1 - N.LOSS.get((l[0], l[1]), 0.0)) + y[N.IF[l] + 1])) for l in N.LINKS for h in [l[2]] if h in active_vh}
        flow_sp = A_sp @ y if len(A_sp) else np.zeros(0)
        # regime from data: shadow rows, ALEGrO at its limit, link directions at zero, congested link directions
        Eb, bnd = [], []
        if abs(y[N.IA]) >= 999.5: bnd.append((N.IA, "hi" if y[N.IA] > 0 else "lo"))
        for l in N.LINKS:
            a_, b_, h = l; j = N.IF[l]
            for jj, (snd, rcv) in ((j, (a_, b_)), (j + 1, (b_, a_))):
                if y[jj] < 0.5: bnd.append((jj, "lo")); continue
                ell = N.LOSS.get((snd, rcv), 0.0)
                if h not in active_vh and abs(p[N.ZI[snd]] - (1 - ell) * p[N.ZI[rcv]]) > 0.01: bnd.append((jj, "hi"))
        for jj, _ in bnd: v = np.zeros(N.NY); v[jj] = 1.0; Eb.append(v)
        AB = np.vstack([A_sp] + ([np.array(Eb)] if Eb else [])) if (len(A_sp) or Eb) else np.zeros((0, N.NY))
        Q, Nb = N.price_subspace(H, AB, e)
        # unpublished binding constraints: first a targeted search over known constraint types (link capacities,
        # ALEGrO, PL/BE allocation), up to two, that bring the observed prices into V; else the generic residual
        # direction with an inferred y-space constraint (normal H'r, binding at the observed point)
        aug = None; A_inf = np.zeros((0, N.NY)); b_inf = np.zeros(0); added = []
        res = lambda QQ: float(np.linalg.norm(p - QQ @ (QQ.T @ p)))
        if res(Q) > 0.05:
            cand = N.candidate_rows(active_vh, y, set(bnd)); cur = AB
            for _round in range(4):
                best = None
                for name, v in cand:
                    if name in added: continue
                    Qc, _ = N.price_subspace(H, np.vstack([cur, v[None, :]]), e); rc = res(Qc)
                    if best is None or rc < best[2]: best = (name, v, rc, Qc)
                if best is None or best[2] >= res(Q) - 1e-9: break
                added.append(best[0]); cur = np.vstack([cur, best[1][None, :]]); Q = best[3]; A_inf = np.vstack([A_inf, best[1][None, :]])
                if res(Q) <= 0.05: break
            b_inf = A_inf @ y
            if res(Q) > 0.05:
                r_ = p - Q @ (Q.T @ p); Uq, sq_, _ = np.linalg.svd(np.c_[Q, r_ / np.linalg.norm(r_)], full_matrices=False); Q = Uq[:, sq_ > 1e-8]
                c_ = H.T @ (r_ / np.linalg.norm(r_)); A_inf = np.vstack([A_inf, c_[None, :]]); b_inf = A_inf @ y; added.append("generic")
            aug = dict(added=added, zone=N.Z[int(np.argmax(np.abs(p - Q @ (Q.T @ p))))])
        n = H @ y; X = np.array([float(HELD[z].asof(t)) for z in N.Z])
        D = np.array([float(load[z].asof(t)) - float(ws[z].asof(t)) for z in N.Z])
        hr = P.index[P.index.get_indexer([t.floor("h")], method="nearest")[0]]; row = P.loc[hr]
        D[0] -= 1000 * float(row.other + row.psgen + row.pump)                       # German residual load net of other plant and storage
        rec = dict(t=t, kind=kind, p=p, y=y, n=n, X=X, D=D, active_vh=active_vh, H=H, e=e, e0=e0, A=A, b_pub=b_pub, A_sp=A_sp, b_sp=b_sp, aug=aug, A_inf=A_inf, b_inf=b_inf,
                   mu_sp=spx.shadowPrice.astype(float).to_numpy() if len(spx) else np.zeros(0), flow_sp=flow_sp, bnd=bnd, Q=Q, k=Q.shape[1],
                   resid_pV=float(np.linalg.norm(p - Q @ (Q.T @ p))), links=lflags,
                   de=dict(Pg=float(row.Pg), tau=float(row.tau), hdd=float(row.hdd), K_L=float(row.K_lignite), K_C=float(row.K_coal), K_G=float(row.K_gas),
                           lig=float(row.lig), coal=float(row.coal), gasg=float(row.gasg), gap=float(row.gap)),
                   Lc=1.85 * float(row.Pg) + 0.37 * float(row.tau),
                   checks=dict(max_violation_published=float(viol_pub.max()), sp_rows=int(len(A_sp)), sp_flow_minus_ram=(flow_sp - b_sp).round(1).tolist(),
                               vh_consistency=vh_cons, balance_resid=float(e @ y - e0)))
        if not np.all(np.isfinite(np.r_[p, y, D, X])): skipped[str(t)] = "non-finite inputs"; continue
        recs.append(rec)
    except Exception as ex:
        skipped[str(t)] = f"{type(ex).__name__}: {ex}"[:200]
pickle.dump(dict(recs=recs, skipped=skipped), open(OUT, "wb"))
R = pd.DataFrame([dict(kind=r["kind"], k=r["k"], resid=r["resid_pV"], viol=r["checks"]["max_violation_published"], bal=abs(r["checks"]["balance_resid"]),
                       ahc=len(r["active_vh"]) > 0, sp=r["checks"]["sp_rows"],
                       spdev=max([abs(v) for v in r["checks"]["sp_flow_minus_ram"]] or [0]),
                       vh=max([abs(a - b) for a, b in r["checks"]["vh_consistency"].values()] or [0]),
                       miss=any(v["missing"] for v in r["links"].values())) for r in recs])
print("records", len(recs), "skipped", len(skipped), list(skipped.items())[:5])
print(R.groupby("kind").size().to_dict(), "AHC share", round(R.ahc.mean(), 3), "missing-link share", round(R.miss.mean(), 3))
print("price-in-V residual > 0.05:", int((R.resid > 0.05).sum()), " quantiles", R.resid.quantile([.5, .9, .99, 1]).round(4).tolist())
print("published domain excludes baseline (> 5 MW):", int((R.viol > 5).sum()), " Core balance resid max", round(R.bal.max(), 3))
print("shadow rows: flow - published RAM |max| quantiles", R.spdev.quantile([.5, .9, 1]).round(1).tolist())
print("virtual-hub NP consistency |JAO - schedules| quantiles (AHC MTUs)", R[R.ahc].vh.quantile([.5, .9, .99, 1]).round(1).tolist())
print("regimes dim V:", R.k.value_counts().sort_index().to_dict())
G = pd.DataFrame([dict(ahc=len(r["active_vh"]) > 0, aug=r["aug"] is not None, generic="generic" in (r["aug"] or {}).get("added", []), added=",".join((r["aug"] or {}).get("added", []))) for r in recs])
print("augmented share by AHC:", G.groupby("ahc").aug.mean().round(3).to_dict(), " generic share by AHC:", G.groupby("ahc").generic.mean().round(3).to_dict())
from collections import Counter; print("added constraints:", Counter(x for a in G.added for x in a.split(",") if x).most_common(12))
