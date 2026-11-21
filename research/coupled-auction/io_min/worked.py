"""
Eleven-zone worked inversion in single MTUs (checks W1-W5 of EXPECTATIONS_11ZONE.md).
Usage: worked.py SP MTU_UTC [MTU_UTC ...]      e.g. worked.py $SP 2026-02-01T03:00 2026-02-11T17:00

Zones Z: DE_LU FR NL BE AT CZ PL (Core, modelled) and DK_1 DK_2 SE_4 NO_2 (Nordic, linked to Core by HVDC).
Decision vector y (endogenous network quantities):
  NP_z  AC Core net position of the seven modelled Core zones (JAO hub_<z>)
  a     ALEGrO flow BE->DE (hub_ALBE = -a, hub_ALDE = +a)
  f_l   seven Core-Nordic links (DE-DK1, DE-DK2, DE-SE4, DE-NO2, NL-NO2, NL-DK1, PL-SE4), flow out of the Core zone.
        A link whose virtual-hub PTDF is published sits in the flow-based domain (hub NP_VH = -f); otherwise it has
        directional limits.
Held at published values (outcome conditional on them):
  - net positions of omitted Core hubs (HR HU RO SI SK);
  - flows among the Nordic zones;
  - all schedules with the rest of Europe.
Zone balance: n_z = S_z(p_z) - D_z - u_z - X_z, where n_z is the system-internal net export and X_z the held exports.
"""
import sys, json, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd, requests
import cvxpy as cp

SP = Path(sys.argv[1]); MTUS = [pd.Timestamp(t, tz="UTC") for t in sys.argv[2:]]
JAO = "https://publicationtool.jao.eu/core/api/data"; CACHE = SP / "jao"; CACHE.mkdir(exist_ok=True)
CORE = ["DE", "FR", "NL", "BE", "AT", "CZ", "PL"]; NORD = ["DK_1", "DK_2", "SE_4", "NO_2"]; Z = CORE + NORD
ENT = {"DE": "DE_LU", **{z: z for z in CORE[1:]}, **{z: z for z in NORD}}
OMIT = ["HR", "HU", "RO", "SI", "SK"]
LINKS = [("DE", "DK_1", "DE_DK1_VH"), ("DE", "DK_2", "DE_DK2_BigHub"), ("DE", "SE_4", "DE_SE4_Baltic"), ("DE", "NO_2", "DE_NO2_BigHub"),
         ("NL", "NO_2", "NL_NO2_NorNed"), ("NL", "DK_1", "NL_DK1_COBRA"), ("PL", "SE_4", "PL_SE4_SwePol")]
NORD_INTERNAL = [("DK_1", "NO_2"), ("DK_1", "DK_2"), ("DK_2", "SE_4")]
ROE = {"DE": ["CH"], "FR": ["CH", "IT_NORD", "ES", "GB"], "NL": ["GB"], "BE": ["GB"], "AT": ["CH", "IT_NORD", "SI", "HU"], "CZ": ["SK"],
       "PL": ["SK", "LT"], "DK_1": ["SE_3", "GB"], "DK_2": [], "SE_4": ["LT", "SE_3"], "NO_2": ["NO_1", "NO_5", "GB"]}

def jao(ep, t0, t1, filt=None):
    f = CACHE / f"{ep}_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}{'_ps' if filt else ''}.json"
    if not f.exists():
        q = f"{JAO}/{ep}?FromUtc={t0:%Y-%m-%dT%H:%M}:00.000Z&ToUtc={t1:%Y-%m-%dT%H:%M}:00.000Z&Take=20000"
        if filt: q += "&Filter=" + urllib.parse.quote(json.dumps(filt))
        r = requests.get(q, timeout=180); r.raise_for_status(); f.write_text(r.text)
    return pd.DataFrame(json.loads(f.read_text())["data"])

E11 = SP / "entsoe/eleven"; TAG = "20260201_20260212"
def ent(kind, a, b=None):
    f = E11 / (f"{kind}_{a}" + (f"_{b}" if b else "") + f"_{TAG}.parquet")
    if not f.exists(): return None
    s = pd.read_parquet(f)["v"]; s.index = s.index.tz_convert("UTC"); return s
def sched(a, b, t):          # net scheduled export a -> b at t (MW), None if no data
    s1, s2 = ent("sx", ENT.get(a, a), ENT.get(b, b)), ent("sx", ENT.get(b, b), ENT.get(a, a))
    if s1 is None and s2 is None: return None
    g = lambda s: float(s.asof(t)) if s is not None and len(s) else 0.0
    return g(s1) - g(s2)
def price(z, t):
    if z == "DE": s = pd.read_parquet(E11 / f"price_DE_LU_{TAG}.parquet")["price"]
    else: s = pd.read_parquet(SP / f"entsoe/neighbours/derived/{z}_price_15min.parquet").iloc[:, 0]
    s.index = s.index.tz_convert("UTC"); return float(s.asof(t))

def link_limits(t):          # directional limits for links outside the domain: max scheduled flow, +-14 days (proxy)
    lim = {}
    for a, b, _ in LINKS:
        s1, s2 = ent("sx", ENT[a], ENT[b]), ent("sx", ENT[b], ENT[a])
        lim[(a, b)] = (float(s1.max()) if s1 is not None else 0.0, float(s2.max()) if s2 is not None else 0.0)
    return lim

def logistic_theta():        # placeholder supply curves (any theta; W5 tests the domain, not theta)
    th = {}
    for z in Z: th[z] = [(60.0 if z in CORE else 15.0, 40.0, 0.35)]      # (capacity GW, cost EUR/MWh, relative spread)
    th["DE"] = [(9.0, 25.0, 0.10), (7.0, 60.0, 0.60), (12.0, 95.0, 0.45)]
    th["FR"] = [(70.0, 45.0, 0.6)]; th["NO_2"] = [(12.0, 70.0, 0.6)]
    return th
def S(blocks, p):
    tot = 0.0; d = 0.0
    for K, c, r in blocks:
        s = r * max(abs(c), 10.0); z = (p - c) / s; g = 1 / (1 + np.exp(-z)); tot += K * g; d += K * g * (1 - g) / s
    return tot, d
def Sstar(blocks, p):        # convex conjugate part: integral of S dp (cvxpy expression)
    return sum(K * r * max(abs(c), 10.0) * cp.logistic((p - c) / (r * max(abs(c), 10.0))) for K, c, r in blocks)

out = {}
for t in MTUS:
    R = {"mtu": str(t)}; t1 = t + pd.Timedelta(minutes=15); d0 = t.normalize(); d1 = d0 + pd.Timedelta(days=1)
    fc = jao("finalComputation", t, t1, {"Presolved": True}); fc = fc[fc.dateTimeUtc == f"{t:%Y-%m-%dT%H:%M}:00Z"]
    npj = jao("netPos", d0, d1); npj = npj[npj.dateTimeUtc == f"{t:%Y-%m-%dT%H:%M}:00Z"].iloc[0]
    sp = jao("shadowPrices", d0, d1); sp = sp[(sp.dateTimeUtc == f"{t:%Y-%m-%dT%H:%M}:00Z")]
    hubs = [c[5:] for c in fc.columns if c.startswith("ptdf_")]
    NPH = {h: (float(npj.get(f"hub_{h}")) if pd.notna(npj.get(f"hub_{h}")) else 0.0) for h in hubs}
    active_vh = [h for _, _, h in LINKS if fc[f"ptdf_{h}"].notna().any()]
    R["presolved_cnecs"] = int(len(fc)); R["external_rows_in_presolved"] = int(fc.cneName.astype(str).str.contains("External|Allocation", case=False).sum())
    R["active_virtual_hubs"] = active_vh; R["binding_rows"] = int((sp.shadowPrice.fillna(0) > 1e-6).sum())
    R["binding_names"] = sp.loc[sp.shadowPrice.fillna(0) > 1e-6, ["cnecName", "shadowPrice"]].round(3).values.tolist()
    # ---- W1 domain form
    P = fc[[f"ptdf_{h}" for h in hubs]].astype(float).fillna(0.0).to_numpy(float); ram = fc.ram.astype(float).to_numpy(float)
    flow = P @ np.array([NPH[h] for h in hubs]); viol = flow - ram
    R["W1_max_violation_MW"] = float(viol.max()); R["W1_n_violated_gt5"] = int((viol > 5).sum())
    R["W1_min_slack_MW"] = float((-viol).min())
    # binding presolved CNECs by matching on the CNE and contingency names
    key = lambda df, a, b: (df[a].astype(str).str.strip() + "|" + df[b].fillna("").astype(str).str.strip())
    bset = set(key(sp[sp.shadowPrice.fillna(0) > 1e-6], "cnecName", "contName"))
    fk = key(fc, "cneName", "contName"); isb = fk.isin(bset).to_numpy()
    R["W1_binding_matched"] = int(isb.sum()); R["W1_binding_slack_MW"] = (-viol[isb]).round(2).tolist()
    # ---- prices
    p = {z: price(z, t) for z in Z}; R["prices"] = {z: round(v, 2) for z, v in p.items()}
    # ---- W2 signs: published shadow prices x PTDF differences against observed Core spreads
    spb = sp[sp.shadowPrice.fillna(0) > 1e-6]; mu = spb.shadowPrice.to_numpy(float)
    G = {z: spb[f"hub_{z}"].astype(float).fillna(0.0).to_numpy(float) for z in CORE}
    w2 = {}
    for sgn in (+1, -1):
        err = [p[z] - p["DE"] - sgn * float(mu @ (G[z] - G["DE"])) for z in CORE[1:]]
        w2[sgn] = float(np.max(np.abs(err)))
    R["W2_max_spread_error_by_sign"] = w2; SGN = min(w2, key=w2.get)
    # ---- network matrices in y = (NP_z for CORE, a, f_l for LINKS)
    ny = len(CORE) + 1 + len(LINKS); iy = {("NP", z): i for i, z in enumerate(CORE)}; iy["a"] = len(CORE)
    for k, (a, b, h) in enumerate(LINKS): iy[("f", a, b)] = len(CORE) + 1 + k
    H = np.zeros((len(Z), ny)); zi = {z: i for i, z in enumerate(Z)}
    for z in CORE: H[zi[z], iy[("NP", z)]] = 1.0
    for a, b, h in LINKS:
        j = iy[("f", a, b)]; H[zi[b], j] -= 1.0
        if h not in active_vh: H[zi[a], j] += 1.0          # under AHC the Core-side export is already inside NP_AC
    A = np.zeros((len(fc), ny)); hc = {h: k for k, h in enumerate(hubs)}
    for z in CORE: A[:, iy[("NP", z)]] = P[:, hc[z]]
    A[:, iy["a"]] = P[:, hc["ALDE"]] - P[:, hc["ALBE"]]
    for a, b, h in LINKS:
        if h in active_vh: A[:, iy[("f", a, b)]] = -P[:, hc[h]]
    b_ram = ram - P[:, [hc[o] for o in OMIT]] @ np.array([NPH[o] for o in OMIT])
    e = np.zeros(ny); e[[iy[("NP", z)] for z in CORE]] = 1.0
    for a, b, h in LINKS:
        if h in active_vh: e[iy[("f", a, b)]] = -1.0
    e0 = -sum(NPH[o] for o in OMIT)
    # observed y
    y_obs = np.zeros(ny); y_obs[:len(CORE)] = [NPH[z] for z in CORE]; y_obs[iy["a"]] = NPH["ALDE"]
    for a, b, h in LINKS: y_obs[iy[("f", a, b)]] = sched(a, b, t) or 0.0
    R["domain_check_observed_y"] = dict(max_violation=float((A @ y_obs - b_ram).max()), balance_resid=float(e @ y_obs - e0))
    # ---- W3 accounting: ENTSO-E zone net position = system-internal n_z + held exports X_z (and = sum of all border schedules)
    n_obs = H @ y_obs; w3 = {}
    def held(z):                       # exports held fixed: rest of Europe outside Core, and flows among the Nordic zones
        x = sum((sched(z, b_, t) or 0.0) for b_ in ROE[z] if not (z in CORE and b_ in ("SI", "HU", "SK")))   # omitted Core: inside NP_AC
        x += sum((sched(a_, b_, t) or 0.0) for a_, b_ in NORD_INTERNAL if a_ == z) - sum((sched(a_, b_, t) or 0.0) for a_, b_ in NORD_INTERNAL if b_ == z)
        return x
    for z in Z:
        npe = ent("np", ENT[z]); np_ent = float(npe.asof(t)) if npe is not None else np.nan
        nbrs = set(ROE[z]) | {l[1] for l in LINKS if l[0] == z} | {l[0] for l in LINKS if l[1] == z} | \
               {b_ for a_, b_ in NORD_INTERNAL if a_ == z} | {a_ for a_, b_ in NORD_INTERNAL if b_ == z} | (set(CORE) - {z} if z in CORE else set())
        tot = sum(v for v in (sched(z, b_, t) for b_ in nbrs) if v is not None)
        w3[z] = dict(np_entsoe=round(np_ent, 1), sum_border_schedules=round(tot, 1), n_plus_X=round(float(n_obs[zi[z]] + held(z)), 1),
                     resid_model=round(np_ent - float(n_obs[zi[z]] + held(z)), 1), resid_schedules=round(np_ent - tot, 1))
    R["W3"] = w3
    R["n_obs_MW"] = {z: round(float(n_obs[zi[z]]), 1) for z in Z}; R["X_held_MW"] = {z: round(float(held(z)), 1) for z in Z}
    R["links_obs_MW"] = {f"{a}->{b}": round(float(y_obs[iy[("f", a, b)]]), 1) for a, b, h in LINKS}; R["alegro_obs_MW"] = float(y_obs[iy["a"]])
    lim = link_limits(t)
    # binding set from published data: CNECs with shadow price, ALEGrO at its limit, non-AHC links with a price gap
    # every binding row of the shadow-price publication (CNECs and external constraints) carries its own hub PTDFs
    def to_y(row):
        g = lambda h: float(row.get(f"hub_{h}")) if pd.notna(row.get(f"hub_{h}")) else 0.0
        v = np.zeros(ny)
        for z in CORE: v[iy[("NP", z)]] = g(z)
        v[iy["a"]] = g("ALDE") - g("ALBE")
        for a, b, h in LINKS:
            if h in active_vh: v[iy[("f", a, b)]] = -g(h)
        return v
    extra = [to_y(r) for _, r in spb.iterrows()]
    for a, b, h in LINKS:                                  # links outside the domain: binding where prices differ
        if h not in active_vh and abs(p[a] - p[b]) > 0.01: extra.append(np.eye(ny)[iy[("f", a, b)]])
    AB = np.vstack(extra) if extra else np.zeros((0, ny))
    R["binding_set_size"] = int(AB.shape[0])
    # ---- W4: price subspace V = {p : H^T p in span(A_B^T, e)} and free n directions
    M = np.hstack([H.T, -AB.T, -e[:, None]])               # H^T p - A_B^T mu - e lam = 0
    _, sv, Vt = np.linalg.svd(M); rk = int((sv > 1e-9 * sv[0]).sum()); Nul = Vt[rk:].T
    Vp = Nul[:len(Z)]; U_, s_, _ = np.linalg.svd(Vp, full_matrices=False); Q = U_[:, s_ > 1e-9]
    C = np.vstack([AB, e[None, :]]); _, sc, Vc = np.linalg.svd(C); rc = int((sc > 1e-9 * sc[0]).sum()); Ny = Vc[rc:].T
    Nn = H @ Ny; Un, sn, _ = np.linalg.svd(Nn, full_matrices=False); Nb = Un[:, sn > 1e-9] if len(sn) else np.zeros((len(Z), 0))
    R["W4"] = dict(dim_price_subspace=int(Q.shape[1]), dim_free_n=int(Nb.shape[1]), total=int(Q.shape[1] + Nb.shape[1]),
                   max_abs_QtN=float(np.abs(Q.T @ Nb).max()) if Nb.size else 0.0,
                   observed_price_in_V_resid=float(np.linalg.norm(np.array([p[z] for z in Z]) - Q @ (Q.T @ np.array([p[z] for z in Z])))))
    # Jacobian: theta part det(Q^T diag(S') Q); unit tests of the two limits with random slopes
    rng = np.random.default_rng(0); ds = rng.uniform(0.05, 2, len(Z))
    one = np.ones((len(Z), 1)) / np.sqrt(len(Z))
    R["W4_unit"] = dict(coupled=float(np.linalg.det(one.T @ np.diag(ds) @ one) * len(Z) / ds.sum()),
                        separated=float(np.linalg.det(np.diag(ds)) / np.prod(ds)))
    # ---- W5 re-clear with recovered shocks (placeholder theta), dual form with exact logistic terms
    th = logistic_theta(); Dz = {z: 0.0 for z in Z}           # D enters only through D+u, which inversion pins down
    n_obs = H @ y_obs; Xz = {}
    for z in Z: Xz[z] = 0.0
    load = {z: S(th[z], p[z])[0] * 1000 - n_obs[zi[z]] for z in Z}    # = D + u + X in MW, from the zone balance
    pv = cp.Variable(len(Z)); mu_v = cp.Variable(len(fc), nonneg=True); lam = cp.Variable()
    cons = []; nb_lo, nb_hi = [], []
    lo = np.full(ny, -np.inf); hi = np.full(ny, np.inf); lo[iy["a"]], hi[iy["a"]] = -1000, 1000
    for a, b, h in LINKS:                      # calibration rule: binding link (price gap) -> limit = observed flow
        if h in active_vh: continue
        j = iy[("f", a, b)]; f0 = y_obs[j]
        hi[j], lo[j] = max(lim[(a, b)][0], f0), -max(lim[(a, b)][1], -f0)
        if abs(p[a] - p[b]) > 0.01:
            if p[b] > p[a]: hi[j] = f0                       # flow a->b is at its export limit
            else: lo[j] = f0
    # calibration rule for the network domain (fixed before fitting): rows with a published shadow price are binding in
    # the allocation -> enter with their own PTDFs and RAM = observed flow; presolved rows violated by the published
    # outcome -> RAM raised to the observed flow.
    A0, b0 = A.copy(), b_ram.copy()
    A_sp = np.vstack([to_y(r) for _, r in spb.iterrows()]) if len(spb) else np.zeros((0, ny))
    A_all = np.vstack([A0, A_sp]); flow_obs = A_all @ y_obs; nA = len(A0)
    b_sp_pub = spb.ram.astype(float).to_numpy() - (spb[[f"hub_{o}" for o in OMIT]].astype(float).fillna(0).to_numpy() @ np.array([NPH[o] for o in OMIT]) if len(spb) else 0)
    b_cal = np.r_[np.maximum(b0, flow_obs[:nA]), flow_obs[nA:]]
    R["calibration"] = dict(rows_raised=int((flow_obs[:nA] > b0 + 1e-6).sum()), max_raise_MW=float(np.maximum(flow_obs[:nA] - b0, 0).max()),
                            shadow_rows=int(len(A_sp)), shadow_rows_flow_minus_published_ram_MW=(flow_obs[nA:] - b_sp_pub).round(1).tolist())
    for variant in ("links_held", "links_endogenous", "links_endogenous_uncalibrated"):
        A, b_ram = (A_all, b_cal) if variant != "links_endogenous_uncalibrated" else (A0, b0)
        lo_v, hi_v = lo.copy(), hi.copy()
        if variant == "links_held":
            for a, b, h in LINKS: lo_v[iy[("f", a, b)]] = hi_v[iy[("f", a, b)]] = y_obs[iy[("f", a, b)]]
        y = cp.Variable(ny); pz = cp.Variable(len(Z))
        # primal via the dual of the zone cost: minimise sum_z [Sstar_z(p_z) - (load_z + n_z) p_z] is awkward; solve the
        # saddle directly as a convex primal in (y, p): max_p min_y is replaced by the equivalent dual program below.
        mu = cp.Variable(A.shape[0], nonneg=True); lam = cp.Variable(); nl = cp.Variable(ny, nonneg=True); nh = cp.Variable(ny, nonneg=True)
        fin_lo, fin_hi = np.isfinite(lo_v), np.isfinite(hi_v)
        obj = sum(1000 * Sstar(th[z], pz[zi[z]]) - load[z] * pz[zi[z]] for z in Z) + b_ram @ mu + lam * e0 \
              + cp.sum(cp.multiply(hi_v[fin_hi], nh[fin_hi])) - cp.sum(cp.multiply(lo_v[fin_lo], nl[fin_lo]))
        cons = [H.T @ pz + A.T @ mu + e * lam + nh - nl == 0, nl[~fin_lo] == 0, nh[~fin_hi] == 0]
        pr = cp.Problem(cp.Minimize(obj), cons); pr.solve(solver="CLARABEL")
        yv = -cons[0].dual_value if cons[0].dual_value is not None else np.full(ny, np.nan)
        pz_v = np.asarray(pz.value)
        R[f"W5_{variant}"] = dict(status=pr.status, max_abs_price_err=float(np.max(np.abs(pz_v - np.array([p[z] for z in Z])))),
                                  price_err={z: round(float(pz_v[zi[z]] - p[z]), 3) for z in Z},
                                  max_abs_n_err_MW=float(np.max(np.abs(H @ yv - n_obs))) if np.all(np.isfinite(yv)) else None,
                                  n_err={z: round(float((H @ yv - n_obs)[zi[z]]), 1) for z in Z} if np.all(np.isfinite(yv)) else None)
    out[str(t)] = R
    print(json.dumps(R, indent=1, default=str))
json.dump(out, open(SP / "jao/worked_results.json", "w"), indent=1, default=str)
