"""
Eleven-zone outage study (prereg/PREREG_OUTAGE.md). Usage: cf11.py RECS.pkl OUT_PREFIX NDRAWS POST.npz [POST.npz ...]
Per posterior draw, outage MTU and headroom scenario (replay, eta = 0.10, 0.25, 0.50, 1.00):
  - recover the zonal shocks from the observed outcome;
  - re-clear the baseline and the 1 GW outage (cheapest flexible German gas) exactly (active-set Newton, net11.polish)
    on the same scenario domain;
  - record outage minus baseline: zonal prices, zonal supply responses (= change in zonal net position, since load is
    fixed), German fuel split, German net import.
Domain per scenario:
  - shadow rows and inferred binding rows: RAM = observed flow (equality);
  - presolved rows: replay keeps the published RAM, raised to the observed flow where violated; eta scenarios use
    flow_obs + eta * max(RAM_pub - flow_obs, 0);
  - links: limit 0 under a genuine outage; observed flow in a congested direction; otherwise replay uses
    max(window max, observed) and eta uses observed + eta * (window max - observed);
  - ALEGrO +-1000.
"""
import sys, pickle, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import net11 as N
E_L, E_C, E_G, V_G, SMOOTH = 0.364, 0.341, 0.202, 3.0, 0.05
RECS, OUTP, ND = sys.argv[1], sys.argv[2], int(sys.argv[3]); POSTS = sys.argv[4:]
SCEN = [("replay", None), ("eta0.10", 0.10), ("eta0.25", 0.25), ("eta0.50", 0.50), ("eta1.00", 1.00)]

def sig(z): return 1 / (1 + np.exp(-np.clip(z, -60, 60)))
def de_supply(th, x, p, outage=0.0):
    c = {"L": th["a_L"] + th["h_L"] * E_L * x["tau"], "C": th["a_C"] + th["h_C"] * E_C * x["tau"], "G": th["h_G"] * (x["Pg"] + E_G * x["tau"]) + V_G}
    F = {"L": th["F_L"], "C": th["F_C"], "G": th["phi0"] + th["phi1"] * x["hdd"]}; S, dS = {}, {}
    for f in ("L", "C", "G"):
        K = th[f"alpha_{f}"] * x[f"K_{f}"]; Ff = -SMOOTH * np.logaddexp(-F[f] / SMOOTH, -K / SMOOTH); s = th[f"sigma_{f}"] * c[f]
        g = sig((p - c[f]) / s); S[f] = Ff + (K - Ff) * g; dS[f] = (K - Ff) * g * (1 - g) / s
        if f == "G" and outage > 0:
            cut = min(outage, S[f] - Ff)
            if S[f] - Ff <= outage: dS[f] = 0.0
            S[f] = S[f] - cut
    return S, dS

def make_supply(th, nb, r, outage=0.0):
    Lc = r["Lc"]
    def supply(p):
        S, dS = de_supply(th, r["de"], p[0], outage)
        sc = nb["sg"] * Lc; g = sig((p[1:] - nb["aa"] - nb["bb"] * Lc) / sc)
        Snb = nb["sbar"] * (nb["al"] + nb["be"] * g); dSnb = nb["sbar"] * nb["be"] * g * (1 - g) / sc
        return 1000 * np.r_[S["L"] + S["C"] + S["G"], Snb], 1000 * np.r_[dS["L"] + dS["C"] + dS["G"], dSnb]
    return supply

def domain(r, eta):
    y = r["y"]; flow = r["A"] @ y
    if eta is None: b = np.maximum(r["b_pub"], flow)
    else: b = flow + eta * np.maximum(r["b_pub"] - flow, 0.0)
    # multipliers the observed prices require, by least squares over all binding rows and data-active bounds.
    # Inferred rows are oriented so their multiplier is >= 0. A published row or a bound whose multiplier still has the
    # wrong sign (< -0.05) cannot be explained by the market model (e.g. an adverse DC flow): it is held at its observed
    # value (equality), so the outcome is conditional on it, and it is counted as a flag.
    A_inf, b_inf = r["A_inf"].copy(), r["b_inf"].copy(); held = []
    Eb = [np.eye(N.NY)[j] * (1.0 if sd == "hi" else -1.0) for j, sd in r["bnd"]]
    Rm = np.vstack([r["A_sp"], A_inf] + ([np.array(Eb)] if Eb else []) + [r["e"][None, :], -r["e"][None, :]])
    mu = np.linalg.lstsq(Rm.T, -r["H"].T @ r["p"], rcond=None)[0]
    ns, ni = len(r["A_sp"]), len(A_inf)
    if ni:
        flip = np.where(mu[ns:ns + ni] < 0, -1.0, 1.0); A_inf = A_inf * flip[:, None]; b_inf = b_inf * flip
    eq_rows = [k for k in range(ns) if mu[k] < -0.05]
    held = [r["bnd"][i] for i in range(len(r["bnd"])) if mu[ns + ni + i] < -0.05]
    A = np.vstack([r["A"], r["A_sp"], A_inf] + ([-r["A_sp"][eq_rows]] if eq_rows else []))
    b = np.r_[b, r["A_sp"] @ y if ns else np.zeros(0), b_inf] if True else None
    if eq_rows: b = np.r_[b, -(r["A_sp"][eq_rows] @ y)]
    lo = np.full(N.NY, -np.inf); hi = np.full(N.NY, np.inf); lo[N.IA], hi[N.IA] = -1000.0, 1000.0
    cong = {j for j, s in r["bnd"] if s == "hi"}
    for l in N.LINKS:
        fl = r["links"][l[2]]
        for jj, win in ((N.IF[l], fl["win_out"]), (N.IF[l] + 1, fl["win_in"])):
            lo[jj] = 0.0
            if fl["outage_zero"]: hi[jj] = 0.0
            elif jj in cong: hi[jj] = y[jj]
            else: hi[jj] = max(win, y[jj]) if eta is None else y[jj] + eta * max(win - y[jj], 0.0)
    for j, sd in held: lo[j] = hi[j] = y[j]
    return A, b, lo, hi, dict(eq_rows=len(eq_rows), held_bounds=len(held))

def work(args):
    th, nb, r = args; out = {}
    S0, _ = make_supply(th, nb, r)(r["p"]); load = S0 - r["n"]                   # = D + X + 1000 u: shocks recovered at this draw
    for name, eta in SCEN:
        A, b, lo, hi, dflag = domain(r, eta)
        act0 = list(range(len(r["A"]), len(b)))                                   # binding rows: shadow and inferred
        bnd0 = [(j, s) for j, s in r["bnd"]] + [(j, "lo") for j in range(N.IA + 1, N.NY) if r["y"][j] < 0.5 and (j, "lo") not in r["bnd"]]
        if abs(r["y"][N.IA]) >= 999.5: bnd0.append((N.IA, "hi" if r["y"][N.IA] > 0 else "lo"))
        pb, yb, ab, bb, okb = N.polish(make_supply(th, nb, r), load, r["H"], A, b, r["e"], r["e0"], lo, hi, r["p"], r["y"], act0, list(set(bnd0)))
        po, yo, ao, bo, oko = N.polish(make_supply(th, nb, r, 1.0), load, r["H"], A, b, r["e"], r["e0"], lo, hi, pb, yb, ab, bb)
        Sb, _ = de_supply(th, r["de"], pb[0]); So, _ = de_supply(th, r["de"], po[0], 1.0)
        nbs, nos = r["H"] @ yb, r["H"] @ yo
        out[name] = dict(ok=bool(okb and oko), base_err_p=float(np.abs(pb - r["p"]).max()), base_err_n=float(np.abs(nbs - r["n"]).max()),
                         dp=(po - pb).tolist(), dS=((nos - nbs) / 1000).tolist(), dL=So["L"] - Sb["L"], dC=So["C"] - Sb["C"], dG=So["G"] - Sb["G"],
                         dimport=float(-(nos[0] - nbs[0]) / 1000), cap=bool(po.max() > 4000), **dflag)
    return out

if __name__ == "__main__":
    d = [r for r in pickle.load(open(RECS, "rb"))["recs"] if r["kind"] == "outage"]
    jobs = []
    for post in POSTS:
        P = dict(np.load(post, allow_pickle=True)); nS = len(P["h_G"]); pick = np.linspace(0, nS - 1, ND).astype(int)
        for i in pick:
            th = {k: float(P[k][i]) for k in [str(x) for x in P["de_keys"]]}
            nb = dict(sbar=P["sbar"][1:], al=P["al"][i], be=P["be"][i], aa=P["aa"][i], bb=P["bb"][i], sg=P["sg"][i])
            for r in d: jobs.append((post, int(i), th, nb, r))
    t0 = time.time()
    with Pool(4) as pool: res = pool.map(work, [(th, nb, r) for _, _, th, nb, r in jobs], chunksize=8)
    rows = [dict(post=post, draw=i, t=str(r["t"]), flags=dict(pub_excl=bool((r["A"] @ r["y"] - r["b_pub"]).max() > 5),
                 generic="generic" in (r["aug"] or {}).get("added", []), ahc=len(r["active_vh"]) > 0,
                 link_missing=any(v["missing"] for v in r["links"].values())), res=o) for (post, i, th, nb, r), o in zip(jobs, res)]
    pickle.dump(rows, open(OUTP + ".pkl", "wb")); print(f"{len(rows)} draw-MTU results in {time.time()-t0:.0f}s")
