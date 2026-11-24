"""
Cold-start equilibrium solver for the eleven-zone market (no observed point or active set needed).

Supply is represented per zone as a constant plus a non-negative sum of logistic terms (GW):
    S_z(p) = c_z + sum_k B_k G((p - m_k) / s_k),   B_k >= 0, s_k > 0
so supply is monotone and its integral is convex: int S_z = c_z p + sum_k B_k s_k softplus((p - m_k) / s_k).
The welfare dual
    min_p,mu>=0,lam,nh>=0,nl>=0  sum_z [int S_z(p_z) - L_z p_z] + b'mu + e0 lam + hi'nh - lo'nl
    s.t.  H'p + A'mu + e lam + nh - nl = 0
is solved by Clarabel (an interior-point method for exponential-cone programs) in GW, so the problem is well scaled.
y comes from the equality's dual. The active set read off the solution starts net11.polish, which makes the
equilibrium exact (KKT to 1e-6).
"""
import numpy as np, cvxpy as cp
import net11 as N

def curves_logistic(th, nb, r, de_key="de"):
    """The fitted curves (cf11.make_supply) in the constant + logistic-sum form. Returns const (NZ,), terms list of
    (zone, B, m, s), all in GW and EUR/MWh."""
    E_L, E_C, E_G, V_G, SMOOTH = 0.364, 0.341, 0.202, 3.0, 0.05
    x = r[de_key]; const = np.zeros(N.NZ); terms = []
    c = {"L": th["a_L"] + th["h_L"] * E_L * x["tau"], "C": th["a_C"] + th["h_C"] * E_C * x["tau"], "G": th["h_G"] * (x["Pg"] + E_G * x["tau"]) + V_G}
    F = {"L": th["F_L"], "C": th["F_C"], "G": th["phi0"] + th["phi1"] * x["hdd"]}
    for f in ("L", "C", "G"):
        K = th[f"alpha_{f}"] * x[f"K_{f}"]; Ff = -SMOOTH * np.logaddexp(-F[f] / SMOOTH, -K / SMOOTH)
        const[0] += Ff; terms.append((0, max(K - Ff, 0.0), c[f], th[f"sigma_{f}"] * c[f]))
    Lc = r["Lc"]
    for i in range(N.NZ - 1):
        sb = nb["sbar"][i]; const[i + 1] += sb * nb["al"][i]
        terms.append((i + 1, sb * nb["be"][i], nb["aa"][i] + nb["bb"][i] * Lc, nb["sg"][i] * Lc))
    return const, terms

def supply_fn(const, terms):
    """S(p), S'(p) in MW for net11.polish, from the same representation."""
    z = np.array([t[0] for t in terms]); B = np.array([t[1] for t in terms]); m = np.array([t[2] for t in terms]); s = np.array([t[3] for t in terms])
    def supply(p):
        g = 1 / (1 + np.exp(-np.clip((p[z] - m) / s, -60, 60)))
        S = const + np.bincount(z, B * g, N.NZ); dS = np.bincount(z, B * g * (1 - g) / s, N.NZ)
        return 1000 * S, 1000 * dS
    return supply

def solve_conic(const, terms, load_mw, H, A, b, e, e0, lo, hi):
    """Interior-point solve of the welfare dual (GW). Returns p (EUR/MWh), y (MW), row multipliers, bound multipliers, status."""
    L = load_mw / 1000.0; bg, e0g = b / 1000.0, e0 / 1000.0
    pz = cp.Variable(N.NZ)
    obj = const @ pz - L @ pz
    for z, B, m, s in terms:
        if B > 1e-9: obj = obj + B * s * cp.logistic((pz[z] - m) / s)
    fh, fl = np.isfinite(hi), np.isfinite(lo)
    mu = cp.Variable(len(b), nonneg=True) if len(b) else None; lam = cp.Variable()
    nh = cp.Variable(int(fh.sum()), nonneg=True); nl = cp.Variable(int(fl.sum()), nonneg=True)
    Eh = np.eye(N.NY)[:, fh]; El = np.eye(N.NY)[:, fl]
    stat = H.T @ pz + e * lam + Eh @ nh - El @ nl
    if mu is not None: stat = stat + A.T @ mu; obj = obj + bg @ mu
    obj = obj + lam * e0g + (hi[fh] / 1000.0) @ nh - (lo[fl] / 1000.0) @ nl
    con = [stat == 0]
    pr = cp.Problem(cp.Minimize(obj), con)
    try: pr.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    except Exception: return None
    if pz.value is None: return None
    y = -1000.0 * np.asarray(con[0].dual_value)
    mu_v = np.asarray(mu.value) if mu is not None else np.zeros(0)
    nhv = np.zeros(N.NY); nhv[fh] = nh.value; nlv = np.zeros(N.NY); nlv[fl] = nl.value
    return np.asarray(pz.value), y, mu_v, nhv, nlv, pr.status

def kkt_check(const, terms, load_mw, H, A, b, e, e0, lo, hi, p, y):
    """Residuals of the equilibrium at (p, y) in MW: zone balance, flow-based rows, Core balance, bounds."""
    S, _ = supply_fn(const, terms)(p)
    bal = np.abs(S - load_mw - H @ y).max()
    feas = max((A @ y - b).max(initial=0.0), abs(e @ y - e0), np.nanmax(np.r_[y - hi, lo - y, 0.0]))
    return float(bal), float(max(feas, 0.0))

def solve_ip(const, terms, load_mw, H, A, b, e, e0, lo, hi, tol_mw=5.0):
    """Interior-point solve accepted on its own equilibrium residuals (balance and feasibility within tol_mw)."""
    c = solve_conic(const, terms, load_mw, H, A, b, e, e0, lo, hi)
    if c is None: return None, None, False, dict(stage="conic failed")
    p, y, mu, nhv, nlv, status = c
    bal, feas = kkt_check(const, terms, load_mw, H, A, b, e, e0, lo, hi, p, y)
    return p, y, bool(bal < tol_mw and feas < tol_mw), dict(stage="ip", status=status, bal=bal, feas=feas)

def solve(const, terms, load_mw, H, A, b, e, e0, lo, hi, deadline=None, tol_act=1e-4):
    """Cold start: conic solve, active set from its multipliers and slacks, exact polish. Returns p, y, ok, info."""
    c = solve_conic(const, terms, load_mw, H, A, b, e, e0, lo, hi)
    if c is None: return None, None, False, dict(stage="conic failed")
    p0, y0, mu, nhv, nlv, status = c
    slack = b - A @ y0 if len(b) else np.zeros(0)
    act0 = [k for k in range(len(b)) if mu[k] > tol_act or slack[k] < 0.5]
    bnd0 = [(j, "hi") for j in range(N.NY) if np.isfinite(hi[j]) and (nhv[j] > tol_act or hi[j] - y0[j] < 0.5)] + \
           [(j, "lo") for j in range(N.NY) if np.isfinite(lo[j]) and (nlv[j] > tol_act or y0[j] - lo[j] < 0.5)]
    fixed = {j for j in range(N.NY) if np.isfinite(lo[j]) and lo[j] == hi[j]}
    bnd0 = [(j, s) for j, s in bnd0 if j not in fixed]
    p, y, act, bnd, ok = N.polish(supply_fn(const, terms), load_mw, H, A, b, e, e0, lo, hi, p0, y0, act0, bnd0, deadline=deadline)
    return p, y, ok, dict(stage="polish", conic_status=status, conic_p=p0, n_act=len(act))
