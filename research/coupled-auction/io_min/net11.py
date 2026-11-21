"""
Eleven-zone network objects shared by build11.py, fit11.py and cf11.py.

y (ny = 22):
  NP_z    AC Core net positions of DE FR NL BE AT CZ PL (JAO hub_<z>);
  a       ALEGrO flow BE->DE (hub_ALBE = -a, hub_ALDE = +a; inside NP_BE/NP_DE, so not in the zone balance);
  f+_l, f-_l  for seven Core-Nordic links, both >= 0. f+ goes from the Core zone to the Nordic zone. The sender
          exports f; the receiver gets (1 - loss) f. Under advanced hybrid coupling (virtual-hub PTDF published),
          the Core-side leg is inside NP_core and the virtual hub has NP_VH = -f+ + (1 - loss_ba) f-.
Zone balance: n = H y (MW, net export inside the system); S_z(p_z) - D_z - u_z - X_z = n_z.
Equilibrium (welfare optimum): S(p) - load = H y,  H'p + A_act' mu + e lam + bound multipliers = 0, active rows tight.
"""
import numpy as np

CORE = ["DE", "FR", "NL", "BE", "AT", "CZ", "PL"]; NORD = ["DK_1", "DK_2", "SE_4", "NO_2"]; Z = CORE + NORD
ENT = {"DE": "DE_LU", **{z: z for z in Z if z != "DE"}}
OMIT = ["HR", "HU", "RO", "SI", "SK"]
LINKS = [("DE", "DK_1", "DE_DK1_VH"), ("DE", "DK_2", "DE_DK2_BigHub"), ("DE", "SE_4", "DE_SE4_Baltic"), ("DE", "NO_2", "DE_NO2_BigHub"),
         ("NL", "NO_2", "NL_NO2_NorNed"), ("NL", "DK_1", "NL_DK1_COBRA"), ("PL", "SE_4", "PL_SE4_SwePol")]
# loss factors fixed in prereg/PREREG_OUTAGE.md addendum (sender -> receiver)
LOSS = {("DE", "SE_4"): 0.025, ("SE_4", "DE"): 0.025, ("DE", "NO_2"): 0.031, ("NO_2", "DE"): 0.031,
        ("NL", "NO_2"): 0.032, ("NO_2", "NL"): 0.032, ("SE_4", "PL"): 0.011}
NORD_INTERNAL = [("DK_1", "NO_2"), ("DK_1", "DK_2"), ("DK_2", "SE_4")]
ROE = {"DE": ["CH"], "FR": ["CH", "IT_NORD", "ES", "GB"], "NL": ["GB"], "BE": ["GB"], "AT": ["CH", "IT_NORD"], "CZ": [], "PL": ["LT"],
       "DK_1": ["SE_3", "GB"], "DK_2": [], "SE_4": ["LT", "SE_3"], "NO_2": ["NO_1", "NO_5", "GB"]}
NZ = len(Z); ZI = {z: i for i, z in enumerate(Z)}
IA = len(CORE); IF = {l: IA + 1 + 2 * k for k, l in enumerate(LINKS)}; NY = IA + 1 + 2 * len(LINKS)

# Link quantities are RECEIVED quantities (ENTSO-E schedules and JAO virtual-hub net positions agree on this:
# DE->NO2 schedule 1399 MW = received in NO2, JAO NP_VH = -1444 = -1399/(1 - 0.031) sent from the Core side).
# f+ : received by the Nordic zone b, sent by a as f+/(1 - l_ab);  f- : received by a, sent by b as f-/(1 - l_ba).
def build_H(active_vh):
    H = np.zeros((NZ, NY))
    for i, z in enumerate(CORE): H[ZI[z], i] = 1.0
    for l in LINKS:
        a, b, h = l; j = IF[l]; lab, lba = LOSS.get((a, b), 0.0), LOSS.get((b, a), 0.0)
        H[ZI[b], j] = -1.0; H[ZI[b], j + 1] = 1.0 / (1 - lba)
        if h not in active_vh: H[ZI[a], j] = 1.0 / (1 - lab); H[ZI[a], j + 1] = -1.0
    return H

def ptdf_to_y(row_get, active_vh):
    """Map a row of hub PTDFs (callable hub -> value) to a row over y."""
    v = np.zeros(NY)
    for i, z in enumerate(CORE): v[i] = row_get(z)
    v[IA] = row_get("ALDE") - row_get("ALBE")
    for l in LINKS:
        a, b, h = l
        if h in active_vh: v[IF[l]] = -row_get(h) / (1 - LOSS.get((a, b), 0.0)); v[IF[l] + 1] = row_get(h)      # NP_VH = -f+/(1-l) + f-
    return v

def core_balance(active_vh):
    e = np.zeros(NY); e[:IA] = 1.0
    for l in LINKS:
        if l[2] in active_vh: e[IF[l]] = -1.0 / (1 - LOSS.get((l[0], l[1]), 0.0)); e[IF[l] + 1] = 1.0
    return e

def price_subspace(H, AB, e, tol=1e-4):
    """Orthonormal basis Q of V = {p : H'p in span(rows of AB, e)} and its complement (free net-position directions)."""
    Rm = np.vstack([AB, e[None, :]]).T if len(AB) else e[:, None]
    Ur, sr, _ = np.linalg.svd(Rm, full_matrices=True); rr = int((sr > 1e-7 * sr[0]).sum())
    Pp = np.eye(H.shape[1]) - Ur[:, :rr] @ Ur[:, :rr].T
    _, sq, Vq = np.linalg.svd(Pp @ H.T); sq = np.r_[sq, np.zeros(NZ - len(sq))]
    Q = Vq[sq < tol * max(sq.max(), 1e-12)].T
    Uq, _, _ = np.linalg.svd(Q, full_matrices=True)
    return Q, Uq[:, Q.shape[1]:]

def polish(supply, load, H, A, b, e, e0, lo, hi, p0, y0, act0=None, bnd0=None, iters=40, outer=80):
    """Exact equilibrium by active-set Newton. supply(p) -> (S, dS) in MW. Starts from (p0, y0) and an optional active
    set; adds the most violated row or bound, drops the most negative multiplier, until KKT holds.
    Returns p, y, active rows, active bounds (index, side), ok."""
    nz, ny = H.shape
    act = set(np.flatnonzero(b - A @ y0 < 1.0)) if act0 is None else set(act0)
    bnd = {(j, "hi" if hi[j] - y0[j] < y0[j] - lo[j] else "lo") for j in range(ny) if np.isfinite(hi[j]) and (hi[j] - y0[j] < 1.0 or y0[j] - lo[j] < 1.0)} if bnd0 is None else set(bnd0)
    p, y = p0.copy(), y0.copy()
    for _o in range(outer):
        ar = sorted(act); br = sorted(bnd); na, nb = len(ar), len(br)
        Eb = np.zeros((nb, ny)); bv = np.zeros(nb)
        for i, (j, side) in enumerate(br): Eb[i, j] = 1.0; bv[i] = hi[j] if side == "hi" else lo[j]
        x = np.r_[p, y, np.zeros(na), 0.0, np.zeros(nb)]
        def resid(x):
            pp, yy = x[:nz], x[nz:nz + ny]; mu, lam, nu = x[nz + ny:nz + ny + na], x[nz + ny + na], x[nz + ny + na + 1:]
            S, dS = supply(pp)
            return np.r_[S - load - H @ yy, H.T @ pp + A[ar].T @ mu + e * lam + Eb.T @ nu, A[ar] @ yy - b[ar], e @ yy - e0, Eb @ yy - bv], dS
        for _i in range(iters):
            F, dS = resid(x)
            if np.max(np.abs(F)) < 1e-6: break
            J = np.zeros((len(F), len(x)))
            J[:nz, :nz] = np.diag(dS); J[:nz, nz:nz + ny] = -H
            J[nz:nz + ny, :nz] = H.T; J[nz:nz + ny, nz + ny:nz + ny + na] = A[ar].T; J[nz:nz + ny, nz + ny + na] = e; J[nz:nz + ny, nz + ny + na + 1:] = Eb.T
            r2 = nz + ny; J[r2:r2 + na, nz:nz + ny] = A[ar]; J[r2 + na, nz:nz + ny] = e; J[r2 + na + 1:, nz:nz + ny] = Eb
            dx = np.linalg.lstsq(J, -F, rcond=None)[0]
            m = np.max(np.abs(dx[:nz])); dx = dx * min(1.0, 50.0 / m) if m > 0 else dx
            f0 = np.linalg.norm(F); t = 1.0
            for _ls in range(30):
                if np.linalg.norm(resid(x + t * dx)[0]) < (1 - 1e-4 * t) * f0: break
                t *= 0.5
            x = x + t * dx
        F, _ = resid(x); pp, yy = x[:nz], x[nz:nz + ny]; mu = x[nz + ny:nz + ny + na]; nu = x[nz + ny + na + 1:]
        slack = b - A @ yy
        viol = [k for k in range(len(b)) if k not in act and slack[k] < -1e-4]
        bviol = [(j, "hi") for j in range(ny) if np.isfinite(hi[j]) and (j, "hi") not in bnd and yy[j] > hi[j] + 1e-4] + \
                [(j, "lo") for j in range(ny) if np.isfinite(lo[j]) and (j, "lo") not in bnd and yy[j] < lo[j] - 1e-4]
        neg = [ar[i] for i in range(na) if mu[i] < -1e-6]
        # bound multipliers, same convention as the rows (H'p + A'mu + ... + Eb'nu = 0): y_j <= hi enters like a row,
        # so nu >= 0; y_j >= lo is -y_j <= -lo, so nu <= 0
        bneg = [br[i] for i in range(nb) if (br[i][1] == "hi" and nu[i] < -1e-6) or (br[i][1] == "lo" and nu[i] > 1e-6)]
        if not viol and not bviol and not neg and not bneg and np.max(np.abs(F)) < 1e-3:
            return pp, yy, ar, br, True
        if viol: act.add(min(viol, key=lambda k: slack[k]))
        elif bviol: bnd.add(bviol[0])
        elif neg: act.discard(min(neg, key=lambda k: mu[ar.index(k)]))
        elif bneg: bnd.discard(bneg[0])
        p, y = pp, yy
    return p, y, sorted(act), sorted(bnd), False

def candidate_rows(active_vh, y, lo_bounds_done):
    """Known constraint types that may bind without being published: each link direction's capacity (as a bound row),
    ALEGrO, and the PL and BE allocation constraints (on the zone's AC Core net position)."""
    C = []
    for l in LINKS:
        for jj, name in ((IF[l], f"{l[0]}->{l[1]} cap"), (IF[l] + 1, f"{l[1]}->{l[0]} cap")):
            if (jj, "lo") in lo_bounds_done or y[jj] < 0.5: continue
            v = np.zeros(NY); v[jj] = 1.0; C.append((name, v))
    v = np.zeros(NY); v[IA] = 1.0; C.append(("ALEGrO", v))
    for z in ("PL", "BE"): v = np.zeros(NY); v[CORE.index(z)] = 1.0; C.append((f"{z} allocation", v))
    return C
