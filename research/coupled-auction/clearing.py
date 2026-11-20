"""
A small coupled day-ahead clearing in the style of EUPHEMIA (public description, 18 Dec 2025), for checking a model
specification on toy markets - not a replica of the production algorithm.

Covered: several bidding zones and periods; stepwise period orders (fully accepted in the money, rejected out of the
money, any fraction at the money); block orders over several periods with one acceptance ratio, a minimum acceptance
ratio, never accepted out of the money (no paradoxically accepted blocks) but possibly rejected in the money
(paradoxically rejected blocks allowed); ATC lines with a capacity per direction and period; harmonised price bounds.
Not covered: flow-based constraints, losses, tariffs, ramping, complex and MIC orders, linked blocks, exclusive groups,
mixed time resolutions, the curtailment and volume-indeterminacy rules.

Method, as in the public description: the master problem maximises economic surplus (consumer surplus + producer
surplus + congestion rent) over the block decisions (a mixed-integer program); the price determination sub-problem
looks for zonal prices consistent with every order and line given that selection; if there are none, the selection is
cut off (a no-good cut) and the master problem is solved again.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

TOL = 1e-6


@dataclass
class Order:
    zone: str
    t: int
    side: str        # "buy" or "sell"
    q: float         # MW
    p: float         # limit price, EUR/MWh


@dataclass
class Block:
    name: str
    zone: str
    side: str
    p: float                 # limit price for the block as a whole, EUR/MWh
    q: list                  # MW per period (0 where the block is absent)
    mar: float = 1.0         # minimum acceptance ratio (1 = fill-or-kill)


@dataclass
class Line:
    frm: str
    to: str
    cap_fwd: list            # MW per period, frm -> to
    cap_bwd: list            # MW per period, to -> frm


@dataclass
class Market:
    zones: list
    periods: int
    orders: list
    blocks: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    pmin: float = -500.0
    pmax: float = 4000.0
    duration: float = 1.0    # hours per period: surplus is in EUR, quantities in MW, so each period counts MW x price x hours


@dataclass
class Result:
    welfare: float
    x: np.ndarray            # acceptance per period order
    a: np.ndarray            # acceptance ratio per block
    f: np.ndarray            # flow per line and period (frm -> to positive)
    prices: np.ndarray       # zone x period
    rounds: int              # master-problem solves
    cuts: list               # block selections cut off, with the reason
    low: np.ndarray = None   # lowest and highest valid price per zone and period, given the accepted orders
    high: np.ndarray = None


def _layout(m: Market):
    no, nb, nl, T = len(m.orders), len(m.blocks), len(m.lines), m.periods
    # variables: x (orders), a (block ratios), y (block on/off), f (flows)
    return no, nb, nl, T, no + 2 * nb + nl * T


def _balance(m: Market):
    """rows: zone x period; supply - demand - exports = 0"""
    no, nb, nl, T, n = _layout(m)
    Z = {z: i for i, z in enumerate(m.zones)}
    A = np.zeros((len(m.zones) * T, n))
    for j, o in enumerate(m.orders):
        A[Z[o.zone] * T + o.t, j] += o.q if o.side == "sell" else -o.q
    for k, b in enumerate(m.blocks):
        for t in range(T):
            A[Z[b.zone] * T + t, no + k] += b.q[t] if b.side == "sell" else -b.q[t]
    for l, ln in enumerate(m.lines):
        for t in range(T):
            col = no + 2 * nb + l * T + t
            A[Z[ln.frm] * T + t, col] -= 1.0
            A[Z[ln.to] * T + t, col] += 1.0
    return A


def _objective(m: Market):
    """coefficients of economic surplus (to maximise)"""
    no, nb, nl, T, n = _layout(m)
    c = np.zeros(n)
    for j, o in enumerate(m.orders):
        c[j] = o.q * o.p if o.side == "buy" else -o.q * o.p
    for k, b in enumerate(m.blocks):
        c[no + k] = (1 if b.side == "buy" else -1) * b.p * sum(b.q)
    return c * m.duration


def _bounds(m: Market, fix_a=None):
    no, nb, nl, T, n = _layout(m)
    lo, hi = np.zeros(n), np.ones(n)
    for l, ln in enumerate(m.lines):
        for t in range(T):
            lo[no + 2 * nb + l * T + t] = -ln.cap_bwd[t]
            hi[no + 2 * nb + l * T + t] = ln.cap_fwd[t]
    if fix_a is not None:
        lo[no:no + nb] = hi[no:no + nb] = fix_a
        lo[no + nb:no + 2 * nb] = hi[no + nb:no + 2 * nb] = (np.asarray(fix_a) > TOL).astype(float)
    return lo, hi


def solve_master(m: Market, cuts=()):
    """maximise economic surplus over block decisions; cuts = block on/off selections to exclude"""
    no, nb, nl, T, n = _layout(m)
    c = _objective(m)
    cons = [LinearConstraint(_balance(m), 0.0, 0.0)]
    if nb:
        # mar * y <= a <= y
        G = np.zeros((2 * nb, n))
        for k, b in enumerate(m.blocks):
            G[2 * k, no + k], G[2 * k, no + nb + k] = 1.0, -1.0          # a - y <= 0
            G[2 * k + 1, no + k], G[2 * k + 1, no + nb + k] = -1.0, b.mar  # mar*y - a <= 0
        cons.append(LinearConstraint(G, -np.inf, 0.0))
        for sel in cuts:   # no-good cut: sum over on (1 - y) + sum over off y >= 1
            row = np.zeros(n)
            for k, on in enumerate(sel):
                row[no + nb + k] = -1.0 if on else 1.0
            cons.append(LinearConstraint(row, 1.0 - sum(sel), np.inf))
    integrality = np.zeros(n)
    integrality[no + nb:no + 2 * nb] = 1
    lo, hi = _bounds(m)
    # the default relative gap (about 1e-4) is far too loose here: price-taking demand at the price cap makes total
    # surplus huge, so a gap of 1e-4 hides block decisions worth thousands of EUR
    r = milp(-c, constraints=cons, integrality=integrality, bounds=Bounds(lo, hi), options={"mip_rel_gap": 0.0})
    if not r.success:
        return None
    return r.x


def surplus_lp(m: Market, a, tie_break=False):
    """
    economic surplus maximised with the block ratios fixed; returns x, f, welfare and the balance duals.
    tie_break: among surplus-maximising solutions, the one with the least total absolute flow. Not used by clear():
    it broke price determination (notes 21.153); volumes are settled after the prices by volumes_at_prices instead,
    which is a volume-indeterminacy rule, not EUPHEMIA's flow indeterminacy (notes 21.155).
    """
    no, nb, nl, T, n = _layout(m)
    lo, hi = _bounds(m, fix_a=a)
    c, A = _objective(m), _balance(m)
    r = linprog(-c, A_eq=A, b_eq=np.zeros(len(m.zones) * T), bounds=list(zip(lo, hi)), method="highs")
    if not r.success:
        return None
    w, duals, x = -r.fun, r.eqlin.marginals.reshape(len(m.zones), T), r.x
    if tie_break and nl:
        nf = nl * T; f0 = no + 2 * nb
        # variables [x (n), g (nf)]; minimise sum g, g >= |f|, surplus >= w - tolerance
        A_eq2 = np.hstack([A, np.zeros((A.shape[0], nf))])
        G1 = np.zeros((nf, n + nf)); G2 = np.zeros((nf, n + nf))
        for i in range(nf): G1[i, f0 + i], G1[i, n + i] = 1.0, -1.0; G2[i, f0 + i], G2[i, n + i] = -1.0, -1.0
        S = np.r_[-c, np.zeros(nf)][None, :]
        r2 = linprog(np.r_[np.zeros(n), np.ones(nf)], A_ub=np.vstack([G1, G2, S]), b_ub=np.r_[np.zeros(2 * nf), -w + 1e-7 * max(1.0, abs(w))],
                     A_eq=A_eq2, b_eq=np.zeros(A.shape[0]), bounds=list(zip(lo, hi)) + [(0, None)] * nf, method="highs")
        if r2.success:
            x = r2.x[:n]
    return x, w, duals


def price_determination(m: Market, x, a, f, target, only=None):
    """
    zonal prices consistent with the given acceptance: period orders and lines by their state, accepted blocks not out
    of the money, prices within bounds. Among valid prices, the ones closest (L1) to `target` (the LP duals) - a choice of
    this toy: EUPHEMIA's own price-indeterminacy rules are not reproduced; Result.low / high give the whole interval.
    Returns prices (zone x period) or None if no valid prices exist.
    """
    no, nb, nl, T, n = _layout(m)
    Z = {z: i for i, z in enumerate(m.zones)}
    nP = len(m.zones) * T
    # variables: prices (nP), deviations d+ d- (nP each)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    pid = lambda z, t: Z[z] * T + t

    def row(entries, nvar=3 * nP):
        r = np.zeros(nvar)
        for i, v in entries:
            r[i] += v
        return r

    for j, o in enumerate(m.orders):
        if o.q <= 0:          # an empty order says nothing about the price
            continue
        i = pid(o.zone, o.t)
        full, none = x[j] >= 1 - TOL, x[j] <= TOL
        buy = o.side == "buy"
        if full:      # accepted: buy p <= limit; sell p >= limit
            A_ub.append(row([(i, 1.0 if buy else -1.0)])); b_ub.append(o.p if buy else -o.p)
        elif none:    # rejected: buy p >= limit; sell p <= limit
            A_ub.append(row([(i, -1.0 if buy else 1.0)])); b_ub.append(-o.p if buy else o.p)
        else:         # partly accepted: at the money
            A_eq.append(row([(i, 1.0)])); b_eq.append(o.p)
    for k, b in enumerate(m.blocks):
        if a[k] > TOL:   # accepted: not out of the money (no PAB)
            sgn = 1.0 if b.side == "sell" else -1.0
            # sell: sum q_t (p_t - limit) >= 0  ->  -sum q_t p_t <= -limit * sum q
            A_ub.append(row([(pid(b.zone, t), -sgn * b.q[t]) for t in range(T)])); b_ub.append(-sgn * b.p * sum(b.q))
    for l, ln in enumerate(m.lines):
        for t in range(T):
            v = f[no + 2 * nb + l * T + t]
            pf, pt = pid(ln.frm, t), pid(ln.to, t)
            at_fwd, at_bwd = v >= ln.cap_fwd[t] - TOL, v <= -ln.cap_bwd[t] + TOL
            if at_fwd and at_bwd:   # zero capacity both ways: prices independent
                continue
            if at_fwd:     # congested frm -> to: p_to >= p_frm
                A_ub.append(row([(pf, 1.0), (pt, -1.0)])); b_ub.append(0.0)
            elif at_bwd:   # congested to -> frm: p_frm >= p_to
                A_ub.append(row([(pt, 1.0), (pf, -1.0)])); b_ub.append(0.0)
            else:          # not congested: one price
                A_eq.append(row([(pf, 1.0), (pt, -1.0)])); b_eq.append(0.0)
    # |p - target| linearised: p - target = d+ - d-
    for i in range(nP):   # the duals of the surplus problem are prices times the period duration
        A_eq.append(row([(i, 1.0), (nP + i, -1.0), (2 * nP + i, 1.0)])); b_eq.append(float(np.clip(target.flat[i] / m.duration, m.pmin, m.pmax)))
    c = np.r_[np.zeros(nP), np.ones(2 * nP)]
    if only is not None:   # (index, +1 / -1): minimise / maximise one price instead, to find the interval of valid prices
        c = np.zeros(3 * nP); c[only[0]] = only[1]
    bounds = [(m.pmin, m.pmax)] * nP + [(0, None)] * (2 * nP)
    r = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None, A_eq=np.array(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    if not r.success:
        return None
    return r.x[:nP].reshape(len(m.zones), T)


def volumes_at_prices(m: Market, a, prices, tol=1e-6, rule="least_flow"):
    """
    Volume indeterminacy, after the prices: with the prices fixed, in-the-money orders are fully accepted, out-of-the-
    money ones rejected, at-the-money ones free. Any such solution maximises surplus; `rule` picks one:
      "least_flow"  the least total absolute line flow (the default since notes 21.153)
      "max_ab"      the most flow from each line's `frm` zone to its `to` zone (one extreme of the admissible set)
      "max_ba"      the most flow the other way (the other extreme)
    With one line between two zones the flow IS the net position of `frm`, so these rules choose accepted volumes of
    at-the-money orders, not routes: this is EUPHEMIA's volume indeterminacy (section 7.9), not its flow indeterminacy,
    which only arises in meshed networks. EUPHEMIA's public description maximises accepted volume and then fixes flows
    by cost coefficients; the remaining split of at-the-money volume between zones is not published.
    Returns x, or None.
    """
    no, nb, nl, T, n = _layout(m)
    Z = {z: i for i, z in enumerate(m.zones)}
    lo, hi = _bounds(m, fix_a=a)
    for j, o in enumerate(m.orders):
        p = prices[Z[o.zone], o.t]
        if (o.side == "buy" and o.p > p + tol) or (o.side == "sell" and o.p < p - tol): lo[j] = hi[j] = 1.0
        elif (o.side == "buy" and o.p < p - tol) or (o.side == "sell" and o.p > p + tol): lo[j] = hi[j] = 0.0
    for l, ln in enumerate(m.lines):
        for t in range(T):
            col = no + 2 * nb + l * T + t; d = prices[Z[ln.to], t] - prices[Z[ln.frm], t]
            if d > tol: lo[col] = hi[col] = ln.cap_fwd[t]          # dearer on the receiving side: line full that way
            elif d < -tol: lo[col] = hi[col] = -ln.cap_bwd[t]
    nf = nl * T; f0 = no + 2 * nb
    if nf == 0:
        return None
    A = _balance(m)
    if rule in ("max_ab", "max_ba"):
        cvec = np.zeros(n); cvec[f0:f0 + nf] = -1.0 if rule == "max_ab" else 1.0
        r = linprog(cvec, A_eq=A, b_eq=np.zeros(A.shape[0]), bounds=list(zip(lo, hi)), method="highs")
        return r.x if r.success else None
    A_eq = np.hstack([A, np.zeros((A.shape[0], nf))])
    G = np.zeros((2 * nf, n + nf))
    for i in range(nf): G[i, f0 + i], G[i, n + i] = 1.0, -1.0; G[nf + i, f0 + i], G[nf + i, n + i] = -1.0, -1.0
    r = linprog(np.r_[np.zeros(n), np.ones(nf)], A_ub=G, b_ub=np.zeros(2 * nf), A_eq=A_eq, b_eq=np.zeros(A.shape[0]),
                bounds=list(zip(lo, hi)) + [(0, None)] * nf, method="highs")
    return r.x[:n] if r.success else None


def clear(m: Market, max_rounds=200, intervals=True, volume_rule="least_flow") -> Result:
    """volume_rule: None keeps the surplus problem's own (arbitrary) vertex; otherwise see volumes_at_prices"""
    no, nb, nl, T, n = _layout(m)
    cuts, log = [], []
    for rnd in range(1, max_rounds + 1):
        sol = solve_master(m, cuts)
        if sol is None:
            raise RuntimeError("no feasible block selection")
        y = tuple(int(round(v)) for v in sol[no + nb:no + 2 * nb])
        a = np.where(np.array(y) > 0, np.clip(sol[no:no + nb], 0, 1), 0.0)
        lp = surplus_lp(m, a)
        if lp is None:
            cuts.append(y); log.append((y, "surplus problem infeasible")); continue
        xx, w, duals = lp
        prices = price_determination(m, xx, a, xx, duals)
        if prices is None:
            cuts.append(y); log.append((y, "no valid prices (a block would be paradoxically accepted)")); continue
        vol = volumes_at_prices(m, a, prices, rule=volume_rule) if volume_rule else None
        if vol is not None:
            xx = vol
        nP = len(m.zones) * T
        if not intervals:
            return Result(w, xx[:no], a, xx[no + 2 * nb:].reshape(nl, T) if nl else np.zeros((0, T)), prices, rnd, log)
        low = np.array([price_determination(m, xx, a, xx, duals, only=(i, 1.0)).flat[i] for i in range(nP)]).reshape(len(m.zones), T)
        high = np.array([price_determination(m, xx, a, xx, duals, only=(i, -1.0)).flat[i] for i in range(nP)]).reshape(len(m.zones), T)
        return Result(w, xx[:no], a, xx[no + 2 * nb:].reshape(nl, T) if nl else np.zeros((0, T)), prices, rnd, log, low, high)
    raise RuntimeError("too many rounds")


def brute_force(m: Market):
    """every block selection (fill-or-kill blocks only), the best one with valid prices - for verification"""
    no, nb, nl, T, n = _layout(m)
    best = None
    for y in product([0, 1], repeat=nb):
        a = np.array(y, dtype=float)
        lp = surplus_lp(m, a)
        if lp is None:
            continue
        xx, w, duals = lp
        if price_determination(m, xx, a, xx, duals) is None:
            continue
        if best is None or w > best[0] + 1e-6:
            best = (w, y)
    return best


def audit(m: Market, r: Result, tol=1e-4):
    """market-consistency checks of a result; returns a list of violations (empty = consistent)"""
    T, Z = m.periods, {z: i for i, z in enumerate(m.zones)}
    bad = []
    for j, o in enumerate(m.orders):
        p, xa = r.prices[Z[o.zone], o.t], r.x[j]
        in_money = (o.p > p + tol) if o.side == "buy" else (o.p < p - tol)
        out_money = (o.p < p - tol) if o.side == "buy" else (o.p > p + tol)
        if in_money and xa < 1 - tol: bad.append(f"order {j} in the money but accepted {xa:.3f}")
        if out_money and xa > tol: bad.append(f"order {j} out of the money but accepted {xa:.3f}")
    for k, b in enumerate(m.blocks):
        if r.a[k] > tol:
            s = sum(b.q[t] * (r.prices[Z[b.zone], t] - b.p) for t in range(T)) * (1 if b.side == "sell" else -1)
            if s < -tol: bad.append(f"block {b.name} paradoxically accepted (surplus {s:.2f})")
            if r.a[k] < b.mar - tol: bad.append(f"block {b.name} below its minimum acceptance ratio")
    for l, ln in enumerate(m.lines):
        for t in range(T):
            v, dp = r.f[l, t], r.prices[Z[ln.to], t] - r.prices[Z[ln.frm], t]
            if v > -ln.cap_bwd[t] + tol and v < ln.cap_fwd[t] - tol and abs(dp) > tol: bad.append(f"line {l} period {t}: not congested but prices differ by {dp:.2f}")
            if v > tol and dp < -tol: bad.append(f"line {l} period {t}: flow from the dearer to the cheaper zone")
            if v < -tol and dp > tol: bad.append(f"line {l} period {t}: flow from the dearer to the cheaper zone")
    for z in m.zones:
        for t in range(T):
            net = sum((o.q if o.side == "sell" else -o.q) * r.x[j] for j, o in enumerate(m.orders) if o.zone == z and o.t == t)
            net += sum((b.q[t] if b.side == "sell" else -b.q[t]) * r.a[k] for k, b in enumerate(m.blocks) if b.zone == z)
            exp_ = sum(r.f[l, t] for l, ln in enumerate(m.lines) if ln.frm == z) - sum(r.f[l, t] for l, ln in enumerate(m.lines) if ln.to == z)
            if abs(net - exp_) > 1e-3: bad.append(f"zone {z} period {t} not balanced ({net - exp_:.3f})")
    if np.any(r.prices < m.pmin - tol) or np.any(r.prices > m.pmax + tol): bad.append("price outside the bounds")
    return bad


def surplus_split(m: Market, r: Result):
    """consumer surplus, producer surplus and congestion rent at the cleared prices; their sum is the economic surplus"""
    Z = {z: i for i, z in enumerate(m.zones)}
    cs = ps = 0.0
    for j, o in enumerate(m.orders):
        p = r.prices[Z[o.zone], o.t]
        if o.side == "buy": cs += r.x[j] * o.q * (o.p - p)
        else: ps += r.x[j] * o.q * (p - o.p)
    for k, b in enumerate(m.blocks):
        v = sum(b.q[t] * r.prices[Z[b.zone], t] for t in range(m.periods)) * r.a[k] - b.p * sum(b.q) * r.a[k]
        if b.side == "sell": ps += v
        else: cs -= v
    cr = sum(r.f[l, t] * (r.prices[Z[ln.to], t] - r.prices[Z[ln.frm], t]) for l, ln in enumerate(m.lines) for t in range(m.periods))
    return cs * m.duration, ps * m.duration, cr * m.duration
