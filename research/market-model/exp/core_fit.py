"""The structural core of the supply shift (notes 21.126): supply-side terms in GW from the calendar and the demand, fitted on a context."""
import numpy as np
def design(x, slot0, weekend, xbar_ref=None):
    """x: demand per quarter hour (whole days, starting at slot-of-day slot0 = 0); weekend: flag per quarter hour. Returns the term matrices."""
    n = len(x); d = n // 96; xd = x[:d * 96].reshape(d, 24, 4); xh = xd.mean(2); xl = xh.mean(1)
    hod = np.tile(np.repeat(np.arange(24), 4), d); we = weekend[:d * 96].astype(int)
    H = np.zeros((d * 96, 48)); H[np.arange(d * 96), hod + 24 * we] = 1                       # C1: hour of day, working day / weekend
    q = (xd - xh[:, :, None]).ravel()                                                           # C2: within-hour ramp
    p = np.repeat((xh - xl[:, None]).ravel(), 4); ref = xl.mean() if xbar_ref is None else xbar_ref; lv = np.repeat(xl - ref, 96)   # C3: daily swing, level
    return H, q, p, lv, ref
def fit(u, x, weekend, levels=3):
    H, q, p, lv, ref = design(x, 0, weekend); cols = [H] + ([q[:, None]] if levels >= 2 else []) + ([p[:, None], lv[:, None]] if levels >= 3 else []); A = np.hstack(cols)
    ok = np.isfinite(u) & np.isfinite(A).all(1); co = np.linalg.lstsq(A[ok], u[ok], rcond=None)[0]; return dict(co=co, ref=ref, levels=levels)
def apply(m, x, weekend):
    H, q, p, lv, _ = design(x, 0, weekend, m["ref"]); cols = [H] + ([q[:, None]] if m["levels"] >= 2 else []) + ([p[:, None], lv[:, None]] if m["levels"] >= 3 else []); return np.hstack(cols) @ m["co"]
