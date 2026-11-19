"""State core of the supply shift (notes 21.127): day-ahead forecasts from the days before only, and their innovations."""
import numpy as np
HL = 5.0
BYTYPE = True
def typical(V, WE, d, hl=HL, back=40):
    """exponentially weighted mean over the previous days of the same type as day d. V: (days, 96), WE: (days,) bool"""
    idx = [k for k in range(d - 1, max(d - back, -1), -1) if (WE[k] == WE[d] or not BYTYPE) and np.isfinite(V[k]).all()][:15]
    if not idx: idx = [k for k in range(d - 1, max(d - back, -1), -1) if np.isfinite(V[k]).all()][:5]
    w = 0.5 ** (np.arange(len(idx)) / hl); return (w[:, None] * V[idx]).sum(0) / w.sum()
def run(Ud, Xd, WEd, first, xf_last=None):
    """Ud, Xd: (days, 96) shift and demand, last row = target day (its Ud is not used; its demand is xf_last if given). Day-ahead forecasts for days first..last.
    Returns dict of forecasts K0, K1, K2 (days-first, 96) and the fitted lambda, rho (from the days before the target only)."""
    n = len(Ud); Xu = Xd.copy()
    if xf_last is not None: Xu[-1] = xf_last
    k0 = np.full((n, 96), np.nan); dx = np.full((n, 96), np.nan)
    for d in range(first, n): k0[d] = typical(Ud[:d + 1], WEd, d); dx[d] = Xu[d] - typical(Xd[:d + 1], WEd, d)
    i0 = Ud - k0; hist = slice(first, n - 1); a, b = dx[hist].ravel(), i0[hist].ravel(); ok = np.isfinite(a) & np.isfinite(b); lam = float((a[ok] * b[ok]).sum() / (a[ok] ** 2).sum())
    k1 = k0 + lam * dx; i1 = np.nanmean(Ud - k1, 1); y, z = i1[first + 1:n - 1], i1[first:n - 2]; ok = np.isfinite(y) & np.isfinite(z); rho = float((y[ok] * z[ok]).sum() / (z[ok] ** 2).sum())
    k2 = k1.copy(); k2[first + 1:] += rho * i1[first:n - 1, None]
    return dict(K0=k0[first:], K1=k1[first:], K2=k2[first:], lam=lam, rho=rho)

def run3(Ud, Xd, WEd, first, xf_last=None):
    """K3: K1 plus the carry-over of the last hour's and of yesterday's mean innovation, with a coefficient pair per hour of the day.
    Returns forecasts (days-first, 96) for K1 and K3, the K3 innovations of the known days, and the coefficients."""
    R = run(Ud, Xd, WEd, first, xf_last); n = len(Ud); k1 = np.full((n, 96), np.nan); k1[first:] = R["K1"]; i1 = Ud - k1
    end = np.nanmean(i1[:, 92:], 1); mean = np.nanmean(i1, 1); k3 = k1.copy(); C = np.zeros((24, 2))
    tr = np.arange(first + 1, n - 1); A = np.column_stack([end[tr - 1], mean[tr - 1]]); okA = np.isfinite(A).all(1)
    for h in range(24):
        y = np.nanmean(i1[tr][:, 4 * h:4 * h + 4], 1); ok = okA & np.isfinite(y); C[h] = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
    for d in range(first + 1, n):
        if np.isfinite(end[d - 1]) and np.isfinite(mean[d - 1]): k3[d] += np.repeat(C @ np.array([end[d - 1], mean[d - 1]]), 4)
    return dict(K1=k1[first:], K3=k3[first:], lam=R["lam"], C=C)
