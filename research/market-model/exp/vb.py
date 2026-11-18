"""Version B (notes 21.65): M1n supply plus a correction in price / cost index, fitted by two-stage least squares on the day-ahead basis."""
import numpy as np, market1 as M
SP = M.SP; KN = np.array([0.25, 0.6, 0.9, 1.2]); RLO, RHI = -0.2, 2.2
def basis(r): r = np.clip(r, RLO, RHI); return np.column_stack([np.ones_like(r), r] + [np.maximum(r - k, 0.0) for k in KN])
def corr(coef, p, L): return basis(np.asarray(p, float) / np.asarray(L, float)) @ coef
def clear(th, D, L, co2, eps, coef):
    """as market1.clear, with coef (6,) added to supply; coef = None gives version A"""
    D = np.asarray(D, float); L = np.broadcast_to(np.asarray(L, float), D.shape); co2 = np.broadcast_to(np.asarray(co2, float), D.shape); eps = np.broadcast_to(np.asarray(eps, float), D.shape)
    def ex(p, wd): return M.supply(th, p, L, co2) + (0.0 if coef is None else corr(coef, p, L)) + eps - wd - D
    up = ex(np.full(D.shape, M.EPS), 0.0) < 0; dn = ex(np.full(D.shape, -M.EPS), th["K_s"]) > 0; lo = np.where(up, M.EPS, -3000.0); hi = np.where(up, 20000.0, -M.EPS); wd = np.where(dn, th["K_s"], 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); e_ = ex(mid, wd); lo = np.where(e_ < 0, mid, lo); hi = np.where(e_ < 0, hi, mid)
    return np.where(up | dn, 0.5 * (lo + hi), 0.0)
