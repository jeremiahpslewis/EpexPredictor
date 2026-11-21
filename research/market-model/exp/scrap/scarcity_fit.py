"""
Bounded import block: fit c and w on FIT quarter hours by least squares on observed net imports (x_sum), with the per-day limits from the
trailing 28 days plugged in and the control's zero-price step kept. As a diagnostic, the control's linear form (level, slope) is refitted
the same way, to see how far a standalone fit lands from the Bayesian values 5.42 / 15.11. Writes $SP/out/scarcity_fit.json.
"""
import json

import numpy as np
from scipy.optimize import least_squares

import scarcity_common as C

f = C.load()
g = f[(f.period == "FIT") & f.xlo.notna() & f.x.notna()]
p, L, x, lo, hi = (g[k].to_numpy() for k in ("p", "L", "x", "xlo", "xhi"))

res_b = least_squares(lambda t: C.imports_bounded(p, L, lo, hi, t[0], abs(t[1])) - x, x0=[1.0, 0.5])
c, w = float(res_b.x[0]), float(abs(res_b.x[1]))
res_c = least_squares(lambda t: t[0] + t[1] * (p / L - 1.0) - C.step(p) - x, x0=[5.0, 15.0])

def r2(pred):
    return 1 - np.mean((x - pred) ** 2) / np.var(x)

pb = C.imports_bounded(p, L, lo, hi, c, w)
pc = C.imports_control(p, L)
pr = res_c.x[0] + res_c.x[1] * (p / L - 1.0) - C.step(p)
ratio = p / L
print(f"FIT quarter hours used: {len(g)}  (p/L quantiles 5/50/95/99: {np.round(np.quantile(ratio, [.05, .5, .95, .99]), 2)})")
print(f"bounded: c = {c:.3f}, w = {w:.3f}; R2 on net imports {r2(pb):.3f}, rmse {np.sqrt(np.mean((x - pb) ** 2)):.2f} GW, mean err {np.mean(x - pb):+.2f}")
print(f"control as is (5.42, 15.11): R2 {r2(pc):.3f}, rmse {np.sqrt(np.mean((x - pc) ** 2)):.2f}, mean err {np.mean(x - pc):+.2f}")
print(f"control form refitted standalone: level {res_c.x[0]:.2f}, slope {res_c.x[1]:.2f}; R2 {r2(pr):.3f}")
# slope of the import term per unit p/L at typical points, GW
for q in (0.5, 1.0, 1.3, 1.6, 2.0):
    sb = (hi.mean() - lo.mean()) * np.exp(-0.5 * ((q - c) / w) ** 2) / (w * np.sqrt(2 * np.pi))
    print(f"  p/L = {q}: bounded slope {sb:5.1f} GW per unit p/L (control 15.11); bounded level {lo.mean() + (hi.mean() - lo.mean()) * C.ndtr((q - c) / w):+.1f} GW, control {C.P['import_level'] + C.P['import_slope'] * (q - 1):+.1f}")
# binned fit check
bins = np.quantile(ratio, np.linspace(0, 1, 11))
k = np.clip(np.digitize(ratio, bins[1:-1]), 0, 9)
print("decile of p/L: mean p/L, observed x, bounded, control")
for j in range(10):
    m = k == j
    print(f"  {ratio[m].mean():5.2f} {x[m].mean():+6.2f} {pb[m].mean():+6.2f} {pc[m].mean():+6.2f}")
json.dump(dict(c=c, w=w, q=list(C.LIMIT_Q), r2_bounded=r2(pb), r2_control=r2(pc), control_refit=list(map(float, res_c.x))),
          open(f"{C.SP}/out/scarcity_fit.json", "w"), indent=1)
