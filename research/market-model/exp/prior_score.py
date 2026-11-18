"""Per arm: parameter paths across the weekly fits (stability), pass-through / elasticity paths, in-sample MAE.  usage: prior_score.py ARM..."""
import sys, os, numpy as np, pandas as pd
import oc, ocxp
from ocx_fits import fit_dates
dates = fit_dates("2025-11-24", "2026-11-09"); X = ocxp.PT_X
for v in sys.argv[1:]:
    R = []
    for t in dates:
        f = f"{oc.SP}/out/ocxw_{v}/{t}.npz"
        if not os.path.exists(f): continue
        A = {k: np.asarray(x, float) for k, x in np.load(f).items()}; pt = ocxp.passthru(A, X, np); el = ocxp.elasticity(A, X, np)
        R.append(dict(fit=t, tstar=float(A["tstar"]), kh=float(A["kh"]), ch=float(A["wn"][6]), be=float(A["wn"][2]), cz=float(A["wn"][5]), ks=float(A["ks"]), qa=float(A["qa"]), qw=float(A["qw"]), qb=float(A["qb"]), c0=float(A["c0"]), pt20=pt[0], pt30=pt[1], pt42=pt[2], pt60=pt[3], el30=el[1], el42=el[2], mae=float(A["mae_in"]) if "mae_in" in A else np.nan))
    T = pd.DataFrame(R).set_index("fit"); st = max(1, len(T) // 8)
    print(f"\narm {v}: {len(T)} weekly fits; every {st}th: " + " ".join(T.index[::st]))
    for k, lab in (("pt20", "pass-through at 20 GW"), ("pt30", "pass-through at 30 GW"), ("pt42", "pass-through at 42.5 GW"), ("el30", "elasticity at 30 GW"), ("tstar", "heat threshold C"), ("kh", "heat slope GW/C"), ("ch", "weight CH"), ("be", "weight BE"), ("ks", "solid availability"), ("qw", "knot qw GW"), ("qb", "knot qb GW"), ("c0", "solid cost c0"), ("mae", "in-sample MAE ct")):
        print(f"   {lab:24s} " + " ".join(f"{x:6.2f}" for x in T[k].values[::st]) + f"   | range {T[k].min():.2f}..{T[k].max():.2f}, largest weekly jump {T[k].diff().abs().max():.2f}")
