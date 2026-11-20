"""Rebuild out/market1n_rows.parquet from the saved M1n posterior (no refit); it was overwritten by market1e.py on 20 Nov (notes 21.149)."""
import numpy as np, sys
sys.argv = ["market1n.py", "none"]
import market1n as M
from scipy.special import logit
P_ = np.load(f"{M.SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; kn = np.median(P_["kn"], 0); H = M.H.copy(); FU, QC, CAP, EPS, ZN, TR = M.FU, M.QC, M.CAP, M.EPS, M.ZN, M.TR
cf = M.cost(th, H.L.values, H.co2.values); g0 = M.fuels(th, H.p.values, H.L.values, H.co2.values); gX, gS = M.other_curves(th, H.p.values, H.L.values)
for f in FU: H[f"e_{f}"] = logit(H[QC[f]].values / CAP[f]) - logit(g0[f] / CAP[f]); H[f"d_{f}"] = H[QC[f]].values - g0[f]; H[f"rc_{f}"] = H.p.values / cf[f]
H["eX"] = H.X - gX - ZN @ kn; H["eS"] = H.S - gS; H["eps"] = H.D - H.lig - H.coal - H.gasg - H.X - H.S - th["B_g"] + th["K_s"] * (H.p.values < -EPS) + 0.5 * th["K_s"] * (np.abs(H.p.values) <= EPS); H["p0"] = M.clear(th, H.D.values, H.L.values, H.co2.values, eX=ZN @ kn); H["tr"] = TR; H.to_parquet(f"{M.SP}/out/market1n_rows.parquet")
for lab, m in (("fitting period", TR), ("held-out year", ~TR)):
    g = H[m]; print(f"M1n {lab}: bare price MAE {np.mean(np.abs(g.p0 - g.p)) / 10:.2f} ct; mean output minus curve GW: lignite {g.d_l.mean():+.2f}, coal {g.d_c.mean():+.2f}, gas {g.d_g.mean():+.2f}, imports {g.eX.mean():+.2f}, storage {g.eS.mean():+.2f}, unmeasured {g.eps.mean():+.2f}; sd {g.d_l.std():.2f}, {g.d_c.std():.2f}, {g.d_g.std():.2f}, {g.eX.std():.2f}")
