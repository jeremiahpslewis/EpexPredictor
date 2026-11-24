"""
Simulate-and-recover for the curve-end parameters (fit11 ENDS=1). Usage:
    ends_sim.py RECS_TRAIN.pkl POST.npz OUT_RECS.pkl NMTU SEED
Truth: the German and neighbour curves and the shock distribution at the posterior medians of POST, plus chosen end
parameters (TRUE_ENDS below). Per MTU of the training records: draw u from the fitted multivariate t, clear the eleven
zones on the replay domain with the cold-start interior-point solver, read the binding set from the solver's
multipliers (as from published shadow prices), rebuild Q, and simulate German fuel outputs from the measurement
equation. The output has the record format fit11.py reads, so the same likelihood code is tested.
"""
import sys, pickle, json
import numpy as np
from multiprocessing import Pool
RECS, POST, OUT, NMTU, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.argv = ["cf11.py", RECS, "/dev/null", "1", POST]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import cf11 as C, net11 as N, solve11 as V
import os
TRUE_ENDS = json.loads(os.environ["ENDS_TRUTH"]) if os.environ.get("ENDS_TRUTH") else \
    dict(m_lo=-15.0, s_lo=8.0, m_hi=220.0, s_hi=40.0, rho_L=0.4, rho_C=0.5, rho_G=0.3, k_lo_de=3.0, k_hi_de=1.5, k_lo_nb=0.10, k_hi_nb=0.08)
P = dict(np.load(POST, allow_pickle=True)); med = lambda k: np.median(P[k], axis=0)
DE_KEYS = [str(x) for x in P["de_keys"]][:15]
TH = {k: float(med(k)) for k in DE_KEYS} | TRUE_ENDS
NB = dict(sbar=P["sbar"][1:], al=med("al"), be=med("be"), aa=med("aa"), bb=med("bb"), sg=med("sg"))
SD, NU, MU_DE = med("sd_u"), float(med("nu")), float(med("mu_de"))
Lc = np.median(P["Lcorr"], axis=0); Lc = np.linalg.cholesky(Lc @ Lc.T)        # median of Cholesky factors, re-orthonormalised
SDQ, BQ, NUQ = med("sd_q"), med("beta_q"), float(med("nu_q"))
E_L, E_C, E_G, V_G, SM = 0.364, 0.341, 0.202, 3.0, 0.05

def fuels(p, x):
    """German fuel outputs at price p under the truth, including floor shedding (GW)."""
    c = {"L": TH["a_L"] + TH["h_L"] * E_L * x["tau"], "C": TH["a_C"] + TH["h_C"] * E_C * x["tau"], "G": TH["h_G"] * (x["Pg"] + E_G * x["tau"]) + V_G}
    F = {"L": TH["F_L"], "C": TH["F_C"], "G": TH["phi0"] + TH["phi1"] * x["hdd"]}; g_lo = 1 / (1 + np.exp(-(p - TH["m_lo"]) / TH["s_lo"])); out = {}
    for f in ("L", "C", "G"):
        K = TH[f"alpha_{f}"] * x[f"K_{f}"]; Ff = -SM * np.logaddexp(-F[f] / SM, -K / SM)
        out[f] = Ff + (K - Ff) / (1 + np.exp(-(p - c[f]) / (TH[f"sigma_{f}"] * c[f]))) - (1 - TH[f"rho_{f}"]) * Ff * (1 - g_lo)
    return out

def one(args):
    k, r = args; rng = np.random.default_rng(SEED * 100003 + k)
    z = rng.standard_normal(N.NZ); w = rng.chisquare(NU) / NU; u = (SD * (Lc @ z)) / np.sqrt(w); u[0] += MU_DE   # GW
    const, terms = V.curves_logistic(TH, NB, r)
    load = np.asarray(r["D"]) + np.asarray(r["X"]) + 1000 * u
    A, b, lo, hi, _ = C.domain(r, None)
    p, y, ok, info = V.solve_ip(const, terms, load, r["H"], A, b, r["e"], r["e0"], lo, hi)
    if not ok or p.max() > 3000 or p.min() < -500: return None
    slack = b - A @ y
    rows = [A[j] for j in range(len(b)) if info["mu"][j] > 1e-6 and slack[j] < 1.0]
    bnds = [np.eye(N.NY)[j] for j in range(N.NY) if (info["nh"][j] > 1e-6 and hi[j] - y[j] < 1.0) or (info["nl"][j] > 1e-6 and y[j] - lo[j] < 1.0)]
    AB = np.vstack(rows + bnds) if (rows or bnds) else np.zeros((0, N.NY))
    Q, _ = N.price_subspace(r["H"], AB, r["e"])
    q = fuels(p[0], r["de"]); de = dict(r["de"])
    for i, (f, col) in enumerate((("L", "lig"), ("C", "coal"), ("G", "gasg"))):
        de[col] = float(q[f] + BQ[i] * de["gap"] + SDQ[i] * rng.standard_t(NUQ))
    rr = dict(r); rr.update(p=p, y=y, n=r["H"] @ y, Q=Q, k=Q.shape[1], de=de, sim_u=u)
    return rr

if __name__ == "__main__":
    d = [r for r in pickle.load(open(RECS, "rb"))["recs"] if r["kind"] == "fit" and not any(v["missing"] for v in r["links"].values())]
    d = d[:: max(1, len(d) // NMTU)][:NMTU]
    with Pool(4) as pool: out = pool.map(one, list(enumerate(d)))
    recs = [r for r in out if r is not None]
    p = np.array([r["p"][0] for r in recs])
    print(f"simulated {len(recs)} of {len(d)}; DE price quantiles", np.round(np.quantile(p, [0, .05, .5, .95, 1]), 1).tolist(),
          f"share p<=0 {np.mean(p <= 0):.3f}, p>200 {np.mean(p > 200):.3f}; dim V", np.bincount([r["k"] for r in recs]).tolist())
    pickle.dump(dict(recs=recs, truth=TH, sbar=np.r_[1.0, NB["sbar"]]), open(OUT, "wb")); json.dump(TRUE_ENDS, open(OUT + ".truth.json", "w"))
