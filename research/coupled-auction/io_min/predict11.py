"""
Eleven-zone out-of-sample forecasts (prereg/OOS_PROTOCOL.md). Usage:
    predict11.py PRED_RECS.pkl POST.npz OUT.pkl NPOST NSHOCK SEED
Per MTU and posterior draw (NPOST draws, evenly spaced):
- variant a: NSHOCK shocks u from the fitted multivariate t;
- variant b: the shock recovered from the same MTU one day earlier (record field 'lag').
Clearing: cold-start interior point (solve11.solve_ip) on the published domain for the MTU. If that is infeasible,
retry once with every RAM raised by 10% of |RAM| + 50 MW (flagged 'relaxed'); if still infeasible the MTU is a miss.
Output rows: t, variant, every solve's DE price and German net import from the ten SDAC neighbours (-n_DE, GW), NaN for a
miss; the share of solves that converged; the share that needed the relaxed domain. Hourly aggregation and scoring are done by score_oos.py.
"""
import sys, pickle
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import net11 as N, solve11 as V
RECS, POST, OUT, NPOST, NSHOCK, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
P = dict(np.load(POST, allow_pickle=True)); KEYS = [str(x) for x in P["de_keys"]]; nS = len(P["h_G"])
DRAWS = np.linspace(0, nS - 1, NPOST).astype(int)
def theta(i):
    th = {k: float(P[k][i]) for k in KEYS}
    nb = dict(sbar=P["sbar"][1:], al=P["al"][i], be=P["be"][i], aa=P["aa"][i], bb=P["bb"][i], sg=P["sg"][i])
    L = P["Lcorr"][i]; return th, nb, dict(sd=P["sd_u"][i], L=L, nu=float(P["nu"][i]), mu_de=float(P["mu_de"][i]))
THETAS = [theta(i) for i in DRAWS]

def clear(const, terms, load, r):
    p, y, ok, info = V.solve_ip(const, terms, load, r["H"], r["A"], r["b"], r["e"], r["e0"], r["lo"], r["hi"])
    if ok: return p, y, "published"
    b2 = r["b"] + 0.10 * np.abs(r["b"]) + 50.0
    p, y, ok, info = V.solve_ip(const, terms, load, r["H"], r["A"], b2, r["e"], r["e0"], r["lo"], r["hi"])
    return (p, y, "relaxed") if ok else (None, None, "miss")

def one(args):
    k, r = args; rng = np.random.default_rng(SEED * 1000003 + k); out = []
    for variant in ("a", "b"):
        pd_draws, md_draws, flags = [], [], []      # one entry per solve; NaN = miss (the scorer substitutes the fallback)
        for th, nb, sh in THETAS:
            const, terms = V.curves_logistic(th, nb, r)
            if variant == "a":
                us = []
                for _ in range(NSHOCK):
                    z = rng.standard_normal(N.NZ); w = rng.chisquare(sh["nu"]) / sh["nu"]; u = sh["sd"] * (sh["L"] @ z) / np.sqrt(w); u[0] += sh["mu_de"]; us.append(u)
            else:
                lg = r["lag"]; cl, tl = V.curves_logistic(th, nb, lg); S_l, _ = V.supply_fn(cl, tl)(lg["p"])
                us = [(S_l - lg["D"] - lg["X"] - lg["n"]) / 1000.0]
            for u in us:
                p, y, fl = clear(const, terms, r["D"] + r["X"] + 1000 * u, r)
                flags.append(fl)
                pd_draws.append(p[0] if p is not None else np.nan); md_draws.append(-(r["H"] @ y)[0] / 1000.0 if p is not None else np.nan)
        out.append(dict(t=r["t"], variant=variant, p_draws=pd_draws, m_draws=md_draws,
                        conv=1 - flags.count("miss") / len(flags), relaxed=flags.count("relaxed") / len(flags)))
    return out

if __name__ == "__main__":
    recs = pickle.load(open(RECS, "rb"))["recs"]
    with Pool(4) as pool: rows = [x for o in pool.map(one, list(enumerate(recs)), chunksize=4) for x in o]
    R = pd.DataFrame(rows); R.to_pickle(OUT)
    print(R.groupby("variant")[["conv", "relaxed"]].mean().round(3).to_string())
