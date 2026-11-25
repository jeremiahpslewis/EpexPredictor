"""
Post-hoc holdout scoring of variant (c) (prereg/OOS_PROTOCOL.md addendum 4; the holdout was already scored once).
Usage: holdout_c.py SP PANEL_DIR POST.npz COEF.npz PRED_RECS.pkl OUT.pkl
For each holdout MTU, with posterior draw 0 (as used to estimate the coefficients): recover the shocks at t-1d, t-2d,
t-7d (outcomes known before the gate), forecast u with the AR coefficients fitted on all training MTUs, and clear on
the pre-gate domain (same relaxation fallback as predict11). Also clears variant (b) with draw 0, so the pair is
compared on equal terms. Output in the predict11 format (variants 'b0' and 'c'), one draw per MTU.
"""
import sys, pickle, os
import numpy as np, pandas as pd
from multiprocessing import Pool
SP, PANEL, POST, COEF, RECS, OUT = sys.argv[1:7]
sys.argv = ["shockar.py", SP, PANEL, POST, "/dev/null", "0", "2026-07-01"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import shockar as A, solve11 as V
coef = np.load(COEF)["coef"]
def clear(c, tm, load, r):
    p, y, ok, _ = V.solve_ip(c, tm, load, r["H"], r["A"], r["b"], r["e"], r["e0"], r["lo"], r["hi"])
    if not ok: p, y, ok, _ = V.solve_ip(c, tm, load, r["H"], r["A"], r["b"] + 0.10 * np.abs(r["b"]) + 50.0, r["e"], r["e0"], r["lo"], r["hi"])
    return (p, y) if ok else (None, None)
def one(r):
    out = []
    try:
        lg = [A.shock(r["t"] - k * A.DAY) for k in A.LAGS]
        if any(l is None for l in lg): raise ValueError("lag missing")
        c, tm = V.curves_logistic(A.TH, A.NB, r)
        for v, u in (("b0", lg[0]), ("c", A.predict_c(coef, np.stack(lg, 1)))):
            p, y = clear(c, tm, r["D"] + r["X"] + 1000 * u, r)
            out.append(dict(t=r["t"], variant=v, p_draws=[p[0] if p is not None else np.nan], m_draws=[-(r["H"] @ y)[0] / 1000 if p is not None else np.nan]))
    except Exception:
        out = [dict(t=r["t"], variant=v, p_draws=[np.nan], m_draws=[np.nan]) for v in ("b0", "c")]
    return out
if __name__ == "__main__":
    recs = pickle.load(open(RECS, "rb"))["recs"]
    with Pool(int(os.environ.get("NWORKERS", "2"))) as pool: rows = [x for o in pool.map(one, recs, chunksize=16) for x in o]
    R = pd.DataFrame(rows); R.to_pickle(OUT); print("missed share", R.p_draws.apply(lambda d: np.isnan(d[0])).groupby(R.variant).mean().round(3).to_dict())
