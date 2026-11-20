"""
How the parameter uncertainty of the joint posterior moves the counterfactuals (notes 21.156). For draws of (markup, pi,
forced_mean) from the prior, from each posterior (post2_TAG.npz) and the truth, with common day shocks (the same day seeds
for every draw), compute: ATC 1000 -> 500 with the same orders (change in B's mean price) and gas 40 -> 52 with bids
regenerated (change in A's mean price). Usage: sbi_cf.py DIR TAG [TAG ...]. Writes cf2.json.
"""
import copy, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V
from clearing import Line, clear
from sbi_sim2 import FREE, TRUTH

D = Path(sys.argv[1]); TAGS = sys.argv[2:]; N = int(os.environ.get("NDRAWS", 120)); DAYSEEDS = (777_001, 777_002, 777_003)
info = V.Info()

def effects(vals):
    th = dict(TRUTH); th.update(dict(zip(FREE, vals))); out = []
    for s in DAYSEEDS:
        day = V.draw_day(np.random.default_rng(s), info, th)
        try:
            m, acc, r = V.simulate(th, day, info)
            m1 = copy.deepcopy(m); m1.lines = [Line("A", "B", [500.0] * V.T, [500.0] * V.T)]; r1 = clear(m1, intervals=False)
            i2 = copy.deepcopy(info); i2.gas = 52.0; _, _, r2 = V.simulate(th, day, i2)
        except Exception:
            return None
        out.append([r1.prices[1].mean() - r.prices[1].mean(), r2.prices[0].mean() - r.prices[0].mean(), r.prices[0].mean(), r.prices[1].mean()])
    return np.mean(out, 0).tolist()

if __name__ == "__main__":
    rng = np.random.default_rng(5); sets = {"truth": [[TRUTH[k] for k in FREE]]}
    pri = [V.draw_theta(rng, extended=True) for _ in range(N)]; sets["prior"] = [[p[k] for k in FREE] for p in pri]
    for tag in TAGS:
        P = np.load(D / f"post2_{tag}.npz")["draws"]; sets[f"posterior_{tag}"] = P[rng.choice(len(P), N, replace=False)].tolist()
    jobs = [(nm, v) for nm, vs in sets.items() for v in vs]
    with Pool(int(os.environ.get("WORKERS", 4))) as pool: out = pool.map(effects, [v for _, v in jobs], chunksize=2)
    res = {}
    for nm in sets:
        E = np.array([o for (n_, _), o in zip(jobs, out) if n_ == nm and o is not None])
        res[nm] = {"draws": int(len(E)), "atc_dB": np.quantile(E[:, 0], [0.05, 0.5, 0.95]).round(2).tolist(), "gas_dA": np.quantile(E[:, 1], [0.05, 0.5, 0.95]).round(2).tolist(),
                   "base_A": np.quantile(E[:, 2], [0.05, 0.5, 0.95]).round(1).tolist(), "base_B": np.quantile(E[:, 3], [0.05, 0.5, 0.95]).round(1).tolist()}
    res["draws_by_set"] = {nm: [[float(a) for a in v] + (o or []) for (n_, v), o in zip(jobs, out) if n_ == nm] for nm in sets}
    json.dump(res, open(D / "cf2.json", "w")); print(json.dumps({k: v for k, v in res.items() if k != "draws_by_set"}, indent=1))
