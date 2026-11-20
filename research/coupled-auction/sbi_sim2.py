"""
Joint recovery (notes 21.156): markup, offered share (pi) and the mean forced-outage share (forced_mean) free together;
the other parameters at their true values. Availability and offered share enter the bids only as a product, so prices
alone cannot separate them; published unavailability can. Each day therefore also carries the thermal available
capacity per zone as published, with 3% noise (two extra summaries at the end, which the fit can drop).
Writes sims2.npz. Seeds: datasets from 2,000,000 (no overlap with sbi_sim.py's ranges).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V

FREE = ("markup", "pi", "forced_mean"); DAYS = 5; info = V.Info(); OFFSET = 2_000_000 + int(os.environ.get("SEED_OFFSET", 0))
TRUTH = V.draw_theta(np.random.default_rng(7)); TRUTH["forced_mean"] = float(np.random.default_rng(8).beta(*V.PRIORS_EXT["forced_mean"][1:]))
THERMAL = {z: [k for k in V.TECH[z] if len(V.TECH[z][k]) > 1] for z in V.TECH}

def summaries(theta, seed):
    out = []
    for d in range(DAYS):
        rng = np.random.default_rng(seed * 100 + d)
        day = V.draw_day(rng, info, theta)
        try:
            m, acc, r = V.simulate(theta, day, info)
        except Exception:
            return None
        avail = [sum(acc[z]["avail"][k][0] for k in THERMAL[z]) * (1 + rng.normal(0, 0.03)) / 10_000.0 for z in ("A", "B")]
        out.append(np.r_[r.prices.reshape(2, 6, 16).mean(2).ravel(), r.f[0].reshape(6, 16).mean(1) / 100.0, avail])
    return np.array(out)

def one(i):
    i += OFFSET
    rng = np.random.default_rng(i); full = V.draw_theta(rng, extended=True); th = dict(TRUTH)
    for k in FREE: th[k] = full[k]
    x = summaries(th, i)
    return None if x is None else ([th[k] for k in FREE], x)

if __name__ == "__main__":
    out, n = Path(sys.argv[1]), int(sys.argv[2])
    with Pool(int(os.environ.get("WORKERS", 4))) as pool:
        res = [r for r in pool.imap(one, range(n), chunksize=8) if r is not None]
    theta = np.array([r[0] for r in res]); x = np.array([r[1] for r in res])
    obs = summaries(TRUTH, 999_998)
    np.savez_compressed(out / os.environ.get("OUTNAME", "sims2.npz"), dropped=n - len(res), theta=theta, x=x, obs=obs,
                        truth=np.array([TRUTH[k] for k in FREE]), names=np.array(FREE))
    print("simulated", len(res), "of", n, "datasets; x", x.shape, "truth", {k: round(TRUTH[k], 4) for k in FREE}, flush=True)
