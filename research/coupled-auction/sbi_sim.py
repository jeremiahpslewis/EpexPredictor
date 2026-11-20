"""Simulate datasets of five synthetic days for amortised inference (notes 21.153). Writes sims.npz."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V

FREE = ("markup", "netting", "flex_wtp"); DAYS = 5; info = V.Info()
TRUTH = V.draw_theta(np.random.default_rng(7))

def summaries(theta, seed):
    out = []
    for d in range(DAYS):
        day = V.draw_day(np.random.default_rng(seed * 100 + d), info)
        try:
            m, acc, r = V.simulate(theta, day, info)
        except Exception:
            return None
        out.append(np.r_[r.prices.reshape(2, 6, 16).mean(2).ravel(), r.f[0].reshape(6, 16).mean(1) / 100.0])
    return np.array(out)

def one(i):
    rng = np.random.default_rng(10_000 + i); full = V.draw_theta(rng); th = dict(TRUTH)
    for k in FREE: th[k] = full[k]
    x = summaries(th, 50_000 + i)
    return None if x is None else ([th[k] for k in FREE], x)

if __name__ == "__main__":
    out, n = Path(sys.argv[1]), int(sys.argv[2])
    with Pool(int(os.environ.get("WORKERS", 4))) as pool:
        res = [r for r in pool.imap(one, range(n), chunksize=8) if r is not None]
    theta = np.array([r[0] for r in res]); x = np.array([r[1] for r in res])
    obs = summaries(TRUTH, 999_999)
    np.savez_compressed(out / "sims.npz", theta=theta, x=x, obs=obs, truth=np.array([TRUTH[k] for k in FREE]), names=np.array(FREE))
    print("simulated", len(res), "of", n, "datasets; x", x.shape, flush=True)
