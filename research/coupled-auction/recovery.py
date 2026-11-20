"""
Posterior recovery on synthetic data (crude rejection ABC; BayesFlow is the tool at scale): three observed days drawn
with known parameters; three parameters inferred (markup, netting, flexible demand's willingness to pay), the others
held at their true values; the day-level shocks are latent and redrawn in every simulation.
Expected before running: markup recovered (it moves price levels); netting NOT identified (it shifts both sides of the
book equally and cannot move prices or flows), posterior close to the prior; willingness to pay weakly identified.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FREE = ("markup", "netting", "flex_wtp"); info = V.Info()
rng = np.random.default_rng(7); truth = V.draw_theta(rng)
def stats(theta, seed):
    out = []
    for d in range(3):
        day = V.draw_day(np.random.default_rng(seed * 10 + d), info); m, acc, r = V.simulate(theta, day, info)
        out += list(r.prices.reshape(2, 6, 16).mean(2).ravel()) + list(r.f[0].reshape(6, 16).mean(1) / 100.0)
    return np.array(out)
obs = stats(truth, 999)
rngp = np.random.default_rng(11); draws, S = [], []; t0 = time.time()
for i in range(N):
    th = dict(truth); full = V.draw_theta(rngp)
    for k in FREE: th[k] = full[k]
    draws.append([th[k] for k in FREE]); S.append(stats(th, 1000 + i))
    if i % 50 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
D, S = np.array(draws), np.array(S); sd = S.std(0) + 1e-9; dist = np.sqrt((((S - obs) / sd) ** 2).sum(1)); keep = np.argsort(dist)[: max(20, N // 10)]
res = {}
for j, k in enumerate(FREE):
    res[k] = dict(truth=truth[k], prior_mean=float(D[:, j].mean()), prior_sd=float(D[:, j].std()), post_mean=float(D[keep, j].mean()), post_sd=float(D[keep, j].std()))
    print(f"{k:9s} true {truth[k]:7.3f} | prior {res[k]['prior_mean']:7.3f} sd {res[k]['prior_sd']:.3f} | posterior {res[k]['post_mean']:7.3f} sd {res[k]['post_sd']:.3f} | sd ratio {res[k]['post_sd'] / res[k]['prior_sd']:.2f}")
json.dump(dict(N=N, accepted=len(keep), params=res), open(OUT / "recovery.json", "w")); print("finished", f"{time.time() - t0:.0f}s")
