"""Is the week-to-week jitter of the weekly fits the posterior moving, or the optimiser not arriving? Same data, different seeds and step counts."""
import sys, numpy as np, jax.numpy as jnp
import oc, ocx2
from ocx2 import *
h1 = hx(sys.argv[1]); seed, steps = int(sys.argv[2]), int(sys.argv[3]); sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows2(sl)
ok = np.isfinite(d["p"]) & np.isfinite(d["sol"]) & np.isfinite(d["sol2"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]); d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}
A = oc.run(modelX2, ({k: jnp.asarray(v) for k, v in d.items()},), steps, seed=seed); ph = pred2(A, d)
import numpyro
from numpyro.infer.util import log_density
lp = float(log_density(modelX2, ({k: jnp.asarray(v) for k, v in d.items()},), {}, {k: jnp.asarray(v) for k, v in A.items()})[0])
print(f"seed {seed} steps {steps:5d}: log posterior {lp:10.2f} | MAE {np.mean(np.abs(d['p'] - ph)) / 10:.3f} | " + " ".join(f"{k} {float(A[k]):.2f}" for k in ("ks", "kn", "kh", "tstar", "qa", "qw", "qb", "c0", "amax", "b0")) + " | wn " + " ".join(f"{v:.2f}" for v in A["wn"]))
