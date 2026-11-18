"""second index model, one fit date, given seed and steps; saves the parameters and the log posterior"""
import sys, numpy as np, jax.numpy as jnp
import oc, ocx2
from ocx2 import *
from numpyro.infer.util import log_density
h1 = hx(sys.argv[1]); seed, steps = int(sys.argv[2]), int(sys.argv[3]); sl = slice(max(h1 - oc.TRAIN_H, 24 * 20), h1); d = rows2(sl)
ok = np.isfinite(d["p"]) & np.isfinite(d["sol"]) & np.isfinite(d["sol2"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["rl"]).all(1) & np.isfinite(d["gas"]); d = {k: np.nan_to_num(v[ok]) for k, v in d.items()}; dj = {k: jnp.asarray(v) for k, v in d.items()}
A = oc.run(modelX2, (dj,), steps, seed=seed); lp = float(log_density(modelX2, (dj,), {}, {k: jnp.asarray(v) for k, v in A.items()})[0]); np.savez(f"{oc.SP}/out/seedfit_{sys.argv[1]}_s{seed}_n{steps}.npz", lp=lp, **A)
print(f"{sys.argv[1]} seed {seed} steps {steps:6d}: log posterior {lp:10.2f} | " + " ".join(f"{k} {float(A[k]):.2f}" for k in ("ks", "kh", "tstar", "qa", "qw", "ww", "qb", "c0", "amax", "b0", "s1", "s2")), flush=True)
