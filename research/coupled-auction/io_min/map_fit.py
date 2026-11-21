"""MAP point on real data (used as theta_true for simulate-and-recover). Usage: map_fit.py PANEL_DIR OUT.json"""
import sys, json
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer import init_to_median
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as Mo
Pn = pd.read_parquet(f"{sys.argv[1]}/panel.parquet"); x, obs, brk = Mo.to_inputs(Pn)
g = AutoDelta(Mo.model, init_loc_fn=init_to_median(num_samples=15))
svi = SVI(Mo.model, g, optax.adam(0.01), Trace_ELBO())
r = svi.run(jax.random.PRNGKey(0), 6000, x, obs, brk, progress_bar=False)
print("loss start/end", float(r.losses[0]), float(r.losses[-1]), float(r.losses[-500:].std()))
p = g.median(r.params); out = {k: np.asarray(v).tolist() for k, v in p.items()}
for k, v in out.items(): print(k, np.round(v, 3))
json.dump(out, open(sys.argv[2], "w"))
