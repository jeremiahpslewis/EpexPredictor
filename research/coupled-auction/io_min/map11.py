"""
MAP screen for the eleven-zone fit (any fit11 configuration). Usage (same arguments as fit11, then a start label):
    map11.py RECS.pkl BASIN.npz OUT.json START_LABEL [STEPS]
Optimises the fit11 log posterior with Adam on the unconstrained scale (numpyro SVI, AutoDelta) from the fit11 start,
with the curve-end parameters overridden by START (env MAP_START, JSON). Reports the MAP end parameters and the implied
supply slope dS/dp by price band for Germany and for the ten neighbours together, at the MTUs of the data.
"""
import sys, os, json
RECS, BASIN, OUT, LABEL = sys.argv[1:5]; STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
sys.argv = ["fit11.py", RECS, BASIN, "/dev/null", "1", "1", "1"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import fit11 as F
import numpy as np, jax, jax.numpy as jnp, numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta
init = dict(F.init) | {k: float(v) for k, v in json.loads(os.environ.get("MAP_START", "{}")).items()}
guide = AutoDelta(F.model, init_loc_fn=init_to_value(values={k: jnp.asarray(v) for k, v in init.items() if k != "Lcorr"}))
svi = SVI(F.model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
res = svi.run(jax.random.PRNGKey(0), STEPS, progress_bar=False)
mp = {k.replace("_auto_loc", ""): np.asarray(v) for k, v in guide.median(res.params).items()}
ends = {k: float(mp[k]) for k in F.END_KEYS}
# implied slopes by band at the observed prices (truth comparison is done by the caller)
p = np.asarray(F.p); bands = {"<=0": p[:, 0] <= 0, "0-40": (p[:, 0] > 0) & (p[:, 0] <= 40), "40-200": (p[:, 0] > 40) & (p[:, 0] <= 200), ">200": p[:, 0] > 200}
out = dict(label=LABEL, final_loss=float(res.losses[-1]), loss_drop_last_500=float(res.losses[-500] - res.losses[-1]), ends=ends,
           de={k: float(mp[k]) for k in F.DE_KEYS}, n_band={b: int(m.sum()) for b, m in bands.items()})
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out))
