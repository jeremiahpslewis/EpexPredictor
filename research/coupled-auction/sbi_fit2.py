"""
Joint fit for sbi_sim2.py (notes 21.156): markup, pi, forced_mean; the shares on the logit scale so no posterior draw
leaves [0, 1]. Usage: sbi_fit2.py DIR TAG; env AVAIL=0 drops the two published-availability summaries. Same training
recipe as sbi_fit.py (early stopping, DeepSet summary_dim 6, 400 held out). Writes fit2_TAG.json, post2_TAG.npz
(posterior draws for the observed dataset) and the estimator.
"""
import json, os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
from pathlib import Path
import numpy as np
import keras, bayesflow as bf

D, TAG = Path(sys.argv[1]), sys.argv[2]; S = np.load(D / "sims2.npz"); names = [str(n) for n in S["names"]]
keep = slice(None) if os.environ.get("AVAIL", "1") == "1" else slice(0, 18)
LOGIT = {"pi", "forced_mean"}
fwd = lambda v, nm: np.log(v / (1 - v)) if nm in LOGIT else v
inv = lambda v, nm: 1 / (1 + np.exp(-v)) if nm in LOGIT else v
theta_raw, x, obs, truth = S["theta"], S["x"][..., keep].astype("float32"), S["obs"][..., keep].astype("float32"), S["truth"]
theta = np.stack([fwd(theta_raw[:, i], nm) for i, nm in enumerate(names)], 1).astype("float32")
n_test = 400; tr, te = slice(0, len(x) - n_test), slice(len(x) - n_test, len(x))
mu, sd = x[tr].reshape(-1, x.shape[-1]).mean(0), x[tr].reshape(-1, x.shape[-1]).std(0) + 1e-6
z = lambda a: ((a - mu) / sd).astype("float32")
def pack(sl): return {**{nm: theta[sl, i:i + 1] for i, nm in enumerate(names)}, "days": z(x[sl])}
keras.utils.set_random_seed(1)
wf = bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(), summary_network=bf.networks.DeepSet(summary_dim=6),
                      inference_variables=names, summary_variables=["days"], initial_learning_rate=5e-4)
train = pack(tr); val = {k: v[-300:] for k, v in train.items()}; fit = {k: v[:-300] for k, v in train.items()}
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
hist = wf.fit_offline(fit, epochs=300, batch_size=64, validation_data=val, verbose=0, callbacks=[stop])
wf.estimator.save(D / f"estimator2_{TAG}.keras"); np.savez(D / f"standardise2_{TAG}.npz", mu=mu, sd=sd)
post = wf.sample(conditions={"days": z(x[te])}, num_samples=1000)
res = {"tag": TAG, "avail_summaries": keep == slice(None), "epochs": len(hist.history["loss"]), "best_val": float(min(hist.history["val_loss"])),
       "n_train": int(len(x) - n_test - 300), "n_test": n_test, "dropped": int(S["dropped"]), "params": {}}
prior_var = theta_raw[tr].var(0)
for i, nm in enumerate(names):
    s = inv(np.asarray(post[nm])[..., 0], nm); tt = theta_raw[te, i]; r = {}
    for lev in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(s, (1 - lev) / 2, 1), np.quantile(s, (1 + lev) / 2, 1); r[f"cov{int(lev * 100)}"] = float(((tt >= lo) & (tt <= hi)).mean())
    r.update(contraction=float((1 - s.var(1) / prior_var[i]).mean()), corr=float(np.corrcoef(s.mean(1), tt)[0, 1]),
             ranks=np.histogram((s < tt[:, None]).mean(1), bins=10, range=(0, 1))[0].tolist())
    res["params"][nm] = r
# correlation between the parameters within each held-out posterior (the ridge shows as a strong correlation)
P = np.stack([inv(np.asarray(post[nm])[..., 0], nm) for nm in names], -1)
res["within_posterior_corr_median"] = {f"{names[a]}~{names[b]}": float(np.median([np.corrcoef(P[d, :, a], P[d, :, b])[0, 1] for d in range(len(P))]))
                                       for a in range(3) for b in range(a + 1, 3)}
po = wf.sample(conditions={"days": z(obs[None])}, num_samples=4000)
O = np.stack([inv(np.asarray(po[nm])[0, :, 0], nm) for nm in names], -1); np.savez(D / f"post2_{TAG}.npz", draws=O, names=np.array(names), truth=truth)
res["observed"] = {nm: dict(truth=float(truth[i]), median=float(np.median(O[:, i])), lo=float(np.quantile(O[:, i], 0.05)), hi=float(np.quantile(O[:, i], 0.95)),
                            prior_sd=float(np.sqrt(prior_var[i]))) for i, nm in enumerate(names)}
res["observed_corr"] = {f"{names[a]}~{names[b]}": float(np.corrcoef(O[:, a], O[:, b])[0, 1]) for a in range(3) for b in range(a + 1, 3)}
# the product the prices see: offered share of nominal thermal capacity, pi * (1 - forced_mean)
prod = O[:, 1] * (1 - O[:, 2]); res["observed_offered_product"] = dict(truth=float(truth[1] * (1 - truth[2])), lo=float(np.quantile(prod, 0.05)),
                                                                        median=float(np.median(prod)), hi=float(np.quantile(prod, 0.95)))
json.dump(res, open(D / f"fit2_{TAG}.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items()}, indent=1))
