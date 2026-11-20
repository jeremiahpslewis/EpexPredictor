"""Amortised posterior with BayesFlow on the simulated datasets (notes 21.153): train, calibrate on held-out, read the observed dataset."""
import json, os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
from pathlib import Path
import numpy as np
import keras, bayesflow as bf

D = Path(sys.argv[1]); S = np.load(D / "sims.npz"); names = [str(n) for n in S["names"]]
theta, x, obs, truth = S["theta"].astype("float32"), S["x"].astype("float32"), S["obs"].astype("float32"), S["truth"]
n_test = 400; tr, te = slice(0, len(x) - n_test), slice(len(x) - n_test, len(x))
mu, sd = x[tr].reshape(-1, x.shape[-1]).mean(0), x[tr].reshape(-1, x.shape[-1]).std(0) + 1e-6
z = lambda a: ((a - mu) / sd).astype("float32")
def pack(sl): return {**{nm: theta[sl, i:i + 1] for i, nm in enumerate(names)}, "days": z(x[sl])}
keras.utils.set_random_seed(1)
wf = bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(), summary_network=bf.networks.DeepSet(summary_dim=int(os.environ.get("SUMDIM", 6))),
                      inference_variables=names, summary_variables=["days"], initial_learning_rate=5e-4)
train = pack(tr); val = {k: v[-300:] for k, v in train.items()}; fit = {k: v[:-300] for k, v in train.items()}
# stop at the best validation loss: the first run (150 epochs, no stopping) overfit (train -0.46, validation 14.4) and its
# posteriors covered the truth 38-65% of the time at nominal 90%
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(os.environ.get("PATIENCE", 15)), restore_best_weights=True)
hist = wf.fit_offline(fit, epochs=int(os.environ.get("EPOCHS", 300)), batch_size=64, validation_data=val, verbose=0, callbacks=[stop])
print("epochs run", len(hist.history["loss"]), "best val", round(float(min(hist.history["val_loss"])), 3), flush=True)
loss = hist.history; print("final loss", round(float(loss["loss"][-1]), 3), "val", round(float(loss.get("val_loss", [np.nan])[-1]), 3), flush=True)

post = wf.sample(conditions={"days": z(x[te])}, num_samples=1000)
prior_var = theta[tr].var(0); res = {"n_train": int(len(x) - n_test - 300), "n_test": n_test, "params": {}}
for i, nm in enumerate(names):
    s = np.asarray(post[nm])[..., 0]; tt = theta[te, i]
    lo, hi = np.quantile(s, 0.05, axis=1), np.quantile(s, 0.95, axis=1)
    contraction = 1 - s.var(1) / prior_var[i]
    ranks = (s < tt[:, None]).mean(1)
    res["params"][nm] = dict(contraction_mean=float(contraction.mean()), coverage90=float(((tt >= lo) & (tt <= hi)).mean()),
                             rank_hist=np.histogram(ranks, bins=10, range=(0, 1))[0].tolist(), corr_mean_true=float(np.corrcoef(s.mean(1), tt)[0, 1]),
                             post_mean=s.mean(1).tolist(), truth=tt.tolist())
po = wf.sample(conditions={"days": z(obs[None])}, num_samples=4000)
res["observed"] = {}
for i, nm in enumerate(names):
    s = np.asarray(po[nm])[0, :, 0]; q = np.quantile(s, [0.05, 0.5, 0.95])
    res["observed"][nm] = dict(truth=float(truth[i]), median=float(q[1]), lo=float(q[0]), hi=float(q[2]), sd=float(s.std()), prior_sd=float(np.sqrt(prior_var[i])), inside=bool(q[0] <= truth[i] <= q[2]))
json.dump(res, open(D / "sbi_results.json", "w"))
for nm in names:
    p, o = res["params"][nm], res["observed"][nm]
    print(f"{nm:9s} held-out: contraction {p['contraction_mean']:.2f}, 90% coverage {p['coverage90']:.2f}, corr(post mean, truth) {p['corr_mean_true']:.2f}, ranks {p['rank_hist']} | observed: truth {o['truth']:.3f}, median {o['median']:.3f} ({o['lo']:.3f} .. {o['hi']:.3f}), sd {o['sd']:.3f} vs prior {o['prior_sd']:.3f}, inside {o['inside']}")
