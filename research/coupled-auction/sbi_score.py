"""
Score the frozen estimator (sbi_fit.py: estimator.keras, standardise.npz) on fresh datasets without retraining (notes
21.155). Usage: sbi_score.py ESTIMATOR_DIR FRESH.npz [OTHER_RULE.npz]. Coverage at 50/80/90% with Wilson 95% intervals,
rank histograms, contraction; with a second file of the same seeds under another volume rule, how far the summaries and
posterior medians move. Writes score.json next to the first fresh file.
"""
import json, os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
from pathlib import Path
import numpy as np
import keras, bayesflow as bf  # noqa: F401  (registers BayesFlow's classes for loading)

E, F = Path(sys.argv[1]), Path(sys.argv[2]); G = Path(sys.argv[3]) if len(sys.argv) > 3 else None
est = keras.saving.load_model(E / "estimator.keras"); st = np.load(E / "standardise.npz")
mu, sd, names = st["mu"], st["sd"], [str(n) for n in st["names"]]
prior_sd = np.load(E / "sims.npz")["theta"][:-400].std(0)

def posterior(x, n=1000):
    s = est.sample(conditions={"days": ((x - mu) / sd).astype("float32")}, num_samples=n)
    return np.stack([np.asarray(s[nm])[..., 0] for nm in names], -1)      # datasets x samples x parameters

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]

A = np.load(F); th, x = A["theta"], A["x"].astype("float32")
post = np.concatenate([posterior(x[i:i + 250]) for i in range(0, len(x), 250)])
res = {"file": F.name, "datasets": int(len(x)), "dropped": int(A["dropped"]) if "dropped" in A else None, "params": {}}
for j, nm in enumerate(names):
    s, t = post[..., j], th[:, j]; r = {"contraction": float(1 - (s.var(1) / prior_sd[j] ** 2).mean()),
                                        "corr": float(np.corrcoef(s.mean(1), t)[0, 1]),
                                        "ranks": np.histogram((s < t[:, None]).mean(1), bins=10, range=(0, 1))[0].tolist()}
    for lev in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(s, (1 - lev) / 2, axis=1), np.quantile(s, (1 + lev) / 2, axis=1); k = int(((t >= lo) & (t <= hi)).sum())
        r[f"cov{int(lev * 100)}"] = round(k / len(t), 3); r[f"cov{int(lev * 100)}_ci"] = wilson(k, len(t))
    res["params"][nm] = r
if G is not None:
    B = np.load(G); assert np.allclose(B["theta"], th), "the two files must hold the same seeds"
    xb = B["x"].astype("float32"); postb = np.concatenate([posterior(xb[i:i + 250]) for i in range(0, len(xb), 250)])
    dx = np.abs(xb - x)
    res["other_rule"] = {"file": G.name, "max_abs_change_price_summaries": float(dx[..., :12].max()),
                         "share_flow_summaries_changed": float((dx[..., 12:] > 1e-6).mean()),
                         "max_abs_change_flow_summary_MW": float(dx[..., 12:].max() * 100)}
    for lev in (0.9,):
        pass
    for j, nm in enumerate(names):
        d = np.abs(np.median(postb[..., j], 1) - np.median(post[..., j], 1)) / prior_sd[j]
        lo, hi = np.quantile(postb[..., j], 0.05, 1), np.quantile(postb[..., j], 0.95, 1); k = int(((th[:, j] >= lo) & (th[:, j] <= hi)).sum())
        res["other_rule"][nm] = {"median_shift_prior_sd_q50_q90_max": np.quantile(d, [0.5, 0.9, 1.0]).round(3).tolist(),
                                 "cov90_other_rule": round(k / len(th), 3), "cov90_ci": wilson(k, len(th))}
json.dump(res, open(F.parent / "score.json", "w"), indent=1)
print(json.dumps(res, indent=1))
