"""
Error-persistence ("anchoring") analysis on stored stride-1 forecasts.

At cutoff k the prices of day k-1 are known, and so is the forecast that was issued for day k-1 at
cutoff k-1 (a genuine out-of-sample 1-day-ahead forecast). r_k = actual - forecast over that day.
Correction for horizon h:  f'(s) = f(s) + b_level[h] * mean(r_k) + b_shape[h] * (r_k(s) - mean(r_k))
The coefficients are re-estimated at every cutoff by least squares on *past* (r, error) pairs only
(pairs whose target day is already known at the cutoff), so the evaluation is walk-forward honest.
"""
import sys

import numpy as np

from evaluate import load, per_iter, summarize

out_dir, name, stage = sys.argv[1], sys.argv[2], sys.argv[3]
regions = sys.argv[4].split(",")
MIN_PAIRS = int(sys.argv[5]) if len(sys.argv) > 5 else 30
z, meta = load(out_dir, name)
assert meta["stride"] == 1
S = 96


def smooth(x, w=5):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")


def run(actual, pred, mode):
    n = len(actual)
    err = actual.astype(np.float64) - pred.astype(np.float64)          # [n, 289]
    r = np.full((n, S), np.nan)
    r[1:] = err[:-1, :S]                                                  # residual of the last known day
    lvl = np.nanmean(r, axis=1)
    shp = np.stack([smooth(x) if np.isfinite(x).all() else x for x in (r - lvl[:, None])])
    corrected = pred.astype(np.float64).copy()
    betas = []
    for k in range(n):
        if k < 1:
            continue
        b = []
        for h in range(3):
            # training pairs j: need r_j (j>=1) and the errors of day j+h, known at cutoff k iff j + h < k
            js = np.arange(1, k - h)
            if len(js) < MIN_PAIRS:
                b.append((0.0, 0.0))
                continue
            tgt = err[js, h * S:(h + 1) * S]                              # [m, 96]
            if mode == "level":
                x = lvl[js]
                bl = float((x * tgt.mean(axis=1)).sum() / (x * x).sum())
                b.append((bl, 0.0))
            else:
                x1 = np.repeat(lvl[js], S)
                x2 = shp[js].ravel()
                A = np.column_stack([x1, x2])
                coef, *_ = np.linalg.lstsq(A, tgt.ravel(), rcond=None)
                b.append((float(coef[0]), float(coef[1])))
        betas.append(b)
        for h in range(3):
            c = b[h][0] * lvl[k] + b[h][1] * shp[k]
            corrected[k, h * S:(h + 1) * S] += c
        # boundary slot (first slot of day 4) keeps the day-3 treatment
        corrected[k, 3 * S] += b[2][0] * lvl[k] + b[2][1] * shp[k][0]
    return corrected, np.array(betas)


print(f"{name}[{stage}] anchoring, walk-forward coefficients (min {MIN_PAIRS} past pairs)")
for mode in ("level", "level+shape"):
    print(f"-- {mode}")
    tot = {"b": np.zeros((2, 3)), "v": np.zeros((2, 3))}
    for r_ in regions:
        actual, pred = z[f"actual_{r_}"], z[f"{stage}_{r_}"]
        corr, betas = run(actual, pred, mode)
        sl = slice(MIN_PAIRS + 4, None)                                   # only cutoffs where a correction is active
        rb, mb = summarize(*per_iter(actual[sl], pred[sl]))
        rv, mv = summarize(*per_iter(actual[sl], corr[sl]))
        tot["b"] += np.stack([rb, mb]); tot["v"] += np.stack([rv, mv])
        last = betas[-1]
        print(f"{r_:<4} " + " | ".join(
            f"{h+1}d RMSE {rb[h]:.2f}>{rv[h]:.2f} {100*(rv[h]/rb[h]-1):+5.1f}%  MAE {mb[h]:.2f}>{mv[h]:.2f} {100*(mv[h]/mb[h]-1):+5.1f}%"
            for h in range(3)) + "   final b(level,shape) per h: " + " ".join(f"({a:.2f},{c:.2f})" for a, c in last))
    if len(regions) > 1:
        print("mean " + " | ".join(
            f"{h+1}d RMSE {100*(tot['v'][0,h]/tot['b'][0,h]-1):+5.1f}%  MAE {100*(tot['v'][1,h]/tot['b'][1,h]-1):+5.1f}%" for h in range(3)))

# raw persistence: correlation between the last known day's mean error and the day-h mean error
print("-- autocorrelation of daily-mean forecast error (lag = h days after the last known day)")
for r_ in regions:
    err = z[f"actual_{r_}"].astype(np.float64) - z[f"{stage}_{r_}"].astype(np.float64)
    lvl = err[:-1, :S].mean(axis=1)
    out = []
    for h in range(3):
        tgt = err[1:, h * S:(h + 1) * S].mean(axis=1)
        out.append(np.corrcoef(lvl, tgt)[0, 1])
    print(f"{r_:<4} " + "  ".join(f"h={h+1}: {c:+.2f}" for h, c in enumerate(out)))
