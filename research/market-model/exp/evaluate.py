"""Offline metrics for stored backtest predictions. Same definitions as performance_testing.py:
per cutoff, MAE / MSE over each horizon day (inclusive slice, 97 slots); MAE = mean over cutoffs,
RMSE = sqrt(mean over cutoffs of the per-cutoff MSE)."""
import json
import sys

import numpy as np

H = [(0, 97), (96, 193), (192, 289)]


def load(out_dir, name):
    z = np.load(f"{out_dir}/{name}.npz")
    meta = json.load(open(f"{out_dir}/{name}.json"))
    return z, meta


HOURS = None  # (lo, hi): only score slots whose local (Europe/Berlin) hour is in [lo, hi)
_MASKS = {}


def hour_mask(cutoffs):
    key = (cutoffs.tobytes(), HOURS)
    if key not in _MASKS:
        import pandas as pd
        t = pd.DatetimeIndex((cutoffs[:, None] + (np.arange(289) * np.timedelta64(15, "m"))[None, :]).ravel(), tz="UTC")
        hr = np.asarray(t.tz_convert("Europe/Berlin").hour).reshape(len(cutoffs), 289)
        _MASKS[key] = (hr >= HOURS[0]) & (hr < HOURS[1])
    return _MASKS[key]


def per_iter(actual, pred, cutoffs=None):
    """-> mae[n_iter, 3], mse[n_iter, 3]"""
    err = pred.astype(np.float64) - actual.astype(np.float64)
    if HOURS is not None:
        err = np.where(hour_mask(cutoffs), err, np.nan)
    mae = np.stack([np.nanmean(np.abs(err[:, a:b]), axis=1) for a, b in H], axis=1)
    mse = np.stack([np.nanmean(err[:, a:b] ** 2, axis=1) for a, b in H], axis=1)
    return mae, mse


def summarize(mae, mse):
    return np.sqrt(mse.mean(axis=0)), mae.mean(axis=0)


def readme_table(z, stage, regions):
    print("| Region | 1d RMSE | 1d MAE | 2d RMSE | 2d MAE | 3d RMSE | 3d MAE |")
    print("|--------|---------|--------|---------|--------|---------|--------|")
    for r in regions:
        rmse, mae = summarize(*per_iter(z[f"actual_{r}"], z[f"{stage}_{r}"], z["cutoffs"]))
        print(f"| {r:<6} | " + " | ".join(f"{rmse[h]:<7.2f} | {mae[h]:<6.2f}" for h in range(3)) + " |")


def block_boot(stat_fn, n, block, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        vals[b] = stat_fn(idx)
    return np.percentile(vals, [2.5, 97.5])


def compare(zb, zv, stage_b, stage_v, regions, block=7, verbose=True, label=""):
    """Paired comparison on the common cutoffs. Returns dict of relative changes (negative = better)."""
    cb, cv = zb["cutoffs"], zv["cutoffs"]
    common = np.intersect1d(cb, cv)
    ib = np.searchsorted(cb, common)
    iv = np.searchsorted(cv, common)
    stride_days = max(1, int(round(np.median(np.diff(common).astype("timedelta64[D]").astype(int))))) if len(common) > 1 else 1
    blk = max(2, int(np.ceil(block / stride_days)))
    rows = {}
    allmae_b, allmae_v, allmse_b, allmse_v = [], [], [], []
    for r in regions:
        mae_b, mse_b = per_iter(zb[f"actual_{r}"][ib], zb[f"{stage_b}_{r}"][ib], common)
        mae_v, mse_v = per_iter(zv[f"actual_{r}"][iv], zv[f"{stage_v}_{r}"][iv], common)
        allmae_b.append(mae_b); allmae_v.append(mae_v); allmse_b.append(mse_b); allmse_v.append(mse_v)
        rmse_b, m_b = summarize(mae_b, mse_b)
        rmse_v, m_v = summarize(mae_v, mse_v)
        rows[r] = (rmse_b, m_b, rmse_v, m_v)
    # pooled over regions (mean of per-region metrics), with a block bootstrap over cutoffs
    MB, MV = np.stack(allmae_b), np.stack(allmae_v)   # [region, iter, h]
    SB, SV = np.stack(allmse_b), np.stack(allmse_v)
    n = MB.shape[1]

    def rel_mae(idx, h):
        return MV[:, idx, h].mean(axis=1).sum() / MB[:, idx, h].mean(axis=1).sum() - 1

    def rel_rmse(idx, h):
        return np.sqrt(SV[:, idx, h].mean(axis=1)).sum() / np.sqrt(SB[:, idx, h].mean(axis=1)).sum() - 1

    full = np.arange(n)
    out = {"n": n}
    for h in range(3):
        out[f"mae_{h+1}d"] = rel_mae(full, h)
        out[f"rmse_{h+1}d"] = rel_rmse(full, h)
        out[f"mae_{h+1}d_ci"] = block_boot(lambda i: rel_mae(i, h), n, blk)
        out[f"rmse_{h+1}d_ci"] = block_boot(lambda i: rel_rmse(i, h), n, blk)
    if verbose:
        print(f"\n== {label}  (n={n} cutoffs, paired; negative = better) ==")
        print("region |  1d RMSE        1d MAE        |  2d RMSE        2d MAE        |  3d RMSE        3d MAE")
        for r in regions:
            rmse_b, m_b, rmse_v, m_v = rows[r]
            print(f"{r:<6} | " + " | ".join(
                f"{rmse_b[h]:.2f}>{rmse_v[h]:.2f} {100*(rmse_v[h]/rmse_b[h]-1):+5.1f}%  {m_b[h]:.2f}>{m_v[h]:.2f} {100*(m_v[h]/m_b[h]-1):+5.1f}%"
                for h in range(3)))
        print("pooled | " + " | ".join(
            f"RMSE {100*out[f'rmse_{h+1}d']:+5.1f}% [{100*out[f'rmse_{h+1}d_ci'][0]:+.1f},{100*out[f'rmse_{h+1}d_ci'][1]:+.1f}]  "
            f"MAE {100*out[f'mae_{h+1}d']:+5.1f}% [{100*out[f'mae_{h+1}d_ci'][0]:+.1f},{100*out[f'mae_{h+1}d_ci'][1]:+.1f}]"
            for h in range(3)))
    return out


def oneline(out):
    return "  ".join(
        f"{h}d: RMSE {100*out[f'rmse_{h}d']:+5.1f}% [{100*out[f'rmse_{h}d_ci'][0]:+5.1f},{100*out[f'rmse_{h}d_ci'][1]:+5.1f}] "
        f"MAE {100*out[f'mae_{h}d']:+5.1f}% [{100*out[f'mae_{h}d_ci'][0]:+5.1f},{100*out[f'mae_{h}d_ci'][1]:+5.1f}]"
        for h in (1, 2, 3))


if __name__ == "__main__":
    # evaluate.py OUT_DIR table NAME STAGE
    # evaluate.py OUT_DIR cmp BASE:STAGE VAR:STAGE [VAR:STAGE ...] [--regions A,B] [--brief]
    out_dir, mode = sys.argv[1], sys.argv[2]
    args = [x for x in sys.argv[3:] if not x.startswith("--")]
    opts = dict(x[2:].split("=", 1) if "=" in x else (x[2:], "1") for x in sys.argv[3:] if x.startswith("--"))
    if "hours" in opts:
        HOURS = tuple(int(v) for v in opts["hours"].split("-"))
    if mode == "table":
        z, meta = load(out_dir, args[0])
        regions = opts.get("regions", ",".join(meta["cfg"]["eval_regions"])).split(",")
        print(f"{args[0]} stage={args[1]} n={meta['n_iter']} stride={meta['stride']} ({meta['seconds']}s)")
        readme_table(z, args[1], regions)
    else:
        bname, bstage = args[0].split(":")
        zb, mb = load(out_dir, bname)
        for v in args[1:]:
            vname, vstage = v.split(":")
            zv, mv = load(out_dir, vname)
            regions = opts.get("regions")
            regions = regions.split(",") if regions else [r for r in mv["cfg"]["eval_regions"] if r in mb["cfg"]["eval_regions"]]
            label = f"{vname}[{vstage}] vs {bname}[{bstage}]"
            o = compare(zb, zv, bstage, vstage, regions, verbose="brief" not in opts, label=label)
            if "brief" in opts:
                print(f"{label:<58} n={o['n']:<4} {oneline(o)}")
