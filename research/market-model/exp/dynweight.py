"""Would a learned, time-varying weight between the price model (raw) and the gas-relative model (ratio) beat
the fixed average?

Uses the stored rolling-origin forecasts of both models. For each cutoff, the weight is estimated only from
forecast errors that were already realized at that cutoff (forecasts issued at earlier cutoffs, slots before
the cutoff), within a trailing window. Evaluated with the same metrics as everywhere else.

usage: dynweight.py OUT_DIR STAGE ZONES [raw_name ratio_name per period ...]
"""
import sys

import numpy as np

import evaluate as ev

out = sys.argv[1]
stage = sys.argv[2]            # key prefix of the stored predictions, e.g. s1
zones = sys.argv[3].split(",")
RAW = sys.argv[4].split(",")   # one file per period, in time order
RATIO = sys.argv[5].split(",")
labels = sys.argv[6].split(",")

BRIEF = sys.argv[7].split(",") if len(sys.argv) > 7 and sys.argv[7] else []
CI = ["ls 28d", "l1 56d"]
SLOT = np.timedelta64(15, "m")
DAY = np.timedelta64(1, "D")


def load(zone):
    """Concatenate the periods. -> cutoffs[n], raw[n,289], ratio[n,289], actual[n,289], period[n]"""
    cut, a, b, y, per = [], [], [], [], []
    for p, (fa, fb) in enumerate(zip(RAW, RATIO)):
        fa, ka = fa.split(":") if ":" in fa else (fa, stage)
        fb, kb = fb.split(":") if ":" in fb else (fb, stage)
        za, zb = np.load(f"{out}/{fa}.npz"), np.load(f"{out}/{fb}.npz")
        assert (za["cutoffs"] == zb["cutoffs"]).all()
        cut.append(za["cutoffs"])
        a.append(za[f"{ka}_{zone}"].astype(np.float64))
        b.append(zb[f"{kb}_{zone}"].astype(np.float64))
        y.append(za[f"actual_{zone}"].astype(np.float64))
        per.append(np.full(len(za["cutoffs"]), p))
    return np.concatenate(cut), np.concatenate(a), np.concatenate(b), np.concatenate(y), np.concatenate(per)


def evidence(i, cut, window_days, horizon_days=3):
    """Index arrays (j, k) of forecast slots realized before cutoff i and inside the trailing window."""
    t = cut[:i, None] + np.arange(289)[None, :] * SLOT          # time of every earlier forecast slot
    ok = (t < cut[i]) & (t >= cut[i] - window_days * DAY)
    ok[:, horizon_days * 96 + 1:] = False
    return np.nonzero(ok)


def w_ls(ya, d):
    """least squares weight of the ratio model: argmin sum (y - a - w (b - a))^2"""
    den = (d * d).sum()
    return 0.5 if den <= 0 else float(np.clip((ya * d).sum() / den, 0.0, 1.0))


GRID = np.linspace(0, 1, 11)


def w_l1(ya, d):
    """weight on a grid that minimizes the absolute error"""
    return float(GRID[np.argmin([np.abs(ya - w * d).mean() for w in GRID])])


def w_invmse(ya, d):
    ma, mb = (ya ** 2).mean(), ((ya - d) ** 2).mean()
    return 0.5 if ma + mb <= 0 else float(ma / (ma + mb))


def w_select(ya, d):
    return 1.0 if ((ya - d) ** 2).mean() < (ya ** 2).mean() else 0.0


EST = {"ls": w_ls, "l1": w_l1, "invmse": w_invmse, "select": w_select}


def weights(cut, a, b, y, est, window_days, shrink=1.0, min_slots=96 * 3, horizon_days=3):
    w = np.full(len(cut), 0.5)
    for i in range(len(cut)):
        j, k = evidence(i, cut, window_days, horizon_days)
        if len(j) < min_slots:
            continue
        ya = y[j, k] - a[j, k]
        d = b[j, k] - a[j, k]
        good = np.isfinite(ya) & np.isfinite(d)
        if good.sum() < min_slots:
            continue
        w[i] = 0.5 + shrink * (EST[est](ya[good], d[good]) - 0.5)
    return w


def score(y, pred, cut, sel):
    res = {}
    for hours, name in [((0, 8), "00-08"), (None, "all")]:
        ev.HOURS = hours
        mae, mse = ev.per_iter(y[sel], pred[sel], cut[sel])
        res[name] = ev.summarize(mae, mse)[1]
    return res


def fmt(m, base):
    return "/".join(f"{x:.2f}" for x in m) + "  " + "/".join(f"{100 * (x / b - 1):+.0f}%" for x, b in zip(m, base))


strategies = [("fixed 0.5 (now)", None)]
strategies += [(f"fixed {w:.2f}", w) for w in (0.0, 0.25, 0.75, 1.0)]
for est in ("ls", "l1", "invmse", "select"):
    for win in (7, 14, 28, 56):
        strategies.append((f"{est} {win}d", (est, win, 1.0)))
strategies += [(f"ls {win}d half-shrunk", ("ls", win, 0.5)) for win in (14, 28, 56)]
strategies += [("ls 28d day-1 errors only", ("ls", 28, 1.0, 1))]

for zone in zones:
    cut, a, b, y, per = load(zone)
    print(f"\n================ {zone}: MAE ct/kWh 1/2/3 days ahead, change vs fixed 0.5")
    results = {}
    for name, spec in strategies:
        if spec is None:
            w = np.full(len(cut), 0.5)
        elif isinstance(spec, float):
            w = np.full(len(cut), spec)
        else:
            est, win, shrink = spec[:3]
            w = weights(cut, a, b, y, est, win, shrink, horizon_days=spec[3] if len(spec) > 3 else 3)
        pred = a + w[:, None] * (b - a)
        results[name] = (w, {p: score(y, pred, cut, per == p) for p in range(len(labels))})
    w_oracle = np.array([w_l1((y[i] - a[i])[np.isfinite(y[i])], (b[i] - a[i])[np.isfinite(y[i])]) for i in range(len(cut))])
    results["oracle per forecast"] = (w_oracle, {p: score(y, a + w_oracle[:, None] * (b - a), cut, per == p) for p in range(len(labels))})
    for p, label in enumerate(labels):
        for hours in ("00-08", "all"):
            print(f"--- {label}, {hours}")
            base = results["fixed 0.5 (now)"][1][p][hours]
            for name in [n for n, _ in strategies] + ["oracle per forecast"]:
                if BRIEF and name not in BRIEF:
                    continue
                w, res = results[name]
                print(f"  {name:<26} {fmt(res[p][hours], base)}   mean weight {w[per == p].mean():.2f}")
            # paired block bootstrap of the change in MAE for the learned weights
            ev.HOURS = (0, 8) if hours == "00-08" else None
            sel = per == p
            stride = max(1, int(round(np.median(np.diff(cut[sel]).astype("timedelta64[D]").astype(int)))))
            blk = max(2, int(np.ceil(7 / stride)))
            mae0, _ = ev.per_iter(y[sel], (a + 0.5 * (b - a))[sel], cut[sel])
            for name in CI:
                w = results[name][0]
                mae1, _ = ev.per_iter(y[sel], (a + w[:, None] * (b - a))[sel], cut[sel])
                ci = [ev.block_boot(lambda idx, h=h: 100 * (mae1[idx, h].mean() / mae0[idx, h].mean() - 1), sel.sum(), blk) for h in range(3)]
                print(f"  95% interval of the change, {name}: " + "  ".join(f"{lo:+.0f}..{hi:+.0f}%" for lo, hi in ci))
    # how the learned weight moves over time
    w = results["ls 28d"][0]
    months = cut.astype("datetime64[M]")
    print("--- weight of the gas-relative model per month (ls 28d): " + "  ".join(f"{str(m)[2:]}:{w[months == m].mean():.2f}" for m in np.unique(months)))
