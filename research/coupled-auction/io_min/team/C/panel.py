"""Load faithful per-day npz into one panel. Usable on any npz directory in the shared contract."""
import glob, os, json
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
F = f"{SP}/faithful"
MEM = ("trees", "timestral", "market")
NS = 100   # slots per day incl. the autumn clock change; normal days padded with NaN

def pad(v):
    o = np.full(NS, np.nan); o[:len(v)] = v; return o

def day_mae(Fc, a):
    return np.nanmean(np.abs(Fc - a), 1)

def load_dirs(dirs, extra=("core", "rl_mixed", "rl_trees", "rl_timestral")):
    rows = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            rows[os.path.basename(f)[:10]] = f      # later dirs override
    days = sorted(rows)
    out = {k: [] for k in MEM + tuple(extra) + ("actual", "qh", "wd")}
    for day in days:
        z = np.load(rows[day], allow_pickle=True)
        for k in MEM + tuple(extra) + ("actual",):
            v = z[k].astype(float) if k in z.files else np.full(len(z["idx"]), np.nan)
            out[k].append(pad(v))
        t = pd.DatetimeIndex(pd.to_datetime(z["idx"], unit="us", utc=True)).tz_convert("Europe/Berlin")
        out["qh"].append(pad((t.hour * 4 + t.minute // 15).to_numpy().astype(float)))   # DST day: 100 slots, qh repeats
        out["wd"].append(np.full(NS, float(t[48].weekday())))
    P = {k: np.stack(v) for k, v in out.items()}
    P["days"] = np.array(days)
    P["X"] = np.stack([P[k] for k in MEM], -1)    # (D,96,3)
    return P

def champion(fresh=False, matched=False):
    s = "_rlmatched" if matched else ""
    return load_dirs([f"{F}/out_strict{s}", f"{F}/out_fresh_strict{s}"])
