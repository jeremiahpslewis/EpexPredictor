"""Agent P: score screened variants paired against a reference variant and the stored champion member.
Usage: score.py TAG REF [VARIANT,...] [--member trees|market]
Days: those present in every file. MAE ct/kWh over the first 96 slots; 90% circular 7-day block bootstrap."""
import os, sys, glob, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "distill"))
from common import boot  # noqa: E402
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
WIN = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24), (17, 21)]


def stored(day, member):
    for o in ("out_strict", "out_fresh_strict"):
        fn = f"{SP}/faithful/{o}/{day}.npz"
        if os.path.exists(fn):
            return np.asarray(np.load(fn)[member], float)
    return None


def table(F, act, hours, wd, ref):
    rows = {}
    for k, f in F.items():
        r = {"MAE": np.mean([np.abs(f[d] - act[d]).mean() for d in range(len(act))])}
        dd = np.array([np.abs(f[d] - act[d]).mean() - np.abs(F[ref][d] - act[d]).mean() for d in range(len(act))])
        lo, hi = boot(dd)
        r[f"vs {ref}"] = f"{dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        for lo_, hi_ in WIN:
            r[f"{lo_:02d}-{hi_:02d}"] = np.mean([np.abs(f[d] - act[d])[(hours[d] >= lo_) & (hours[d] < hi_)].mean() for d in range(len(act))])
        mon = [d for d in range(len(act)) if wd[d] == 0]
        r["Mon"] = np.mean([np.abs(f[d] - act[d]).mean() for d in mon]) if mon else np.nan
        r["bias"] = np.mean([np.mean(f[d] - act[d]) for d in range(len(act))])
        rows[k] = r
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("ref"); ap.add_argument("variants", nargs="?", default="")
    ap.add_argument("--member", default="trees"); ap.add_argument("--days", default="")
    a = ap.parse_args()
    files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{SP}/P/screen/{a.tag}/*.npz")}
    names = [a.ref] + [v for v in (a.variants.split(",") if a.variants else sorted(files)) if v != a.ref]
    Z = {n: np.load(files[n]) for n in names}
    days = sorted(set.intersection(*[set(z.files) for z in Z.values()]))
    if a.days:
        days = [d for d in days if a.days[:10] <= d <= a.days[-10:]]
    act = [Z[a.ref][d][:96, 2] for d in days]
    idx = [pd.Timestamp(d, tz="Europe/Berlin") + pd.to_timedelta(np.arange(96) * 15, "min") for d in days]
    hours = [i.hour.to_numpy() for i in idx]; wd = [i.weekday[0] for i in idx]
    F = {n: [Z[n][d][:96, 1] for d in days] for n in names}
    F[f"champion {a.member}"] = [stored(d, a.member)[:96] for d in days]
    print(f"{a.tag}: {len(days)} days {days[0]}..{days[-1]}")
    print(table(F, act, hours, wd, a.ref).round(3).to_markdown())


if __name__ == "__main__":
    main()
