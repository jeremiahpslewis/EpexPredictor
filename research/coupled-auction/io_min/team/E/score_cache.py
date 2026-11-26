"""Scorer for the shared walk-forward cache (team E, neutral). Seconds per call.

Compares a challenger with the champion on the days both have (per-day npz in the faithful contract: idx, actual and
the member arrays). Reports per member (default trees, timestral, market, issued = blend):
  - window MAE (mean over days of the daily MAE over all quarter hours, ct/kWh), rMAE vs naive (same local quarter
    hour one week earlier, from the app's price store);
  - paired difference challenger - champion with a paired circular 7-day block bootstrap, two-sided 90 %
    (distill/common.boot; days in date order);
  - by month: champion, challenger, difference, and the count of months the challenger wins;
  - the promotion rule of TEAM_PLAN: wins >= 9 of 12 months AND the 90 % interval excludes 0 (shown for the blend and
    each member; with fewer than 12 months present it says so).
A challenger file may hold any subset of members; a member missing there is skipped. `--challenger-key` maps one array
name of the challenger onto a member, e.g. --map issued=blend_new.

usage: venv/bin/python score_cache.py --champion DIR --challenger DIR [--days FILE] [--map member=array ...] [--md OUT.md]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{HERE}/../../distill"))
sys.path.insert(0, HERE)
from common import boot  # noqa: E402
from naive import naive_for, price_series  # noqa: E402

MEMBERS = ("trees", "timestral", "market", "issued")


def idx_of(z):
    v = z["idx"]
    return pd.DatetimeIndex(pd.to_datetime(v, unit="us" if v[0] < 1e17 else "ns", utc=True))


def day_maes(dirname, days, keymap=None):
    keymap = keymap or {}
    rows = {}
    for d in days:
        fn = f"{dirname}/{d}.npz"
        if not os.path.exists(fn):
            continue
        z = np.load(fn, allow_pickle=False)
        a = z["actual"]
        r = {"_idx": idx_of(z), "_actual": a}
        for m in MEMBERS:
            k = keymap.get(m, m)
            if k in z.files and np.isfinite(z[k]).all():
                r[m] = float(np.mean(np.abs(z[k] - a)))
        rows[d] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--days", help="file with one ISO date per line (default: all days in both)")
    ap.add_argument("--map", nargs="*", default=[], help="member=array in the challenger files")
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.days:
        lines = [x.split("\t") for x in open(a.days) if x.strip()]
        days = [x[0].strip() for x in lines]
        group = {x[0].strip(): x[1].strip() for x in lines if len(x) > 1}   # e.g. the quarter of sample_wf16.txt
    else:
        group = {}
        days = sorted(set(os.path.basename(f)[:10] for f in glob.glob(f"{a.champion}/*.npz"))
                      & set(os.path.basename(f)[:10] for f in glob.glob(f"{a.challenger}/*.npz")))
    ch = day_maes(a.champion, days)
    cl = day_maes(a.challenger, days, dict(x.split("=") for x in a.map))
    both = [d for d in days if d in ch and d in cl]
    missing = [d for d in days if d not in both]
    price = price_series()
    naive = {}
    for d in both:
        assert np.allclose(ch[d]["_actual"], cl[d]["_actual"], equal_nan=True), f"{d}: actual prices differ between runs"
        nv = naive_for(ch[d]["_idx"], price)
        naive[d] = float(np.mean(np.abs(nv - ch[d]["_actual"])))
    nv_mae = np.mean(list(naive.values()))
    out = [f"# {os.path.basename(a.challenger.rstrip('/'))} vs champion {os.path.basename(a.champion.rstrip('/'))}", "",
           f"Days scored: {len(both)} ({both[0]}..{both[-1]})" + (f"; missing in a run: {len(missing)} ({', '.join(missing)})" if missing else ""),
           f"Naive MAE (same local QH, D-7): {nv_mae:.3f} ct/kWh. Differences: challenger - champion, negative = better; "
           "paired circular 7-day block bootstrap, 90 %.", "",
           "| member | champion | challenger | rMAE champ / chall | diff [90%] | months won | promote? |", "|---|---|---|---|---|---|---|"]
    bymonth = {}
    for m in MEMBERS:
        dd = [d for d in both if m in ch[d] and m in cl[d]]
        if not dd:
            continue
        c = np.array([ch[d][m] for d in dd])
        x = np.array([cl[d][m] for d in dd])
        diff = x - c
        lo, hi = boot(diff)
        mon = pd.DataFrame({"c": c, "x": x}, index=pd.Index([d[:7] for d in dd], name="month")).groupby("month").mean()
        mon["diff"] = mon.x - mon.c
        bymonth[m] = mon
        wins, nmon = int((mon["diff"] < 0).sum()), len(mon)
        promote = (wins >= 9 and hi < 0) if nmon >= 12 else f"n/a ({nmon} months)"
        out.append(f"| {m if m != 'issued' else 'blend'} | {c.mean():.3f} | {x.mean():.3f} | {c.mean() / nv_mae:.3f} / {x.mean() / nv_mae:.3f} | "
                   f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {wins}/{nmon} | {promote} |")
    out += ["", "## By month (champion / challenger / diff)", "", "| month | n | " + " | ".join(m if m != "issued" else "blend" for m in bymonth) + " |",
            "|---|---|" + "---|" * len(bymonth)]
    months = sorted(set().union(*[set(v.index) for v in bymonth.values()]))
    for mo in months:
        n = sum(1 for d in both if d.startswith(mo))
        cells = [f"{v.loc[mo, 'c']:.2f} / {v.loc[mo, 'x']:.2f} / {v.loc[mo, 'diff']:+.2f}" if mo in v.index else "" for v in bymonth.values()]
        out.append(f"| {mo} | {n} | " + " | ".join(cells) + " |")
    if group:
        out += ["", "## By group of the days file (e.g. quarter): mean diff challenger - champion, days, sign", "",
                "| group | n | " + " | ".join(m if m != "issued" else "blend" for m in bymonth) + " |", "|---|---|" + "---|" * len(bymonth)]
        for gname in sorted(set(group[d] for d in both if d in group)):
            gd = [d for d in both if group.get(d) == gname]
            cells = []
            for m in bymonth:
                diffs = [cl[d][m] - ch[d][m] for d in gd if m in ch[d] and m in cl[d]]
                cells.append(f"{np.mean(diffs):+.3f}" if diffs else "")
            out.append(f"| {gname} | {len(gd)} | " + " | ".join(cells) + " |")
        for m in bymonth:
            signs = [np.mean([cl[d][m] - ch[d][m] for d in both if group.get(d) == g and m in cl[d]]) < 0
                     for g in sorted(set(group.values()))]
            out.append(f"- {m if m != 'issued' else 'blend'}: negative in {sum(signs)} of {len(signs)} groups")
    txt = "\n".join(out) + "\n"
    print(txt)
    if a.md:
        open(a.md, "w").write(txt)


if __name__ == "__main__":
    main()
