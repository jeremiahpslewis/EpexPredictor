"""Stage-only check of faithful_bt staging (no training; team E). Saves the staged weather (8 zones + DE grid, gate..D+2)
and the cut price/generation tails for each day to OUT/<D>.npz, so two configurations can be compared bit for bit.

  prep:  venv-free; run with tsvenv:  stage_check.py prep --day D --dir X    (writes X/prev_<zone>.parquet and
         X/grid_prev.parquet from the DEFAULT local vintages on D-1..D+2, in the --vintage-extra formats)
  stage: stage_check.py stage --days D[,D..] --out OUT   (env decides the config, e.g. FAITHFUL_VINTAGE_EXTRA=X
         FAITHFUL_VINTAGE_EXTRA_ONLY=1 makes the extra files REPLACE the local ones)
  cmp:   stage_check.py cmp OUT_A OUT_B
"""
import argparse
import asyncio
import glob
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{HERE}/../../faithful"))


def cmd_cmp(a_dir, b_dir):
    worst = 0.0
    for fa in sorted(glob.glob(f"{a_dir}/*.npz")):
        za, zb = np.load(fa), np.load(f"{b_dir}/{os.path.basename(fa)}")
        assert sorted(za.files) == sorted(zb.files), (fa, set(za.files) ^ set(zb.files))
        for k in za.files:
            x, y = za[k], zb[k]
            assert x.shape == y.shape, (k, x.shape, y.shape)
            same_nan = np.array_equal(np.isnan(x), np.isnan(y))
            d = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
            worst = max(worst, d)
            if not same_nan or d > 0:
                print(f"{os.path.basename(fa)} {k}: max|diff| {d}, NaN pattern equal {same_nan}")
    print(f"compared {len(glob.glob(f'{a_dir}/*.npz'))} days, worst max|diff| {worst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "stage", "cmp"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--day")
    ap.add_argument("--days")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "cmp":
        return cmd_cmp(*a.args)
    import faithful_bt as F
    if a.cmd == "prep":
        d = date.fromisoformat(a.day)
        lo, hi = pd.Timestamp(d - timedelta(days=1), tz="UTC"), pd.Timestamp(d + timedelta(days=4), tz="UTC")
        v = F.load_vintages()
        os.makedirs(a.dir, exist_ok=True)
        for zone, f in F.ZONE_FILE.items():
            parts = [v[zone][k].loc[lo:hi].rename(columns=lambda c, k=k: f"{c}_d{k}") for k in (1, 2, 3)]
            pd.concat(parts, axis=1).to_parquet(f"{a.dir}/prev_{f}.parquet")
        inv = {"wind": "w120", "radiation": "sw", "temp": "t2"}
        g = [v["GRID"][k].loc[lo:hi].rename(columns=lambda c, k=k: f"{inv[c.split('_')[0]]}_{c.split('_')[1]}_d{k}") for k in (1, 2, 3)]
        pd.concat(g, axis=1).to_parquet(f"{a.dir}/grid_prev.parquet")
        print("prepared", a.dir)
        return

    class Args:
        vintage, latency, rl_variant, out = "strict", 6.0, "base", a.out

    async def run():
        h = F.Harness(Args())
        await h.setup()
        os.makedirs(a.out, exist_ok=True)
        for ds in a.days.split(","):
            d = date.fromisoformat(ds)
            gate, cutoff, info = h.stage_day(d)
            end = F.day_index(d + timedelta(days=2))[-1]
            res = {}
            for name, p in h.bases.items():
                res[f"w_{name.value}"] = p.weatherstore.data.loc[gate:end].to_numpy(float)
                res[f"p_{name.value}"] = p.pricestore.data["price"].iloc[-200:].to_numpy(float)
            res["grid"] = h.rl.gridweather.data.loc[gate:end].to_numpy(float)
            res["gen"] = h.rl.generation.data.iloc[-200:].to_numpy(float)
            np.savez(f"{a.out}/{d}.npz", **res)
            print(d, {k: v["fallback_cells"] if "fallback_cells" in v else v for k, v in info["weather"].items()}, flush=True)
    asyncio.run(run())


if __name__ == "__main__":
    main()
