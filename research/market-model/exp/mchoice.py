"""Compare ways of handling the gas price (stage-1 models): raw target, gas-relative target, their plain average,
and the encompassing model price = A(x) + gas * B(x)."""
import sys

import numpy as np

import evaluate as ev

out, prefix = sys.argv[1], sys.argv[2]
zones = sys.argv[3].split(",")
names = {"raw": f"{prefix}_raw", "gas-relative": f"{prefix}_ratio", "encompassing": f"{prefix}_vc"}
z = {k: np.load(f"{out}/{v}.npz") for k, v in names.items()}
cut = z["raw"]["cutoffs"]
for hours, label in [((0, 8), "00-08"), (None, "all hours")]:
    ev.HOURS = hours
    print(f"--- {label}: MAE ct/kWh 1/2/3 days ahead (change vs raw target)")
    for zone in zones:
        actual = z["raw"][f"actual_{zone}"]
        preds = {k: v[f"s1_{zone}"].astype(np.float64) for k, v in z.items()}
        preds["average of both"] = (preds["raw"] + preds["gas-relative"]) / 2
        base = None
        for k in ["raw", "gas-relative", "average of both", "encompassing"]:
            mae, mse = ev.per_iter(actual, preds[k], cut)
            rmse, m = ev.summarize(mae, mse)
            if base is None:
                base = m
            print(f"{zone:<3} {k:<16} {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  " + "/".join(f"{100*(m[h]/base[h]-1):+.0f}%" for h in range(3)) + f"   RMSE {rmse[0]:.2f}/{rmse[1]:.2f}/{rmse[2]:.2f}")
