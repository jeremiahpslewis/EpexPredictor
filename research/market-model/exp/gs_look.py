"""Reachability of the gas-share prior on the candidate's curve: per fit, the achieved share against its target (in prior sd), b and h at the four points, in-sample error."""
import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; X = np.array([20.0, 30.0, 42.5, 60.0]); M = np.array([0.78, 0.91, 0.95, 0.97]); SD = np.array([0.05, 0.02, 0.02, 0.02]); eua = np.load(f"{SP}/out/eua_15min.npz")["eua"]
import pandas as pd, live_common as lc
idx = lc.load()["DE"].index
for tag in ("2026-01-06", "2026-03-31", "2026-09-18"):
    d0 = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(tag, tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)))); c = float(np.nanmean(eua[d0 - 540 * 96:d0 - 96]))
    for name in sys.argv[1:]:
        z = np.load(f"{SP}/out/struct_{name}/{tag}.npz"); g = float(z["gref"]); b = np.interp(X, z["grid"], z["b_curve"]); h = np.interp(X, z["grid"], z["h_curve"]); k = 1.85 * g / (1.85 * g + 0.37 * c); sh = g * h / (b + g * h); m = z["misc"]
        print(f"{tag} {name:5s}: window gas {g:.0f} carbon {c:.0f} | share at 20 / 30 / 42.5 / 60 GW " + " ".join(f"{v:.2f}" for v in sh) + " (target " + " ".join(f"{v:.2f}" for v in M * k) + "; off by " + " ".join(f"{v:+.1f}" for v in (sh - M * k) / (SD * k)) + " sd) | b " + " ".join(f"{v:.0f}" for v in b) + " | h " + " ".join(f"{v:.2f}" for v in h) + f" | in-sample MAE {m[6] / 10:.3f} ct, last 30 d {m[7] / 10:.2f} | loss {m[3]:.1f}")
