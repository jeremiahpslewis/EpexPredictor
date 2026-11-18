"""Within-hour scale for the merit-order forecasts (notes 21.30). The rules: hourly orders are flat across the four quarters, so within the hour only
part of supply follows and the price swings further than the hourly curves say. forecast_q = hour mean + k (forecast_q - hour mean); k is estimated
walk-forward from the origins BEFORE each one (all 152 full-hour slots of their horizons), prior value 2.5 until 5 origins exist. usage: m1_within.py RUN"""
import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; RUN = sys.argv[1]; hist_f, hist_a = [], []; H = 152
dev = lambda v: v[:, :H] - v[:, :H].reshape(len(v), -1, 4).mean(2).repeat(4, 1)
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float); z = np.load(f"{SP}/out/structfc_{RUN}_{per}_price.npz"); out = {}; ks = []
    for key in ("vint_ts",):
        f = z[key].astype(float); df, da = dev(f), dev(a); g = f.copy()
        for j in range(len(f)):
            if key == "vint_ts":
                k = float(np.sum(np.concatenate(hist_a) * np.concatenate(hist_f)) / np.sum(np.concatenate(hist_f) ** 2)) if len(hist_f) >= 5 else 2.5; ks.append(k); hist_f.append(df[j]); hist_a.append(da[j])
            else: k = ks_ts[j]
            g[j, :H] = f[j, :H] - df[j] + k * df[j]
        if key == "vint_ts": ks_ts = ks
        out[key] = g
    # the ts arm defines k; redo the known arm with the same k
    f = z["vint"].astype(float); out["vint"] = f.copy(); out["vint"][:, :H] = f[:, :H] - dev(f) + np.array(ks_ts)[:, None] * dev(f)
    np.savez_compressed(f"{SP}/out/structfc_{RUN}w_{per}_price.npz", vint=out["vint"].astype(np.float32), vint_ts=out["vint_ts"].astype(np.float32), k=np.array(ks_ts)); e = lambda v: np.abs(v - a)[:, :24]
    print(f"{per}: k from the past {min(ks_ts):.1f}..{max(ks_ts):.1f} (median {np.median(ks_ts):.1f}) | MAE 00:00-06:00: {e(z['vint_ts'].astype(float)).mean():.2f} -> {e(out['vint_ts']).mean():.2f}; first quarter-hour of the day {e(z['vint_ts'].astype(float))[:, 0].mean():.2f} -> {e(out['vint_ts'])[:, 0].mean():.2f}; with the true within-hour shape it would be {np.abs((z['vint_ts'].astype(float) - a)[:, :24].reshape(len(a), 6, 4).mean(2)).mean():.2f}")
