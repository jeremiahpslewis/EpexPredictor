"""Builds the load-forecast files of the midnight issue from the Timestral outputs (notes 21.29): gen_mid_<period>.npz and d1_mid_<period>.npz
(the day before the target is published, so its "forecast" is the published figure)."""
import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; Z = [np.load(f"{SP}/out/tsgen_mid_{per}/{t}.npz") for t in tags]; names = [str(x) for x in Z[0]["names"]]; T = {n: np.stack([z["median"][i] for z in Z]).astype(float) for i, n in enumerate(names)}; TR = {n: np.stack([z["true"][i] for z in Z]).astype(float) for i, n in enumerate(names)}
    for X in (T, TR): X["rl"] = X["load"] - X["wind"] - X["solar"]
    old = np.load(f"{SP}/out/gen_ts_{per}.npz"); d1 = np.load(f"{SP}/out/d1_ts_{per}.npz"); sl = slice(0, 24); e = lambda p: np.abs(p - TR["rl"])[:, sl]
    print(f"{per}: residual-load forecast error 00:00-06:00, mean abs GW: issued 14:00 two days before {e(old['rl_pes'].astype(float)).mean():.2f} (worst night {e(old['rl_pes'].astype(float)).mean(1).max():.1f}); issued at midnight {e(T['rl']).mean():.2f} (worst night {e(T['rl']).mean(1).max():.1f}); check of the target series against the old file {np.abs(TR['rl'] - old['rl_true']).max():.3f}")
    np.savez_compressed(f"{SP}/out/gen_mid_{per}.npz", cutoffs=I["cutoffs"], **{f"{n}_pes": T[n].astype(np.float32) for n in T}, **{f"{n}_true": TR[n].astype(np.float32) for n in TR}); np.savez_compressed(f"{SP}/out/d1_mid_{per}.npz", cutoffs=I["cutoffs"], rl_d1=d1["rl_d1_true"], rl_d1_true=d1["rl_d1_true"])
