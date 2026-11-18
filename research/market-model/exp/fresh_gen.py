"""gen_mid_<period>.npz for the fresh nights from the Timestral outputs (as mid_build.py, without the comparison to the 14:00 files)."""
import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for per in sys.argv[1:]:
    I = np.load(f"{SP}/out/inputs_{per}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; Z = [np.load(f"{SP}/out/tsgen_mid_{per}/{t}.npz") for t in tags]; names = [str(x) for x in Z[0]["names"]]; T = {n: np.stack([z["median"][i] for z in Z]).astype(float) for i, n in enumerate(names)}; TR = {n: np.stack([z["true"][i] for z in Z]).astype(float) for i, n in enumerate(names)}
    for X in (T, TR): X["rl"] = X["load"] - X["wind"] - X["solar"]
    assert np.abs(TR["rl"] - I["rl_true"]).max() < 0.01; np.savez_compressed(f"{SP}/out/gen_mid_{per}.npz", cutoffs=I["cutoffs"], **{f"{n}_pes": T[n].astype(np.float32) for n in T}, **{f"{n}_true": TR[n].astype(np.float32) for n in TR}); print(per, len(tags), "nights; residual-load error 00-06 GW", round(float(np.abs(T["rl"] - TR["rl"])[:, :24].mean()), 2), flush=True)
