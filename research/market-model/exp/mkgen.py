import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; name = sys.argv[1]; I = np.load(f"{SP}/out/inputs_{name}.npz")
for arm in ("known", "own"):
    if arm == "own" and "rl_pes" not in I.files: continue
    np.savez_compressed(f"{SP}/out/genarm_{name}_{arm}.npz", cutoffs=I["cutoffs"], rl_pes=I["rl_true"] if arm == "known" else I["rl_pes"], rl_true=I["rl_true"])
