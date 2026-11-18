import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
rng = np.random.default_rng(0)
def ci(e1, e0, blk):
    n = len(e0); r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(4000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
for lab, a, rl, ts, blk in (("autumn, 66 forecasts", np.load(f"{SP}/out/live_ho.npz")["actual"], np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"], np.load(f"{SP}/out/tsl_123_8k.npz")["vint"], 7),
                            ("spring/summer, 50 forecasts", np.load(f"{SP}/out/ssn_trees.npz")["actual"], np.load(f"{SP}/out/ssn_rl.npz")["vint"], np.load(f"{SP}/out/ssn_ts.npz")["median"], 3)):
    a, rl, ts = a.astype(float), rl.astype(float), ts.astype(float); bl = (rl + ts) / 2
    for win, sl in (("night 00-08", slice(0, 32)), ("34-48 h", slice(0, 56))):
        e0, e1 = np.abs(rl - a)[:, sl].mean(1), np.abs(bl - a)[:, sl].mean(1)
        print(f"{lab:<28}{win:<12} trees + residual load {e0.mean():.2f}  blend {e1.mean():.2f}  change {ci(e1, e0, blk)}")
