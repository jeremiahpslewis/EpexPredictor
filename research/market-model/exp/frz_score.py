"""Frozen-fit test of the gas-share prior on the candidate's curve: night / first 14 h MAE, curve alone and with the online layer, Timestral loads and published loads."""
import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; rng = np.random.default_rng(0)
def ci(p1, p0, a, sl, blk):
    n = len(a); idxs = [np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(3000)]; e1, e0 = np.abs(p1 - a)[:, sl].mean(1), np.abs(p0 - a)[:, sl].mean(1); r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in idxs]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
for period, ref, blk in (("summer", "hs2", 3), ("autumn", "ha", 7)):
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); sc = lambda v: f"{np.abs(v - a)[:, :32].mean():.2f} / {np.abs(v - a)[:, :56].mean():.2f}"
    for arm in ("ts", "known"):
        L = lambda name, key: np.load(f"{SP}/out/structfc_{name}_{arm}_dyn2xxh.npz")[key].astype(float)
        print(f"\n{period}, {'Timestral loads' if arm == 'ts' else 'published loads'}: MAE ct night / first 14 h      curve alone      with its online layer   | prior against baseline at night: curve alone, with layer")
        print(f"  refitted at every origin (the candidate's curve)   {sc(L(ref, 'curve')):>14s}   {sc(L(ref, 'vint')):>14s}")
        for d, dl in (("j", "frozen 6 Jan"), ("m", "frozen 31 Mar")):
            for st in ("4", "12"):
                sfx = "_s" if period == "summer" else "_a"; h, g = f"zh{st}{d}{sfx}", f"zg{st}{d}{sfx}"
                print(f"  {dl:13s} {st + '000 steps':>11s}: as the candidate    {sc(L(h, 'curve')):>14s}   {sc(L(h, 'vint')):>14s}")
                print(f"  {'':13s} {'':>11s}  with the prior      {sc(L(g, 'curve')):>14s}   {sc(L(g, 'vint')):>14s}   | {ci(L(g, 'curve'), L(h, 'curve'), a, slice(0, 32), blk)}   {ci(L(g, 'vint'), L(h, 'vint'), a, slice(0, 32), blk)}")
    b = {n_: (a - np.load(f"{SP}/out/structfc_{n_}_{period[0]}_ts_dyn2xxh.npz")["curve"].astype(float))[:, :32].mean() for n_ in ("zh4j", "zg4j", "zh12j", "zg12j", "zh4m", "zg4m", "zh12m", "zg12m")}; print(f"  night bias (actual - curve alone, Timestral loads): " + ", ".join(f"{k} {v:+.2f}" for k, v in b.items()))
