"""Timestral vs the tree model for the TSO wind, solar and load figures; writes generation files for the price models.
usage: ts_gen_eval.py RUN PERIOD TREEGEN"""
import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; run, period, treegen = sys.argv[1], sys.argv[2], sys.argv[3]
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; a = I["actual"].astype(float)[:, :96]
Z = [np.load(f"{SP}/out/tsgen_{run}/{t}.npz") for t in tags]; names = [str(x) for x in Z[0]["names"]]
T = {n: np.stack([z["median"][i] for z in Z]).astype(float) for i, n in enumerate(names)}; TR = {n: np.stack([z["true"][i] for z in Z]).astype(float) for i, n in enumerate(names)}
D1 = {n: np.stack([z["d1"][i] for z in Z]).astype(float) for i, n in enumerate(names)}; D1T = {n: np.stack([z["d1_true"][i] for z in Z]).astype(float) for i, n in enumerate(names)}
if "rl" not in T:
    for X in (T, TR, D1, D1T): X["rl"] = X["load"] - X["wind"] - X["solar"]
g = np.load(f"{SP}/out/{treegen}.npz"); G = {n: g[f"{n}_pes"].astype(float) for n in ("wind", "solar", "load", "rl")}
print(f"{period}: {len(tags)} forecasts, first forecast day (34-58 h ahead), error against the published TSO figure in GW")
print(f"{'':<8}{'--- MAE ---':^34}{'--- bias, all hours ---':^30}{'--- bias, price > 30 ct ---':^30}")
print(f"{'':<8}{'trees':>10}{'Timestral':>12}{'average':>10}  {'trees':>10}{'Timestral':>12}{'average':>8}  {'trees':>10}{'Timestral':>12}{'average':>8}")
hi = a > 30
for n in ("wind", "solar", "load", "rl"):
    if n not in T: continue
    t, tr, tt = T[n][:, :96], G[n][:, :96], TR[n][:, :96]; av = (t + tr) / 2
    f = lambda p: np.abs(p - tt).mean(); b = lambda p, m=slice(None): (p - tt)[m].mean()
    print(f"{n:<8}{f(tr):>10.2f}{f(t):>12.2f}{f(av):>10.2f}  {b(tr):>+10.2f}{b(t):>+12.2f}{b(av):>+8.2f}  {b(tr, hi):>+10.2f}{b(t, hi):>+12.2f}{b(av, hi):>+8.2f}")
print("by hours ahead of the last published figure (residual load MAE): " + "  ".join(f"{lab} trees {np.abs(G['rl'][:, s] - TR['rl'][:, s]).mean():.2f} / Timestral {np.abs(T['rl'][:, s] - TR['rl'][:, s]).mean():.2f}" for lab, s in (("24-32 h", slice(0, 32)), ("32-38 h", slice(32, 56)), ("38-48 h", slice(56, 96)), ("48-62 h", slice(96, 153)))))
print(f"day before (D+1, 0-24 h after the last published figure), Timestral: " + "  ".join(f"{n} {np.abs(D1[n] - D1T[n]).mean():.2f}" for n in ("wind", "solar", "load", "rl")))
for nm, src in (("gen_ts", T), ("gen_avg", {n: (T[n] + G[n]) / 2 for n in T})):
    np.savez_compressed(f"{SP}/out/{nm}_{period}.npz", cutoffs=I["cutoffs"], **{f"{n}_pes": src[n].astype(np.float32) for n in src}, **{f"{n}_true": TR[n].astype(np.float32) for n in TR})
np.savez_compressed(f"{SP}/out/d1_ts_{period}.npz", cutoffs=I["cutoffs"], rl_d1=D1["rl"].astype(np.float32), rl_d1_true=D1T["rl"].astype(np.float32))
