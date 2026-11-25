"""DESCRIPTIVE ONLY (notes 21.121: no further Timestral weighting form is to be decided on these sets).
The deployed lead-weighted mix (Timestral weight 1 -> 0.5 over 96 qh, 0.5 after) with the stored Timestral load forecasts
(gen_v2 = 2-day-old weather, strict; gen_mid = 1-day-old, optimistic) and tree candidates on gate weather.
usage: mix.py PROTO cand [cand ...]"""
import sys, os, numpy as np
import common as c
proto = sys.argv[1]; cands = sys.argv[2:]
TS = {k: np.load(f"{c.SP}/out/gen_{k}_autumn.npz") for k in ("v2", "mid")}
a = np.arange(c.HZ); w = (1.0 + (0.5 - 1.0) * np.clip(a / 96, 0, 1))[None]
B = np.load(f"{c.OUT}/base_{proto}_autumn.npz"); T = B["load_true"] - B["wind_true"] - B["solar_true"]
assert np.abs(T - TS["v2"]["rl_true"]).max() < 0.02
def rl(R, k="gate"): return R[f"load_{k}"] - R[f"wind_{k}"] - R[f"solar_{k}"]
base_mix = w * TS["v2"]["rl_pes"] + (1 - w) * rl(B)
print(f"== {proto}: RL MAE GW night / day1 / h24-38; diff vs (base trees + Timestral v2, LW mix), paired bootstrap")
for k in ("v2", "mid"): print(f"  Timestral alone {k:<4}", {s: round(v, 2) for s, v in c.mae_seg(TS[k]["rl_pes"], T).items()})
near = (np.arange(c.HZ) < c.GATE_D1)[None]
for n in cands:
    if n == "VM1":
        Rb, R1 = np.load(f"{c.OUT}/base_{proto}_autumn.npz"), np.load(f"{c.OUT}/trd1_{proto}_autumn.npz")
        R = {f"{t}_gate": np.where(near, Rb[f"{t}_gate"], R1[f"{t}_gate"]) for t in ("wind", "solar", "load")}
    else:
        f = f"{c.OUT}/{n}_{proto}_autumn.npz"
        if not os.path.exists(f): continue
        R = np.load(f)
    for k in ("v2", "mid"):
        m = w * TS[k]["rl_pes"] + (1 - w) * rl(R); mm = c.mae_seg(m, T)
        print(f"  LW mix {n:<10} TS {k:<4}", "  ".join(f"{s} {mm[s]:.2f} ({c.boot(m, base_mix, T, s)[0]:+.2f} [{c.boot(m, base_mix, T, s)[1]:+.2f},{c.boot(m, base_mix, T, s)[2]:+.2f}])" for s in c.SEG))
