"""RL MAE by 6-h lead bins (hours after local midnight of D), rolling autumn, for base, base + load decay, VM + load decay, VM1 + load decay."""
import sys, numpy as np
import common as c
g = sys.argv[1] if len(sys.argv) > 1 else "gate12"; L = lambda n: np.load(f"{c.OUT}/{n}_rolling_autumn.npz"); B, R2, R1 = L("base"), L("trd2"), L("trd1")
lead = np.arange(c.HZ) / 4.0; near = lead < (8 if g == "gate8" else 12)
def comp(R, t, k): return R[f"{t}_{k}"].astype(float)
def fc(Rn, Rf, corr):
    out = 0
    for t, s in (("load", 1), ("wind", -1), ("solar", -1)):
        if g == "interp": f = c.gate_mix(comp(Rn, t, "d1"), comp(Rf, t, "d2"), "interp")
        else: f = np.where(near[None], comp(Rn, t, "d1"), comp(Rf, t, "d2"))
        if corr and t == "load": f = f + (B["load_prevtrue"] - Rn["load_prev_d1"])[:, -24:].mean(1, keepdims=True) * np.exp(-(np.arange(c.HZ) + 1) / 96.0)[None]
        out = out + s * f
    return out
T = B["load_true"] - B["wind_true"] - B["solar_true"]
bins = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 38))
print(f"== RL MAE GW by lead (h after local midnight of D; gate at -12 h), weather map {g}")
print(f"{'':<20}" + "".join(f"{a:>5}-{b:<5}" for a, b in bins))
for n, f in (("base", fc(B, B, False)), ("base+decay(load)", fc(B, B, True)), ("VM+decay(load)", fc(B, R2, True)), ("VM1+decay(load)", fc(B, R1, True))):
    e = np.abs(f - T); print(f"{n:<20}" + "".join(f"{e[:, a*4:b*4].mean():>11.2f}" for a, b in bins))
