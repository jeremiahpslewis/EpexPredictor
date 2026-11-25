"""E1: error anatomy of a stored run (default base_rolling_autumn). usage: anatomy.py [NAME]"""
import sys, numpy as np
import common as c
name = sys.argv[1] if len(sys.argv) > 1 else "base_rolling_autumn"
R = np.load(f"{c.OUT}/{name}.npz"); comps = ("wind", "solar", "load"); sign = {"wind": -1, "solar": -1, "load": 1}
vint = [k[5:] for k in R.files if k.startswith("load_") and k[5:] not in ("true", "prevtrue") and not k[5:].startswith("prev")]
T = {t: R[f"{t}_true"].astype(float) for t in comps}; rl = T["load"] - T["wind"] - T["solar"]
def E(t, k): return R[f"{t}_{k}"].astype(float) - T[t]
def rlf(k): return R[f"load_{k}"] - R[f"wind_{k}"] - R[f"solar_{k}"]
print(f"== {name}: RL MAE GW by test weather (night / day1 / h24-38)")
for k in vint: print(f"  {k:<5}", {a: round(b, 2) for a, b in c.mae_seg(rlf(k), rl).items()})
print("== component MAE GW and bias (forecast - truth), day 1")
for k in vint:
    print(f"  {k:<5}", "  ".join(f"{t} {np.nanmean(np.abs(E(t,k)[:, :96])):.2f} ({np.nanmean(E(t,k)[:, :96]):+.2f})" for t in comps),
          "| night", "  ".join(f"{t} {np.nanmean(np.abs(E(t,k)[:, :24])):.2f}" for t in comps))
for seg in ("night", "day1", "h24_38"):
    s = c.SEG[seg]; print(f"== {seg}: share of RL mean squared error (signed errors; RL err = load - wind - solar errs)")
    for k in vint:
        e = {t: sign[t] * E(t, k)[:, s].ravel() for t in comps}; tot = np.nanmean((e["wind"] + e["solar"] + e["load"]) ** 2)
        parts = {f"{t}": np.nanmean(e[t] ** 2) / tot for t in comps}
        parts.update({f"2cov({a},{b})": 2 * np.nanmean(e[a] * e[b]) / tot for a, b in (("wind", "load"), ("wind", "solar"), ("solar", "load"))})
        sw = {t: np.nanmean(np.abs((rlf(k) - rl)[:, s] - sign[t] * E(t, k)[:, s])) for t in comps}
        print(f"  {k:<5} RMSE {np.sqrt(tot):.2f} |", " ".join(f"{a} {b:+.0%}" for a, b in parts.items()),
              "| RL MAE with one part true:", " ".join(f"{t} {v:.2f}" for t, v in sw.items()), f"(none {np.nanmean(np.abs(rlf(k)-rl)[:, s]):.2f})")
if "d0" in vint:
    print("== weather vs model: MSE(v) = MSE(d0) + MS(shift v-d0) + 2 E[err_d0 * shift]  (RL, GW^2), MAE ratio d0/v")
    for k in [v for v in vint if v != "d0"]:
        for seg in ("night", "day1", "h24_38"):
            s = c.SEG[seg]; e0 = (rlf("d0") - rl)[:, s]; sh = (rlf(k) - rlf("d0"))[:, s]; ev = (rlf(k) - rl)[:, s]
            print(f"  {k:<5} {seg:<7} MSE {np.nanmean(ev**2):5.2f} = d0 {np.nanmean(e0**2):5.2f} + shift {np.nanmean(sh**2):5.2f} + cross {2*np.nanmean(e0*sh):+5.2f}"
                  f" | MAE d0/v {np.nanmean(np.abs(e0)):.2f}/{np.nanmean(np.abs(ev)):.2f} = {np.nanmean(np.abs(e0))/np.nanmean(np.abs(ev)):.0%}"
                  + "  | by comp shift MAE " + " ".join(f"{t} {np.nanmean(np.abs(R[f'{t}_{k}'][:, s]-R[f'{t}_d0'][:, s])):.2f}" for t in comps))
print("== by local hour, day 1 (first 96 qh), MAE GW, test weather", vint[0], "and d0")
H = R["hours"][:, :96]
for k in (vint[0], "d0") if "d0" in vint else (vint[0],):
    print(f"  {k}: hour " + " ".join(f"{h:5d}" for h in range(0, 24, 2)))
    for t in comps + ("rl",):
        e = np.abs(rlf(k) - rl)[:, :96] if t == "rl" else np.abs(E(t, k))[:, :96]
        print(f"   {t:<6}     " + " ".join(f"{np.nanmean(e[(H == h) | (H == h + 1)]):5.2f}" for h in range(0, 24, 2)))
print("== by lead (hours after d0), RL MAE, all vintages: 0-6 / 6-12 / 12-18 / 18-24 / 24-30 / 30-38")
for k in vint:
    e = np.abs(rlf(k) - rl); print(f"  {k:<5}", " ".join(f"{np.nanmean(e[:, a*4:b*4]):.2f}" for a, b in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 38))))
