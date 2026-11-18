import numpy as np, os
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
rows = [("RL, gas, clock (quarter hour, weekday, holiday)", "Z_rl_gas_clock"), ("  + timestamp", "Y_rl_gas_clock_ts"), ("  + sun position instead (season)", "X_rl_gas_clock_sun"), ("RL, gas, clock, sun, timestamp", "A_rl_gas_cal"), ("  + Europe-wide weather", "B_plus_eu"), ("  + German point weather", "C_plus_de_weather")]
PS = [(P, r) for P, r in (("winter", "hw2"), ("summer", "hs2"), ("autumn", "ha")) if os.path.exists(f"{SP}/out/ladder2_{P}.npz")]
print("trees as analysis tool, published residual load known: MAE ct/kWh night / first 14 h")
print(f"{'':<50}" + "".join(f"{P:>14}" for P, _ in PS))
for lab, k in rows:
    line = f"{lab:<50}"
    for P, _ in PS:
        L = np.load(f"{SP}/out/ladder2_{P}.npz"); a = np.load(f"{SP}/out/inputs_{P}.npz")["actual"].astype(float); e = np.abs(L[f"{k}__known"].astype(float) - a); line += f"{e[:, :32].mean():>8.2f} /{e[:, :56].mean():5.2f}"
    print(line)
for lab, f in (("  + other zones (two stages)", "two_{}_known"), ("structural, curve only", "S:curve"), ("structural + online correction", "S:vint")):
    line = f"{lab:<50}"
    for P, r in PS:
        a = np.load(f"{SP}/out/inputs_{P}.npz")["actual"].astype(float)
        p = np.load(f"{SP}/out/structfc_{r}_known_dyn2xxh.npz")[f.split(":")[1]].astype(float) if f.startswith("S:") else np.load(f"{SP}/out/" + (f.format(P) if P != "autumn" else "two_autumn_known_d1") + ".npz")["vint"].astype(float)
        e = np.abs(p - a); line += f"{e[:, :32].mean():>8.2f} /{e[:, :56].mean():5.2f}"
    print(line)
