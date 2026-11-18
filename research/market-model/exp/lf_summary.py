"""one line per run: structural + level filter, autumn (all arms) and March (known, curve only and with a 2-day-lag level)"""
import sys, os, subprocess, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
I = np.load(f"{SP}/out/inputs_autumn.npz"); a = I["actual"].astype(float); gas = I["gas"].astype(float); IM = np.load(f"{SP}/out/inputs_march.npz"); am = IM["actual"].astype(float)
def sc(p, a_=a):
    e = np.abs(p - a_); hi = a_[:, :96] > 30; d = (p - a_)[:, :96]
    return f"{e[:, :32].mean():5.2f}{e[:, :56].mean():6.2f}{e[:, 56:96].mean():6.2f} {d[hi].mean() if hi.any() else float('nan'):+5.1f}"
print(f"{'run':<6}" + "".join(f"{arm + ': night 34-48 48-58 >30bias':>30}" for arm in ("known", "tree RL", "Timestral RL", "avg RL")) + "   known >=88 gas: MAE (bias) | March known: curve MAE (bias), +level")
for pair in sys.argv[1:]:
    run, mrun = pair.split(":")
    for cmd in ([f"{SP}/venv/bin/python", "level_filter.py", "autumn", run], [f"{SP}/venv/bin/python", "level_filter.py", "march", mrun]):
        env = dict(os.environ, RKEY="R1" if cmd[2] == "autumn" else "R", ARMS="known,own,ts,avg" if cmd[2] == "autumn" else "known"); subprocess.run(cmd, env=env, capture_output=True)
    P = {arm: np.load(f"{SP}/out/structfc_{run}_{arm}_R1.npz")["vint"].astype(float) for arm in ("known", "own", "ts", "avg")}; m = gas >= 88
    M = np.load(f"{SP}/out/structfc_{mrun}_known_R.npz")
    print(f"{run:<6}" + "".join(f"{sc(P[arm]):>30}" for arm in ("known", "own", "ts", "avg")) + f"   {np.abs(P['known'] - a)[m][:, :56].mean():.2f} ({(P['known'] - a)[m][:, :56].mean():+.2f}) | {np.abs(M['curve'] - am)[:, :96].mean():.2f} ({(M['curve'] - am)[:, :96].mean():+.2f}), {np.abs(M['vint'] - am)[:, :96].mean():.2f}")
