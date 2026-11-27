"""One walk-forward wave on the 16 days of sample_wf16.txt (team E): 4 processes x 4 days, 1 thread each, every
process under `timeout 1140` (19 min; a day not finished is dropped, not topped up). Days are dealt round-robin over the
date-sorted list, so each process gets one day per quarter.

  champion: faithful_bt.py --rl-variant base                 -> SP/E/wf16/champion/<D>.npz
  matched : faithful_bt.py --rl-variant matched, FAITHFUL_RL_LEAN=1 -> SP/E/wf16/matched/<D>.npz
  nbpast  : team/C/dump_ts.py --de-only --context 8192 --tag wf16 --also nbpast (C's hook) -> SP/C/tsdump/wf16/<D>.npz,
            SP/C/tsvar_wf16/nbpast_<D>.npy; then `assemble` writes SP/E/wf16/nbpast/<D>.npz = the champion's file with
            timestral := nbpast and issued := (trees + nbpast + market) / 3, after checking that C's base Timestral
            equals the champion's.
All with FAITHFUL_VINTAGE_EXTRA=SP/wf16_hist (backfilled vintages; local values win).

usage: venv/bin/python wave_wf16.py run champion|matched|nbpast      (blocks until the wave ends)
       venv/bin/python wave_wf16.py assemble                         (nbpast challenger files from C's outputs)
"""
import os
import subprocess
import sys
import time

import numpy as np

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
HERE = os.path.dirname(os.path.abspath(__file__))
FAITHFUL = os.path.abspath(f"{HERE}/../../faithful/faithful_bt.py")
DUMP_TS = os.path.abspath(f"{HERE}/../C/dump_ts.py")
PY = f"{SP}/tsvenv/bin/python"
OUT = f"{SP}/E/wf16"
DAYS = [x.split()[0] for x in open(f"{HERE}/sample_wf16.txt") if x.strip()]


def run(kind):
    groups = [DAYS[i::4] for i in range(4)]
    env = dict(os.environ, SP=SP, OMP_NUM_THREADS="1", FAITHFUL_VINTAGE_EXTRA=f"{SP}/wf16_hist")
    if kind == "matched":
        env["FAITHFUL_RL_LEAN"] = "1"
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    procs = []
    t0 = time.time()
    for i, g in enumerate(groups):
        if kind in ("champion", "matched"):
            cmd = [PY, FAITHFUL, "--days", ",".join(g), "--rl-variant", "base" if kind == "champion" else "matched",
                   "--out", f"{OUT}/{kind}", "--torch-threads", "1", "--vintage-extra", f"{SP}/wf16_hist"]
        else:
            cmd = [PY, DUMP_TS, "--days", ",".join(g), "--de-only", "--context", "8192", "--tag", "wf16", "--also", "nbpast"]
        procs.append(subprocess.Popen(["timeout", "1140"] + cmd, env=env, cwd=os.path.dirname(cmd[1]),
                                      stdout=open(f"{OUT}/logs/{kind}_{i}.log", "w"), stderr=subprocess.STDOUT))
        print(f"proc {i}: {','.join(g)}", flush=True)
    for i, p in enumerate(procs):
        p.wait()
        print(f"proc {i} exit {p.returncode} after {time.time() - t0:.0f}s", flush=True)
    print(f"wave {kind}: {time.time() - t0:.0f}s wall", flush=True)


def assemble():
    os.makedirs(f"{OUT}/nbpast", exist_ok=True)
    n_ok, dev = 0, []
    for d in DAYS:
        fc, fd, fv = f"{OUT}/champion/{d}.npz", f"{SP}/C/tsdump/wf16/{d}.npz", f"{SP}/C/tsvar_wf16/nbpast_{d}.npy"
        if not all(os.path.exists(f) for f in (fc, fd, fv)):
            print(f"{d}: missing input, skipped")
            continue
        z, dz = dict(np.load(fc)), np.load(fd)
        base_dev = float(np.max(np.abs(dz["fc_D"] - z["timestral"])))
        dev.append(base_dev)
        assert base_dev < 1e-6, f"{d}: C's base Timestral differs from the champion's by {base_dev}"
        nb = np.asarray(np.load(fv), float).reshape(-1)
        nb = nb[:len(z["idx"])] if len(nb) >= len(z["idx"]) else nb
        assert len(nb) == len(z["idx"]) and np.isfinite(nb).all(), f"{d}: nbpast length {len(nb)}"
        z["timestral"] = nb
        z["issued"] = (z["trees"] + nb + z["market"]) / 3
        np.savez(f"{OUT}/nbpast/{d}.npz", **z)
        n_ok += 1
    print(f"assembled {n_ok}/{len(DAYS)} days; base Timestral vs champion max |diff| {max(dev) if dev else 'n/a'}")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "assemble": assemble}[sys.argv[1]]()
