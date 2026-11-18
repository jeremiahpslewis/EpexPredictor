"""Run a list of harness variants sequentially. Usage: batch.py BATCHFILE [common harness args...]
Each non-empty line of BATCHFILE: NAME | extra harness args (shell-like). Variants whose output exists are skipped."""
import os
import shlex
import subprocess
import sys

SP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
batchfile, common = sys.argv[1], sys.argv[2:]
out_dir = common[common.index("--out") + 1]
for line in open(batchfile):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, _, extra = line.partition("|")
    name = name.strip()
    if os.path.exists(f"{out_dir}/{name}.npz"):
        print(f"{name}: exists, skipping", flush=True)
        continue
    cmd = [sys.executable, os.path.join(SP, "exp", "harness.py"), "--name", name] + common + shlex.split(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    print(tail if r.returncode == 0 else f"{name}: FAILED\n{r.stderr[-1500:]}", flush=True)
