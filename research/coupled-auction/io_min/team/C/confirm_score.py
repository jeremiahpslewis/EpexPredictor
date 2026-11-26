"""Score the confirmation wave: base (fc_D of the dump, must equal the stored champion Timestral) vs nbpast replay.
Paired day diffs, 90% circular 7-day block bootstrap (distill/common.boot), per-month signs; Timestral and blend."""
import glob, os, sys, numpy as np
sys.path.insert(0, "/home/user/EpexPredictor/research/coupled-auction/io_min/distill")
from common import boot
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
pre = [l for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "CONFIRM_DAYS.md")) if l.startswith("- proc")]
days = sorted(sum([l.split(":", 1)[1].strip().split(",") for l in pre], []))
rows, missing, invalid = [], [], []
for d in days:
    fz = glob.glob(f"{SP}/C/tsdump/confirm*/{d}.npz"); fv = glob.glob(f"{SP}/C/tsvar_confirm*/nbpast_{d}.npy")
    if not fz or not fv: missing.append(d); continue
    z = np.load(fz[0]); st = np.load(glob.glob(f"{SP}/faithful/out_*strict/{d}.npz")[0]); a = st["actual"]
    if np.nanmax(np.abs(z["fc_D"] - st["timestral"])) > 1e-9: invalid.append(d); continue
    y = np.load(fv[0]); bl = (st["trees"] + y + st["market"]) / 3
    rows.append((d, np.nanmean(abs(st["timestral"] - a)), np.nanmean(abs(y - a)), np.nanmean(abs(st["issued_eval"] - a)), np.nanmean(abs(bl - a))))
print(f"pre-registered {len(days)}; scored {len(rows)}; missing {missing}; invalid (base != champion) {invalid}")
r = np.array([x[1:] for x in rows]); dts = r[:, 1] - r[:, 0]; dbl = r[:, 3] - r[:, 2]
for nm, x, base in (("Timestral", dts, r[:, 0]), ("blend", dbl, r[:, 2])):
    b = boot(x); print(f"{nm}: champion {base.mean():.3f} -> {base.mean() + x.mean():.3f}, diff {x.mean():+.3f}, 90% [{b[0]:+.3f}, {b[1]:+.3f}], days better {np.sum(x < 0)}/{len(x)}")
    mon = {}
    for (d, *_), v in zip(rows, x): mon.setdefault(d[:7], []).append(v)
    print("   per month:", {m: f"{np.mean(v):+.3f} (n={len(v)})" for m, v in sorted(mon.items())})
for half, sel in (("61-window", [i for i, x in enumerate(rows) if x[0] <= "2026-11-07"]), ("fresh", [i for i, x in enumerate(rows) if x[0] >= "2026-11-09"])):
    print(f"  {half}: Timestral {dts[sel].mean():+.3f}, blend {dbl[sel].mean():+.3f} (n={len(sel)})")
