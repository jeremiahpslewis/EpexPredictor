"""Score screen 3 (SCREEN3_PREREG.md): validity (dump forecast == stored rlmatched Timestral), each variant vs the champion
Timestral (base RL) and in the blend, 61/fresh parts, per-month signs, 90% 7-day block bootstrap (indicative, 12 days).
gasD variants are labelled UPPER BOUND (stored gas-day-D values are post-gate)."""
import glob, os, sys, numpy as np
sys.path.insert(0, "/home/user/EpexPredictor/research/coupled-auction/io_min/distill")
from common import boot
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
pre = [l for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCREEN3_PREREG.md")) if l.startswith("- proc")]
days = sorted(sum([l.split(":", 1)[1].strip().split(",") for l in pre], []))
res = {}; invalid, missing = [], []
for d in days:
    fz = f"{SP}/C/tsdump/screen3/{d}.npz"
    if not os.path.exists(fz): missing.append(d); continue
    z = np.load(fz); st = np.load(glob.glob(f"{SP}/faithful/out_*strict/{d}.npz")[0]); sm = np.load(glob.glob(f"{SP}/faithful/out_*strict_rlmatched/{d}.npz")[0])
    if np.nanmax(np.abs(z["fc_D"] - sm["timestral"])) > 1e-9: invalid.append(d); continue
    a = st["actual"]; F = {"rlmH": z["fc_D"]}
    for v in ("nbpast", "gasD", "nbpast_gasD"):
        f = f"{SP}/C/tsvar_screen3/{v}_{d}.npy"
        if os.path.exists(f): F[v] = np.load(f)
    c_ts = np.nanmean(abs(st["timestral"] - a)); c_bl = np.nanmean(abs(st["issued_eval"] - a))
    for v, y in F.items():
        res.setdefault(v, []).append((d, np.nanmean(abs(y - a)) - c_ts, np.nanmean(abs((st["trees"] + y + st["market"]) / 3 - a)) - c_bl, np.nanmean(abs(y - a))))
print(f"pre-registered {len(days)}; missing {missing}; invalid {invalid}")
for v, r in res.items():
    ds = [x[0] for x in r]; t = np.array([x[1] for x in r]); b = np.array([x[2] for x in r]); m = np.array([x[3] for x in r])
    i61 = [i for i, d in enumerate(ds) if d <= "2026-11-07"]; ifr = [i for i, d in enumerate(ds) if d >= "2026-11-09"]
    mon = {}
    for d, x in zip(ds, t): mon.setdefault(d[:7], []).append(x)
    lab = " (UPPER BOUND: post-gate gas)" if "gasD" in v else ""
    print(f"{v}{lab}: n={len(r)} Timestral MAE {m.mean():.3f}, diff vs champion {t.mean():+.3f} {np.round(boot(t),3)} better {np.sum(t<0)}/{len(t)}; "
          f"61 {t[i61].mean():+.3f} fresh {t[ifr].mean():+.3f}; blend {b.mean():+.3f} {np.round(boot(b),3)}; months {({k: round(float(np.mean(x)),3) for k, x in mon.items()})}")
