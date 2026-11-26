"""Score Timestral variants (SP/C/tsvar/<variant>_<day>.npy) against the stored champion on the fixed screening list,
Timestral alone and in the equal-thirds blend (trees, market from the champion npz). 61-window and fresh halves apart."""
import glob, os, sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
def champ(day):
    for d in ("out_strict", "out_fresh_strict"):
        f = f"{SP}/faithful/{d}/{day}.npz"
        if os.path.exists(f): return np.load(f, allow_pickle=True)
vs = sorted({os.path.basename(f).split("_20")[0] for f in glob.glob(f"{SP}/C/tsvar/*.npy")})
days = sorted({os.path.basename(f)[-14:-4] for f in glob.glob(f"{SP}/C/tsvar/base_*.npy")})
for half, sel in (("61", [d for d in days if d <= "2026-11-07"]), ("fresh", [d for d in days if d >= "2026-11-09"])):
    if not sel: continue
    print(f"== {half}: {len(sel)} days {','.join(d[5:] for d in sel)}")
    for v in vs:
        r = []
        for d in sel:
            f = f"{SP}/C/tsvar/{v}_{d}.npy"
            if not os.path.exists(f): break
            z = champ(d); y = np.load(f); a = z["actual"]
            r.append((np.nanmean(np.abs(y - a)), np.nanmean(np.abs(z["timestral"] - a)),
                      np.nanmean(np.abs((z["trees"] + y + z["market"]) / 3 - a)), np.nanmean(np.abs(z["issued_eval"] - a))))
        if len(r) < len(sel): print(f"  {v:10s} incomplete ({len(r)}/{len(sel)})"); continue
        r = np.array(r); dts = r[:, 0] - r[:, 1]; db = r[:, 2] - r[:, 3]
        print(f"  {v:10s} Timestral {r[:,0].mean():.3f} (champ {r[:,1].mean():.3f}, diff {dts.mean():+.3f}, wins {np.sum(dts<0)}/{len(dts)})  blend {r[:,2].mean():.3f} (champ {r[:,3].mean():.3f}, diff {db.mean():+.3f})")
