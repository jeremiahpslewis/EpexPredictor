"""Spring/summer confirmation, frozen configuration, first 14 h after the last known price (34-48 h after issue)."""
import os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
t = np.load(f"{SP}/out/ssn_trees.npz"); a = t["actual"].astype(float); cut = t["cutoffs"]; trees = t["vint"].astype(float)
C = {"trees, PR #1": trees}
if os.path.exists(f"{SP}/out/ssn_rl.npz"):
    r = np.load(f"{SP}/out/ssn_rl.npz"); assert (r["cutoffs"] == cut).all(); C["trees + residual load (PR #2)"] = r["vint"].astype(float)
if os.path.exists(f"{SP}/out/ssn_ts.npz"):
    s = np.load(f"{SP}/out/ssn_ts.npz"); assert (s["cutoffs"] == cut).all(); C["Timestral alone, weather"] = s["median"].astype(float)
    C["blend: trees + resid. load, 50/50 with Timestral (PR #2)"] = (C.get("trees + residual load (PR #2)", trees) + C["Timestral alone, weather"]) / 2
    C["blend without residual load: PR #1 trees + Timestral"] = (trees + C["Timestral alone, weather"]) / 2
rng = np.random.default_rng(0); n = len(cut)
def ci(e1, e0):
    blk = 3   # forecasts are 3 days apart: blocks of 3 = 9 days
    r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(3000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
print(f"{n} forecasts {str(cut[0])[:10]} to {str(cut[-1])[:10]}, MAE ct/kWh; mean price {a[:, :56].mean():.1f}, share below 0.5 ct {100*np.mean(a[:, :56] < 0.5):.0f}%")
print(f"{'':<60}{'night 00-08':>12}{'34-48 h':>9}{'above 30 ct':>13}{'below 0.5 ct':>14}   34-48 h vs PR #1")
e0 = np.abs(trees - a)[:, :56].mean(1)
for lab, p in C.items():
    e = np.abs(p - a)[:, :56]; hi, lo = a[:, :56] > 30, a[:, :56] < 0.5
    print(f"{lab:<60}{e[:, :32].mean():12.2f}{e.mean():9.2f}{(e[hi].mean() if hi.any() else np.nan):13.1f}{e[lo].mean():14.2f}   {ci(e.mean(1), e0) if lab != 'trees, PR #1' else ''}")
for m in sorted(set(str(c)[:7] for c in cut)):
    sel = np.array([str(c)[:7] == m for c in cut]); print(f"  {m}: " + "  ".join(f"{lab.split(',')[0].split(':')[0][:22]} {np.abs(p - a)[sel, :56].mean():.2f}" for lab, p in C.items()))
