import numpy as np, os
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
z = np.load(f"{SP}/out/live_ho.npz"); a = z["actual"].astype(float); f = z["vint"].astype(float); s = np.load(f"{SP}/out/tsl_123_8k.npz")["vint"].astype(float)
rng = np.random.default_rng(0)
def row(name, p, ref=None):
    e = np.abs(p - a); hi = a > 30; t = f"  {name:<46} night {e[:, :32].mean():.2f}  34-48h {e[:, :56].mean():.2f}  48-58h {e[:, 56:96].mean():.2f}  above 30 ct: MAE {e[hi].mean():.1f}, bias {np.mean((p - a)[hi]):+.1f}"
    if ref is not None:
        e0 = np.abs(ref - a)[:, :56].mean(1); e1 = e[:, :56].mean(1); r = []
        for _ in range(3000):
            idx = np.concatenate([(st + np.arange(7)) % 66 for st in rng.integers(0, 66, 10)])[:66]; r.append(100 * (e1[idx].mean() / e0[idx].mean() - 1))
        t += f"   34-48h vs trees {100*(e1.mean()/e0.mean()-1):+.0f}% [{np.percentile(r,2.5):+.0f}..{np.percentile(r,97.5):+.0f}]"
    print(t)
row("trees (PR model)", f); row("trees + 123M, 50/50", (f + s) / 2, f)
for m, lab in (("A", "+ TSO residual load, as if known"), ("C0", "+ own residual load, 11 points"), ("D_pes_O", "+ dense grid, 2-3 d old weather, trained on own fc"), ("D_opt_O", "+ dense grid, 1-2 d old weather, trained on own fc"), ("D_pes_T", "+ dense grid, 2-3 d old weather, trained on TSO"), ("D_opt_T", "+ dense grid, 1-2 d old weather, trained on TSO")):
    if not os.path.exists(f"{SP}/out/rl_{m}.npz"): continue
    r = np.load(f"{SP}/out/rl_{m}.npz"); assert (r["cutoffs"] == z["cutoffs"]).all(); p = r["vint"].astype(float)
    row(lab, p, f); row("   averaged 50/50 with 123M", (p + s) / 2, f)
    if m[0] in "CD": print(f"     own residual-load forecast: MAE {np.nanmean(np.abs(r['rl_used'] - r['rl_true'])):.1f} GW, correlation {np.corrcoef(r['rl_used'].ravel(), r['rl_true'].ravel())[0,1]:.3f}")
