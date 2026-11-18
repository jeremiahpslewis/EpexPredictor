import numpy as np, os
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
z = np.load(f"{SP}/out/live_ho.npz"); a = z["actual"].astype(float); f = z["vint"].astype(float); lead = np.arange(153) / 4.0
w = (0.8 * np.exp(-lead / 6))[None, :]
def row(name, p):
    e = np.abs(p - a); print(f"  {name:<34} 0-3h {e[:, :12].mean():.2f}  3-6h {e[:, 12:24].mean():.2f}  6-8h {e[:, 24:32].mean():.2f}  night {e[:, :32].mean():.2f}  34-48h {e[:, :56].mean():.2f}  48-58h {e[:, 56:96].mean():.2f}")
row("trees", f)
for n, lab in (("tslive_ho", "28M ctx 8192"), ("tsl_28_2k", "28M ctx 2048"), ("tsl_123_2k", "123M ctx 2048"), ("tsl_209_2k", "209M ctx 2048"), ("tsl_123_8k", "123M ctx 8192")):
    if not os.path.exists(f"{SP}/out/{n}.npz"): continue
    s = np.load(f"{SP}/out/{n}.npz")["vint"].astype(float)
    row(lab + " alone", s); row(lab + " fade blend", (1 - w) * f + w * s)
    hi = a > 30; print(f"     above 30 ct ({hi.sum()} slots): mean error trees {np.mean((f - a)[hi]):+.1f}, this model {np.mean((s - a)[hi]):+.1f}")
