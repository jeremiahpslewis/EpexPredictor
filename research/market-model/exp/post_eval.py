"""Full-sampling fits of the structural model: diagnostics, mode vs posterior, response curves with bands. usage: post_eval.py RUN"""
import sys, glob, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; run = sys.argv[1]
files = sorted(glob.glob(f"{SP}/out/struct_{run}/post_*.npz")); q = lambda v: np.percentile(v, [5, 50, 95], axis=0)
for f in files:
    z = np.load(f); tag = f.split("post_")[1][:-4]; d = z["diag"]; g = float(z["gas"]); grid = z["grid"]; b, h = z["b"].astype(float), z["h"].astype(float)
    print(f"\n{tag}: gas {g:.0f}; {len(b)} draws kept; sampling {d[3]:.0f} s, max r_hat {d[0]:.3f}, min effective sample size {d[1]:.0f}, divergences {int(d[2])}")
    m, mp = z["m_known"].astype(float) / 10, z["map_known"].astype(float) / 10; sc = z["s_known"].astype(float) / 10
    print(f"  forecast (known residual load, first day): posterior median vs mode, mean absolute difference {np.abs(np.median(m, 0) - mp)[:96].mean():.3f} ct; posterior sd of the mean {m.std(0)[:96].mean():.2f} ct against error scale {sc.mean(0)[:96].mean():.2f} ct")
    P = (b + g * h) / 10; i = [int(np.where(grid == x)[0][0]) for x in (0, 10, 20, 30, 40, 50, 60)]
    sl = 10 * (P[:, 1:] - P[:, :-1]) / 5.0          # EUR/MWh per GW per segment
    print("  price ct/kWh at this gas price, by effective residual load [5%..95%]: " + "  ".join(f"{int(grid[k])}: {q(P[:, k])[1]:.1f} [{q(P[:, k])[0]:.1f}..{q(P[:, k])[2]:.1f}]" for k in i))
    print("  EUR/MWh per GW, by segment:  " + "  ".join(f"{int(grid[k])}-{int(grid[k]) + 5}: {q(sl[:, k])[1]:.1f} [{q(sl[:, k])[0]:.1f}..{q(sl[:, k])[2]:.1f}]" for k in i[:-1] + [i[-1]]))
    print("  gas pass-through (EUR/MWh per EUR/MWh gas): " + "  ".join(f"{int(grid[k])}: {q(h[:, k])[1]:.2f} [{q(h[:, k])[0]:.2f}..{q(h[:, k])[2]:.2f}]" for k in i))
    print("  base cost EUR/MWh: " + "  ".join(f"{int(grid[k])}: {q(b[:, k])[1]:.0f} [{q(b[:, k])[0]:.0f}..{q(b[:, k])[2]:.0f}]" for k in i))
