"""Where the error of the Timestral shift forecast sits: daily level, hourly profile, within-hour pattern. Notes 21.122."""
import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; Z = np.load(f"{SP}/out/shift15_lite.npz"); U = Z["rl"] - Z["other"] - Z["Sp"]; tot = np.zeros(3); res = {}
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2"), ("winterf", "mid"), ("summerf", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int)
    u = np.stack([U[c:c + 96] for c in cuts]); f = T["median"].astype(float)[:, :96]; ok = ~np.isnan(u).any(1); u, f = u[ok], f[ok]; e = f - u
    lvl = e.mean(1, keepdims=True); hr = e.reshape(len(e), 24, 4).mean(2); prof = np.repeat(hr, 4, 1) - lvl; qh = e - np.repeat(hr, 4, 1)
    ss = np.array([(lvl ** 2).mean(), (prof ** 2).mean(), (qh ** 2).mean()]); tot += ss * len(e)
    # how well is each part forecast: the same split for the shift itself, and the forecast's skill on each part against simple references
    ul = u.mean(1); fl = f.mean(1); prev = np.stack([U[c - 96:c] for c in cuts])[ok]; pl = np.nanmean(prev, 1)
    uq = u - np.repeat(u.reshape(len(u), 24, 4).mean(2), 4, 1); fq = f - np.repeat(f.reshape(len(f), 24, 4).mean(2), 4, 1)
    res[per] = dict(u=u, f=f)
    print(f"{per:8s} {len(e):3d} days | shift error RMSE {np.sqrt((e ** 2).mean()):.2f} GW | share of squared error: level {100 * ss[0] / ss.sum():.0f}%  hourly profile {100 * ss[1] / ss.sum():.0f}%  within hour {100 * ss[2] / ss.sum():.0f}% | level: MAE Timestral {np.abs(fl - ul).mean():.2f}  yesterday's level {np.abs(pl - ul).mean():.2f}  sd of the level's daily change {np.std(ul - pl):.2f} | within-hour part: sd true {uq.std():.2f}  sd forecast {fq.std():.2f}  correlation {np.corrcoef(uq.ravel(), fq.ravel())[0, 1]:.2f}")
print("all sets: level", round(100 * tot[0] / tot.sum()), "% hourly profile", round(100 * tot[1] / tot.sum()), "% within hour", round(100 * tot[2] / tot.sum()), "%")
