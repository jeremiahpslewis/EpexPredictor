import os, numpy as np
SP = os.environ["SP"]; z = np.load(f"{SP}/out/load_bt.npz"); T, S, Y = z["trees"], z["ts"], z["true"]; n = len(Y); rng = np.random.default_rng(1)
def boot(d, B=4000, L=5):
    out = []
    for _ in range(B):
        st = rng.integers(0, n, int(np.ceil(n / L))); ix = np.concatenate([(s + np.arange(L)) % n for s in st])[:n]; out.append(d[ix].mean())
    return np.percentile(out, [5, 95])
print("cut-offs", n, z["cut"][0][:10], "to", z["cut"][-1][:10])
for nm, sl in (("day 1 night (first 6 h)", slice(0, 24)), ("day 1 whole", slice(0, 96)), ("day 2 whole", slice(96, 192)), ("first half of day 3", slice(192, 237))):
    a = np.nanmean(np.abs(T[:, sl] - Y[:, sl]), 1); b = np.nanmean(np.abs(S[:, sl] - Y[:, sl]), 1); d = b - a; lo, hi = boot(d)
    print(f"{nm:26s} MAE GW trees {a.mean():.2f}  timestral {b.mean():.2f}  change {100 * d.mean() / a.mean():+.0f}%  ({lo:+.2f} .. {hi:+.2f} GW)  Timestral better on {int((d < 0).sum())} of {n} | worst day trees {a.max():.1f} timestral {b.max():.1f} | bias trees {np.nanmean(T[:, sl] - Y[:, sl]):+.2f} timestral {np.nanmean(S[:, sl] - Y[:, sl]):+.2f}")
e = np.abs(S - Y)[:, :236].reshape(n, -1, 4).mean(2).mean(0); f = np.abs(T - Y)[:, :236].reshape(n, -1, 4).mean(2).mean(0)
print("MAE by hour after the last published figure, every 6th: trees", np.round(f[::6], 2).tolist(), "timestral", np.round(e[::6], 2).tolist())
w = np.clip(1 - np.arange(T.shape[1]) / 96, 0, 1)
for cn, C in (("(a) average", 0.5 * (T + S)), ("(b) hand-over in 24 h", w * S + (1 - w) * T)):
    for nm, sl in (("day 1 night", slice(0, 24)), ("day 1 whole", slice(0, 96)), ("day 2 whole", slice(96, 192))):
        a = np.nanmean(np.abs(T[:, sl] - Y[:, sl]), 1); b = np.nanmean(np.abs(C[:, sl] - Y[:, sl]), 1); d = b - a; lo, hi = boot(d)
        print(f"{cn:22s} {nm:12s} MAE {b.mean():.2f} against trees {a.mean():.2f}  {100 * d.mean() / a.mean():+.0f}%  ({lo:+.2f} .. {hi:+.2f})  better on {int((d < 0).sum())} of {n}  worst {b.max():.1f}")
