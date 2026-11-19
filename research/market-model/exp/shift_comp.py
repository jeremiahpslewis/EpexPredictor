"""Is the error of the Timestral shift forecast predictable from the forecast composition (solar, wind, load)? Notes 21.124."""
import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; Z = np.load(f"{SP}/out/shift15_lite.npz"); U = Z["rl"] - Z["other"] - Z["Sp"]
for per, tag, gen in (("winter", "mid", "mid"), ("summer", "mid", "mid"), ("autumn", "v2", "v2"), ("winterf", "mid", "mid"), ("summerf", "mid", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); G = np.load(f"{SP}/out/gen_{gen}_{per}.npz"); cuts = I["cuts"].astype(int)
    u = np.stack([U[c:c + 96] for c in cuts]); ok = ~np.isnan(u).any(1); e = (T["median"].astype(float)[:, :96] - u)[ok]; n = len(e)
    so, wi, lo = (G[f"{k}_pes"].astype(float)[:, :96][ok] for k in ("solar", "wind", "load")); rl = lo - so - wi
    hod = np.tile(np.arange(96) // 4, n); F = {"composition (solar, wind | residual load)": np.column_stack([np.ones(n * 96), rl.ravel(), so.ravel(), wi.ravel()]), "residual load only": np.column_stack([np.ones(n * 96), rl.ravel()]), "composition + hour of day": np.column_stack([np.eye(24)[hod], rl.ravel(), so.ravel(), wi.ravel(), (so * rl).ravel() / 50])}
    y = e.ravel(); day = np.repeat(np.arange(n), 96); K = 5; fold = (day * K // n); line = [f"{per:8s} {n:3d} days, error sd {y.std():.2f} GW"]
    for nm, A in F.items():
        pred = np.zeros_like(y)
        for k in range(K):
            tr, te = fold != k, fold == k; pred[te] = A[te] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        line.append(f"{nm}: out-of-sample R2 {100 * (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()):+.1f}%")
    print(" | ".join(line))
