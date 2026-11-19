"""Durability of the structural layer over 2.7 years (notes 21.132). D1 shift by month and frozen-profile price error; D2 stability of static terms;
D3 frozen against refitted static core; D4 top output per fuel against the assumed capacities."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc, core_fit as cf
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; F = lc.load(); D = F["DE"]; idx = D.index
WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); out = {}
loc = idx.tz_convert("Europe/Berlin"); hod = loc.hour.to_numpy(); mon = loc.to_period("M"); qtr = loc.to_period("Q"); FITEND = pd.Timestamp("2025-12-01", tz="Europe/Berlin")
for name, f in (("full model", "shift15.npz"), ("slim (as in the application)", "shift15_lite.npz")):
    Z = np.load(f"{SP}/out/{f}"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"] - Z["nbterm"]; p = Z["p"]; s = pd.Series(U, index=idx)
    g = s.groupby(mon).agg(["mean", "std"]); infit = np.asarray(idx < FITEND.tz_convert("UTC"))
    print(f"D1 {name}: monthly mean of the shift GW, min {g['mean'].min():+.1f} ({g['mean'].idxmin()}) max {g['mean'].max():+.1f} ({g['mean'].idxmax()}) | mean in the fit window {np.nanmean(U[infit]):+.2f}, in the 12 months after {np.nanmean(U[~infit]):+.2f} | sd within a month, median {g['std'].median():.1f}")
    print("   by half year:", {str(k): round(float(v), 1) for k, v in s.groupby(loc.to_period('6M') if False else (loc.year.astype(str) + np.where(loc.month <= 6, 'H1', 'H2'))).mean().items()})
    out[name] = dict(U=U, X=X, p=p, month=g)
# frozen hour-of-day / day-type profile of the shift from the fit window, price error by month with it (slim model, constants as in the application)
Z = np.load(f"{SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]; p = Z["p"]; infit = np.asarray(idx < FITEND.tz_convert("UTC"))
key = hod + 24 * WE.astype(int); prof = pd.Series(U[infit]).groupby(key[infit]).mean().reindex(range(48)).to_numpy(); eps = prof[key]
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC"); gas = R.gas.reindex(idx, method="ffill").shift(192).ffill().bfill().to_numpy(); L = 1.85 * gas + 0.37 * 75.0
ok = np.isfinite(p) & np.isfinite(X) & np.isfinite(L); ph = np.full(len(idx), np.nan); ph[ok] = np.asarray(M.clear(th, X[ok], L[ok], np.full(ok.sum(), 75.0), eps=eps[ok]))
err = pd.Series(np.abs(ph - p) / 10, index=idx); bym = err.groupby(mon).mean(); lev = pd.Series(p / 10, index=idx).groupby(mon).mean()
ins, outs = err[infit].mean(), err[~infit].mean(); print(f"D1 curves with the shift profile frozen at the fit window: price MAE ct in the fit window {ins:.2f}, in the 12 months after {outs:.2f} ({100 * (outs / ins - 1):+.0f}%); relative to the price level: {100 * ins / np.nanmean(p[infit] / 10):.0f}% and {100 * outs / np.nanmean(p[~infit] / 10):.0f}%")
print("   by quarter, MAE ct / mean price ct:", {str(k): f"{err.groupby(qtr).mean()[k]:.2f}/{pd.Series(p / 10, index=idx).groupby(qtr).mean()[k]:.1f}" for k in sorted(set(qtr))})
# D2 and D3: static core per quarter, frozen (first 12 months) against refitted on the trailing 85 days
days = len(idx) // 96; c0 = (np.where((loc.hour == 0) & (loc.minute == 0))[0])  # local midnights
first_year = [c for c in c0 if idx[c] < pd.Timestamp("2025-03-01", tz="UTC") and c >= 0]; a, b = first_year[0], first_year[-1]
def stack(lo, hi):  # whole local days between two midnight indices, skipping days that are not 96 long (clock change)
    cs = [c for c in c0 if lo <= c < hi]; cs = [c for i, c in enumerate(cs[:-1]) if cs[i + 1] - c == 96 and np.isfinite(U[c:c + 96]).all() and np.isfinite(X[c:c + 96]).all()]; return cs
tr = stack(a, b); uf = np.concatenate([U[c:c + 96] for c in tr]); xf = np.concatenate([X[c:c + 96] for c in tr]); wf = np.concatenate([WE[c:c + 96] for c in tr]); frozen = cf.fit(uf, xf, wf, 3)
rows = []; qs = sorted(set(qtr))
for q in qs:
    cs = stack(np.where(qtr == q)[0][0], np.where(qtr == q)[0][-1] + 1)
    if len(cs) < 20: continue
    u = np.concatenate([U[c:c + 96] for c in cs]); x = np.concatenate([X[c:c + 96] for c in cs]); w = np.concatenate([WE[c:c + 96] for c in cs]); own = cf.fit(u, x, w, 3)
    r2 = lambda pred: 100 * (1 - np.mean((u - pred) ** 2) / np.var(u)); pf = cf.apply(frozen, x, w); pr = []
    for c in cs:
        h = stack(c - 86 * 96, c)
        if len(h) < 40: pr.append(np.full(96, np.nan)); continue
        m = cf.fit(np.concatenate([U[k:k + 96] for k in h]), np.concatenate([X[k:k + 96] for k in h]), np.concatenate([WE[k:k + 96] for k in h]), 3); pr.append(cf.apply(m, X[c:c + 96], WE[c:c + 96]))
    pr = np.concatenate(pr); okr = np.isfinite(pr); prof_q = own["co"][:24] - own["co"][:24].mean()
    rows.append(dict(q=str(q), days=len(cs), b=own["co"][48], l=own["co"][49], g=own["co"][50], frozen=r2(pf), frozen_shape=100 * (1 - np.mean(((u - u.mean()) - (pf - pf.mean())) ** 2) / np.var(u)), refit=100 * (1 - np.mean((u[okr] - pr[okr]) ** 2) / np.var(u[okr])) if okr.sum() > 2000 else np.nan, own=r2(cf.apply(own, x, w)), prof=prof_q, level=u.mean()))
T = pd.DataFrame(rows); pc = np.corrcoef(np.stack(T.prof.to_numpy())); fz = frozen["co"][:24] - frozen["co"][:24].mean()
print("D2 static terms per quarter (first-year fit: within hour %+.2f, daily swing %+.2f, level %+.2f)" % tuple(frozen["co"][48:51]))
print(T[["q", "days", "b", "l", "g", "level"]].round(2).to_string(index=False))
print("   hour-of-day profile (working days): correlation between quarters, median %.2f, min %.2f; with the first-year profile by quarter:" % (np.median(pc[np.triu_indices(len(T), 1)]), pc[np.triu_indices(len(T), 1)].min()), [round(float(np.corrcoef(fz, v)[0, 1]), 2) for v in T.prof])
print("D3 share of the shift's variance explained, by quarter (%): frozen at the first year / the same with the quarter's own mean level / refitted on the trailing 85 days / fitted on the quarter itself")
print(T[["q", "frozen", "frozen_shape", "refit", "own"]].round(0).to_string(index=False))
# D4 top output per fuel
S = pd.read_parquet(f"{SP}/out/star_data.parquet"); S.index = S.index.tz_convert("Europe/Berlin"); top = S[["g_de_lignite", "g_de_coal", "g_de_gas"]].groupby(S.index.to_period("Q")).quantile(0.995)
print("D4 top output per fuel (99.5th percentile of the quarter, GW); the model assumes lignite 18, hard coal 12, gas 33"); print(top.round(1).T.to_string())
np.savez_compressed(f"{SP}/out/durable.npz", months=np.array([str(m) for m in bym.index]), err=bym.to_numpy(), level=lev.reindex(bym.index).to_numpy(), shift_full=out["full model"]["month"]["mean"].to_numpy(), shift_slim=out["slim (as in the application)"]["month"]["mean"].to_numpy(), q=T.q.to_numpy().astype(str), frozen=T.frozen.to_numpy(), frozen_shape=T.frozen_shape.to_numpy(), refit=T.refit.to_numpy(), own=T.own.to_numpy(), b=T.b.to_numpy(), g=T.g.to_numpy(), topq=np.array([str(k) for k in top.index]), top=top.to_numpy())
