"""A durable law for net imports? Flows follow price differences up to line capacity; fitted on the first year only, frozen. Notes 21.134."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
import market1n as M
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}
S = pd.read_parquet(f"{SP}/out/star_data.parquet"); S.index = S.index.tz_convert("Europe/Berlin"); B = ["fr", "nl", "be", "at", "dk", "pl", "cz", "ch"]
H = S[["p_de", "g_de_trade", "gas"] + [f"p_{b}" for b in B] + [f"x_{b}" for b in B]].resample("1h").mean().dropna(); half = pd.Index(H.index.year.astype(str) + np.where(H.index.month <= 6, "H1", "H2"))
E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; co2 = pd.Series(E[:len(S)], index=S.index[:len(E)]).resample("1h").mean().reindex(H.index).ffill().bfill(); L = (1.85 * H.gas + 0.37 * co2)
y = H.g_de_trade.to_numpy(); first = np.asarray(H.index < pd.Timestamp("2025-03-01", tz="Europe/Berlin")); D = np.stack([(H.p_de - H[f"p_{b}"]).to_numpy() for b in B], 1)
cur = np.asarray(M.other_curves(th, H.p_de.to_numpy(), L.to_numpy())[0])                                   # the import curve as fitted (whole fit window)
# the current form refitted on the first year only, for a like-for-like frozen test
def f_cur(q, p, l): r = p / l; return q[0] + q[1] * (r - 1) - q[2] * (1 - 0.5 * (1 + np.tanh((p - q[3]) / 6.0)))
q0 = least_squares(lambda q: f_cur(q, H.p_de.to_numpy()[first], L.to_numpy()[first]) - y[first], [5.0, 15.0, 6.0, 0.0]).x; cur1 = f_cur(q0, H.p_de.to_numpy(), L.to_numpy())
# spread law: one saturating term per border, a common width per border
def f_sp(q, d): return q[0] + (q[1:9] * np.tanh(d / np.exp(q[9:17]))).sum(1)
s0 = least_squares(lambda q: f_sp(q, D[first]) - y[first], np.r_[2.0, -np.ones(8), np.full(8, np.log(20.0))], loss="soft_l1", f_scale=3.0).x; sp = f_sp(s0, D)
print("spread law, first-year fit: constant %.2f GW; per border amplitude GW / width EUR/MWh:" % s0[0], {b: (round(float(s0[1 + i]), 2), round(float(np.exp(s0[9 + i])), 0)) for i, b in enumerate(B)})
R = pd.DataFrame({"actual": y, "current curve": y - cur, "current form, first-year fit": y - cur1, "spread law, first-year fit": y - sp}, index=H.index)
print("\nresidual (actual minus law), GW, mean by half year"); print(R.drop(columns="actual").groupby(half).mean().round(2).T.to_string())
print("\nshare of the hourly variance of net imports explained, %, by half year (first year = in sample for the two first-year fits)")
r2 = lambda g: pd.Series({c: 100 * (1 - (g[c] ** 2).mean() / g["actual"].var()) for c in R.columns[1:]}); print(R.groupby(half).apply(r2).round(0).T.to_string())
print("\nthe same after removing each half year's mean residual (shape only), %"); r2s = lambda g: pd.Series({c: 100 * (1 - g[c].var() / g["actual"].var()) for c in R.columns[1:]}); print(R.groupby(half).apply(r2s).round(0).T.to_string())
# border by border: does each border's flow follow its own spread with frozen coefficients?
print("\nper border, first-year fit of flow on its own spread: residual mean by half year, GW")
rows = {}
for i, b in enumerate(B):
    xb = H[f"x_{b}"].to_numpy(); q = least_squares(lambda q: q[0] + q[1] * np.tanh(D[first, i] / np.exp(q[2])) - xb[first], [0.0, -1.0, np.log(20.0)], loss="soft_l1", f_scale=1.0).x; rb = xb - (q[0] + q[1] * np.tanh(D[:, i] / np.exp(q[2]))); rows[b] = pd.Series(rb).groupby(half.to_numpy()).mean()
print(pd.DataFrame(rows).T.round(2).to_string()); R.groupby(half).mean().to_csv(f"{SP}/out/durable_imports.csv")

# the frozen spread law with the neighbours' prices projected from last week's level relative to the gas cost index (notes 21.135)
ratio = pd.DataFrame({b: H[f"p_{b}"] / L for b in B}); full = ratio.asfreq("1h")
proj = pd.concat([full.shift(24 * k) for k in range(2, 9)]).groupby(level=0).mean().reindex(H.index)          # same hour of day, days 2 to 8 before (day 1 is not complete at a midnight issue for the next day)
Dp = np.stack([(H.p_de - proj[b] * L).to_numpy() for b in B], 1); okp = np.isfinite(Dp).all(1); spp = np.full(len(H), np.nan); spp[okp] = f_sp(s0, Dp[okp])
R2 = R.copy(); R2["spread law, neighbours projected"] = y - spp; R2 = R2[okp]; h2 = half[okp]
print("\nwith the neighbours' prices projected from the previous week (known at the issue):")
print("residual mean by half year, GW"); print(R2[["current form, first-year fit", "spread law, first-year fit", "spread law, neighbours projected"]].groupby(h2).mean().round(2).T.to_string())
print("hourly variance explained, %"); print(R2.groupby(h2).apply(lambda g: pd.Series({c: 100 * (1 - (g[c] ** 2).mean() / g["actual"].var()) for c in ("current form, first-year fit", "spread law, first-year fit", "spread law, neighbours projected")})).round(0).T.to_string())
print("error of the projected neighbour prices, MAE EUR/MWh by half year:", {k: round(float(v), 1) for k, v in pd.Series(np.abs(np.stack([(proj[b] * L - H[f"p_{b}"]).to_numpy() for b in B], 1)).mean(1)[okp]).groupby(h2.to_numpy()).mean().items()})
R2.groupby(h2).mean().to_csv(f"{SP}/out/durable_imports.csv"); np.savez_compressed(f"{SP}/out/import_law.npz", s0=s0, borders=np.array(B))

# minimal coupled form (notes 21.136): each neighbour's price over the gas cost index from its own published residual load, first-year fit, frozen
RLc = S[[f"rl_{b}" for b in B]].resample("1h").mean().reindex(H.index).ffill(); ph = {}; rows = {}
for i, b in enumerate(B):
    z = ((RLc[f"rl_{b}"] - RLc[f"rl_{b}"][first].mean()) / RLc[f"rl_{b}"][first].std()).to_numpy(); A = np.column_stack([np.ones(len(z)), z, z ** 2, z ** 3]); r = (H[f"p_{b}"] / L).to_numpy()
    co = np.linalg.lstsq(A[first], r[first], rcond=None)[0]; ph[b] = (A @ co) * L.to_numpy(); e = H[f"p_{b}"].to_numpy() - ph[b]
    rows[b] = pd.concat([pd.Series(e).groupby(half.to_numpy()).mean().rename("mean error"), pd.Series(e).groupby(half.to_numpy()).apply(lambda g: 0.0).rename("x")], axis=1)["mean error"]
    rows[b + " R2%"] = pd.Series({h: 100 * (1 - ((H[f"p_{b}"].to_numpy() - ph[b])[half == h] ** 2).mean() / H[f"p_{b}"].to_numpy()[half == h].var()) for h in sorted(set(half))})
print("\nneighbours' own curves (price from own residual load and the gas cost), first-year fit, frozen: mean error EUR/MWh and variance explained % by half year")
print(pd.DataFrame(rows).T.round(0).to_string())
Dc = np.stack([(H.p_de.to_numpy() - ph[b]) for b in B], 1); spc = f_sp(s0, Dc); R3 = pd.DataFrame({"actual": y, "current form, first-year fit": y - cur1, "spread law, actual neighbour prices": y - sp, "spread law, neighbours from their own curves": y - spc}, index=H.index)
print("\nnet imports: residual mean by half year, GW"); print(R3.drop(columns="actual").groupby(half).mean().round(2).T.to_string())
print("hourly variance explained, %"); print(R3.groupby(half).apply(lambda g: pd.Series({c: 100 * (1 - (g[c] ** 2).mean() / g["actual"].var()) for c in R3.columns[1:]})).round(0).T.to_string())

# frozen shape + slow level state (notes 21.137): neighbours' curves plus their mean error over days 2 to 8 before
ph2 = {}
for b in B:
    e = (H[f"p_{b}"] - ph[b]).asfreq("1h"); lvl = e.rolling(24 * 7, min_periods=72).mean().shift(24).reindex(H.index); ph2[b] = ph[b] + lvl.to_numpy()
Dl = np.stack([(H.p_de.to_numpy() - ph2[b]) for b in B], 1); okl = np.isfinite(Dl).all(1); spl = np.full(len(H), np.nan); spl[okl] = f_sp(s0, Dl[okl])
R4 = pd.DataFrame({"actual": y, "current form, first-year fit": y - cur1, "spread law, actual neighbour prices": y - sp, "spread law, neighbours: frozen curve + level of last week": y - spl}, index=H.index)[okl]; h4 = half[okl]
# and a level state for the import law itself: its own mean error over the same days
own = R4["spread law, neighbours: frozen curve + level of last week"].asfreq("1h").rolling(24 * 7, min_periods=72).mean().shift(24).reindex(R4.index); R4["the same + the law's own level of last week"] = R4["spread law, neighbours: frozen curve + level of last week"] - own
cur_l = R4["current form, first-year fit"].asfreq("1h").rolling(24 * 7, min_periods=72).mean().shift(24).reindex(R4.index); R4["current form + its own level of last week"] = R4["current form, first-year fit"] - cur_l; R4 = R4.dropna(); h4 = pd.Index(R4.index.year.astype(str) + np.where(R4.index.month <= 6, "H1", "H2"))
print("\nfrozen shapes + slow level states. Net imports: residual mean by half year, GW"); print(R4.drop(columns="actual").groupby(h4).mean().round(2).T.to_string())
print("hourly variance explained, %"); print(R4.groupby(h4).apply(lambda g: pd.Series({c: 100 * (1 - (g[c] ** 2).mean() / g["actual"].var()) for c in R4.columns[1:]})).round(0).T.to_string())
print("neighbour prices with the level state, MAE EUR/MWh by half year:", {k: round(float(v), 1) for k, v in pd.Series(np.abs(np.stack([ph2[b] - H[f"p_{b}"].to_numpy() for b in B], 1)).mean(1)[okl]).groupby(half[okl].to_numpy()).mean().items()})
R4.groupby(h4).apply(lambda g: pd.Series({c: 100 * (1 - (g[c] ** 2).mean() / g["actual"].var()) for c in R4.columns[1:]})).to_csv(f"{SP}/out/durable_imports_r2.csv")
