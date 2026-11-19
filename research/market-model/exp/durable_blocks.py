"""Which block of the curves ages (notes 21.132): actual output of each block minus the curve at the observed price, by half year."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1n as M
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; R["co2"] = pd.Series(np.nanmean(E[:len(R) * 4].reshape(-1, 4), 1), index=R.index).ffill()
R["L"] = (1.85 * R.gas + 0.37 * R.co2).shift(48).ffill().bfill(); co2 = R.co2.shift(48).ffill().bfill(); ok = R[["p", "L", "lig", "coal", "gasg", "trade"]].notna().all(1); R = R[ok]; co2 = co2[ok]
fu = M.fuels(th, R.p.to_numpy(), R.L.to_numpy(), co2.to_numpy()); gX, gS = M.other_curves(th, R.p.to_numpy(), R.L.to_numpy()); print("fuel keys", list(fu.keys()))
k = list(fu.keys()); mod = pd.DataFrame({"lignite": np.asarray(fu[k[0]]), "hard coal": np.asarray(fu[k[1]]), "gas": np.asarray(fu[k[2]]), "net imports": np.asarray(gX), "storage": np.asarray(gS)}, index=R.index)
act = pd.DataFrame({"lignite": R.lig, "hard coal": R.coal, "gas": R.gasg, "net imports": R.trade, "storage": R.psgen - R.pump}); loc = R.index.tz_convert("Europe/Berlin"); half = pd.Index(loc.year.astype(str) + np.where(loc.month <= 6, "H1", "H2"))
T = pd.DataFrame({"price EUR/MWh": R.p.groupby(half).mean(), "gas EUR/MWh": R.gas.groupby(half).mean(), "carbon EUR/t": R.co2.groupby(half).mean(), "price / cost index": (R.p / R.L).groupby(half).mean()})
print(T.round(2).T.to_string()); print("\nactual minus curve at the observed price, GW, mean by half year (fit window: to Nov 2025)")
print((act - mod).groupby(half).mean().round(1).T.to_string()); print("\nactual output, GW"); print(act.groupby(half).mean().round(1).T.to_string()); print("\ncurve at the observed price, GW"); print(mod.groupby(half).mean().round(1).T.to_string())
cst = M.cost(th, R.L.to_numpy(), co2.to_numpy()); print("\nprice over each fuel's modelled cost, mean by half year"); print(pd.DataFrame({"lignite": R.p.to_numpy() / cst["l"], "hard coal": R.p.to_numpy() / cst["c"], "gas": R.p.to_numpy() / cst["g"]}, index=R.index).groupby(half).mean().round(2).T.to_string())
(act - mod).groupby(half).mean().to_csv(f"{SP}/out/durable_blocks.csv")

# what moves with the residuals of the blocks that age: the gas price level, or time?
res = (act - mod); m = loc.to_period("M"); Mo = res.groupby(m).mean(); Mo["gas"] = R.gas.groupby(m).mean(); Mo["t"] = np.arange(len(Mo)) / 12.0; Mo["rl"] = R.rl.groupby(m).mean(); Mo = Mo.iloc[:-1]
print("\nmonthly residuals (32 months): share of variance explained by a straight line in time / by the gas price level / by both")
for c in ("net imports", "lignite", "hard coal", "gas"):
    y = Mo[c].to_numpy(); r2 = lambda A: 100 * (1 - ((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()); o = np.ones(len(y)); A3 = np.column_stack([o, Mo.t, np.log(Mo.gas)]); co = np.linalg.lstsq(A3, y, rcond=None)[0]
    print(f"   {c:12s} time {r2(np.column_stack([o, Mo.t])):3.0f}%   gas price {r2(np.column_stack([o, np.log(Mo.gas)])):3.0f}%   both {r2(A3):3.0f}%  | trend {co[1]:+.2f} GW per year, {co[2]:+.2f} GW per doubling-e of the gas price | sd of the monthly residual {y.std():.2f} GW")
tot = res[["lignite", "hard coal", "gas", "net imports"]].sum(1).groupby(m).mean().iloc[:-1]; print("   sum of the four blocks: sd of the monthly residual", round(float(tot.std()), 2), "GW; by half year", res[["lignite", "hard coal", "gas", "net imports"]].sum(1).groupby(half).mean().round(1).to_dict())
