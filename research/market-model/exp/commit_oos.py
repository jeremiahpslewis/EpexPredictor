"""Out-of-sample test of the lignite commitment model (notes 21.44): hazards fitted on transitions before 2025-12-01, tested on the year after.
Also the utilisation of committed capacity against price / cost."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; rng = np.random.default_rng(0); P = []
for area in ("DE_50HZ", "DE_AMPRION"):
    x = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/entsoe/units_all/{area}_lignite_*.parquet"))]).sort_index(); x = x[~x.index.duplicated()]; x.columns = [c.split(" | ")[0] for c in x.columns]; P.append(x.T.groupby(level=0).sum(min_count=1).T)
U = P[0].join(P[1], how="outer") / 1000; cap = U.quantile(0.995); dd = U.groupby(U.index.normalize()).mean(); dd.index = dd.index.tz_localize(None); on = dd / cap > 0.3; live = [u for u in U.columns if cap[u] > 0.2 and on[u].mean() > 0.1]; on = on[live]; cap = cap[live]
R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); day = R.index.normalize().tz_localize(None); PD = pd.DataFrame({"rc": R.rc_l.groupby(day).mean(), "mn": R.rc_l.groupby(day).min()}).asfreq("D").interpolate(limit=3); days = on.index; W = pd.DataFrame({"rc": PD.rc.rolling(2).mean(), "mn": PD.mn.rolling(2).min()}).reindex(days)
ok = np.array([(days[i] - days[i - 1]).days == 2 and not np.isnan(W.rc.iloc[i]) for i in range(1, len(days))]); X = np.column_stack([np.ones(len(days)), np.clip(W.rc.fillna(1.0), 0, 2.5), np.clip(W.mn.fillna(0.4), -1, 2), (days.dayofweek >= 5).astype(float)])
def logit_fit(Xm, y):
    b = np.zeros(Xm.shape[1])
    for _ in range(30): p = 1 / (1 + np.exp(-Xm @ b)); b = b + np.linalg.solve(Xm.T @ (Xm * (p * (1 - p))[:, None]) + 1e-6 * np.eye(Xm.shape[1]), Xm.T @ (y - p))
    return b
sig = lambda z: 1 / (1 + np.exp(-z)); tr = np.array(days[1:] < pd.Timestamp("2025-12-01")) & ok; A = on.values; rows_on, rows_off = [], []
for i in np.where(tr)[0] + 1:
    for j in range(len(live)): (rows_off if A[i - 1, j] else rows_on).append((X[i], A[i, j]))
b_on = logit_fit(np.array([r[0] for r in rows_on]), np.array([float(r[1]) for r in rows_on])); b_off = logit_fit(np.array([r[0] for r in rows_off]), np.array([float(not r[1]) for r in rows_off])); print("hazards fitted before December 2025 (constant, mean price / cost, cheapest hour, weekend): on", np.round(b_on, 2), "off", np.round(b_off, 2))
K = (A * cap.values[None]).sum(1); one = np.full(len(days), np.nan)
for i in range(1, len(days)): po, pf = sig(X[i] @ b_on), sig(X[i] @ b_off); one[i] = (np.where(A[i - 1], 1 - pf, po) * cap.values).sum()
for lab, m in (("fitting period", tr), ("HELD-OUT year", np.array(days[1:] >= pd.Timestamp("2025-12-01")) & ok)):
    i = np.where(m)[0] + 1; print(f"{lab}: {len(i)} days; committed capacity two days ahead, MAE GW: with the hazards {np.abs(one[i] - K[i]).mean():.2f}; carrying the state unchanged {np.abs(K[i - 1] - K[i]).mean():.2f}; R2 {np.corrcoef(one[i], K[i])[0, 1] ** 2:.2f} vs {np.corrcoef(K[i - 1], K[i])[0, 1] ** 2:.2f}")
ho0 = int(np.argmax(days >= pd.Timestamp("2025-12-01"))); S = np.zeros((200, len(days) - ho0)); st = np.tile(A[ho0], (200, 1)); S[:, 0] = (st * cap.values[None]).sum(1)
for k, i in enumerate(range(ho0 + 1, len(days)), 1): po, pf = sig(X[i] @ b_on), sig(X[i] @ b_off); u_ = rng.random(st.shape); st = np.where(st, u_ > pf, u_ < po); S[:, k] = (st * cap.values[None]).sum(1)
q = np.percentile(S, [5, 50, 95], axis=0); Kh = K[ho0:]; print(f"HELD-OUT year, run forward from 1 December 2025 on prices alone: median path R2 {np.corrcoef(q[1], Kh)[0, 1] ** 2:.2f}, MAE {np.abs(q[1] - Kh).mean():.2f} GW, bias {np.mean(q[1] - Kh):+.2f} GW; observed inside the 5-95% band on {100 * ((Kh >= q[0]) & (Kh <= q[2])).mean():.0f}% of days")
# utilisation of what is committed
Kd = pd.Series(K, index=days); H = R[["lig", "rc_l", "tr"]].copy(); H["K"] = Kd.reindex(R.index.normalize().tz_localize(None)).values; H = H.dropna(); H["ut"] = H.lig / H.K; e = np.r_[-1, 0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 6]
print("utilisation of committed lignite capacity (output / GW of units on) by price / cost band: " + "  ".join(f"{a:.1f}-{b:.1f}: {H[(H.rc_l >= a) & (H.rc_l < b)].ut.median():.2f}" for a, b in zip(e[:-1], e[1:])), "| sd of hourly output: total", round(float(H.lig.std()), 2), "GW; given the committed capacity and the price band", round(float((H.lig - H.K * H.groupby(np.digitize(H.rc_l, e)).ut.transform("median")).std()), 2), "GW")
np.savez(f"{SP}/out/commit_lignite.npz", days=days.values.astype("datetime64[D]").astype(str), K=K, b_on=b_on, b_off=b_off, cap=cap.values)
