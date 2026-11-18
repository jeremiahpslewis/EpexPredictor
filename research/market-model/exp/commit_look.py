"""The commitment state of lignite units and its dynamics (notes 21.43): switch-on and switch-off hazards against price / cost, and a two-state
model per unit simulated forward from prices alone against the committed capacity observed."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def fin(ax, t):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.set_title(t, loc="left", fontsize=9.3, fontweight="bold", color=INK)
rng = np.random.default_rng(0); P = []
for area in ("DE_50HZ", "DE_AMPRION"):
    x = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/entsoe/units_all/{area}_lignite_*.parquet"))]).sort_index(); x = x[~x.index.duplicated()]; x.columns = [c.split(" | ")[0] for c in x.columns]; P.append(x.T.groupby(level=0).sum(min_count=1).T)
U = P[0].join(P[1], how="outer") / 1000; cap = U.quantile(0.995); dd = U.groupby(U.index.normalize()).mean(); dd.index = dd.index.tz_localize(None); on = dd / cap > 0.3; live = [u for u in U.columns if cap[u] > 0.2 and on[u].mean() > 0.1]; on = on[live]; cap = cap[live]
R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); day = R.index.normalize().tz_localize(None); PD = pd.DataFrame({"rc": R.rc_l.groupby(day).mean(), "mn": R.rc_l.groupby(day).min()}).asfreq("D").interpolate(limit=3)
days = on.index; rows = []
for i in range(len(days) - 1):
    d0, d1 = days[i], days[i + 1]
    if (d1 - d0).days != 2: continue
    w = PD.loc[d0 + pd.Timedelta(days=1): d1]; 
    if w.rc.isna().any() or len(w) < 2: continue
    for u in live: rows.append((d1, u, bool(on[u].iloc[i]), bool(on[u].iloc[i + 1]), float(w.rc.mean()), float(w.mn.min()), d1.dayofweek >= 5, cap[u]))
T = pd.DataFrame(rows, columns=["d", "u", "a", "b", "rc", "mn", "we", "cap"]); offs = T[~T.a]; ons = T[T.a]; print(f"unit transitions over two days: {len(T)}; from off: {len(offs)}, switched on {offs.b.mean():.3f}; from on: {len(ons)}, switched off {(~ons.b).mean():.3f}")
def haz(df, y, xcol, e): b = np.digitize(df[xcol], e) - 1; return np.array([y[b == k].mean() if (b == k).sum() >= 40 else np.nan for k in range(len(e) - 1)]), np.array([(b == k).sum() for k in range(len(e) - 1)])
e = np.r_[0.0, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 4.0]; cen = np.array([0.35, 0.6, 0.78, 0.93, 1.08, 1.22, 1.45, 1.9]); em = np.r_[-2.0, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 3.0]; cm = np.array([-0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.3])
fig, axs = plt.subplots(1, 4, figsize=(19.5, 4.9), gridspec_kw=dict(wspace=0.27)); h_on, _ = haz(offs, offs.b.values, "rc", e); h_off, _ = haz(ons, (~ons.b).values, "rc", e)
ax = axs[0]; ax.plot(cen, h_on, "-o", color=BLUE, linewidth=1.9, markersize=4, label="off unit is on two days later"); ax.plot(cen, h_off, "-o", color=ORANGE, linewidth=1.9, markersize=4, label="on unit is off two days later"); ax.set_ylim(0, 1); ax.set_xlabel("mean price / lignite cost over the two days"); ax.set_ylabel("share of units switching"); ax.legend(frameon=False, fontsize=8); fin(ax, f"As price / cost rises, switching on goes from\n{np.nanmin(h_on):.2f} to {np.nanmax(h_on):.2f} and switching off from {np.nanmax(h_off):.2f} to {np.nanmin(h_off):.2f}")
ax = axs[1]; g_on, _ = haz(offs, offs.b.values, "mn", em); g_off, _ = haz(ons, (~ons.b).values, "mn", em); ax.plot(cm, g_on, "-o", color=BLUE, linewidth=1.9, markersize=4, label="off unit is on two days later"); ax.plot(cm, g_off, "-o", color=ORANGE, linewidth=1.9, markersize=4, label="on unit is off two days later"); ax.set_ylim(0, 1); ax.set_xlabel("cheapest hour's price / lignite cost over the two days"); ax.set_ylabel("share of units switching"); ax.legend(frameon=False, fontsize=8); fin(ax, f"Against the cheapest hour: switching off\n{np.nanmax(g_off):.2f} with a deep trough, {np.nanmin(g_off):.2f} with none")
# two-state model per unit: logistic hazards in (mean, cheapest hour, weekend), fitted by a few Newton steps
def logit_fit(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(30): p = 1 / (1 + np.exp(-X @ b)); b = b + np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return b
Xf = lambda df: np.column_stack([np.ones(len(df)), np.clip(df.rc, 0, 2.5), np.clip(df.mn, -1, 2), df.we.astype(float)]); b_on = logit_fit(Xf(offs), offs.b.values.astype(float)); b_off = logit_fit(Xf(ons), (~ons.b).values.astype(float)); print("hazard coefficients (constant, mean price / cost, cheapest hour, weekend): on", np.round(b_on, 2), "off", np.round(b_off, 2))
sig = lambda z: 1 / (1 + np.exp(-z)); xx = np.linspace(0.2, 2.2, 50); pon = sig(b_on[0] + b_on[1] * xx + b_on[2] * (xx - 0.6)); poff = sig(b_off[0] + b_off[1] * xx + b_off[2] * (xx - 0.6)); stat = pon / (pon + poff)
ax = axs[2]; ax.plot(xx, stat, color=INK, linewidth=2, label="share of units on if the price stayed there"); ax.plot(xx, 1 / np.maximum(poff, 1e-3) * 2, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 2)), label="expected length of an on-spell, days / 100" if False else None); ax.set_ylim(0, 1); ax.set_xlabel("mean price / lignite cost"); ax.set_ylabel("long-run share of units on"); half = lambda p: np.log(0.5) / np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)) * 2
i1 = int(np.argmin(np.abs(xx - 1.0))); fin(ax, f"At price = cost a unit stays on about {2 / poff[i1]:.0f} days\nand stays off about {2 / pon[i1]:.0f}: inertia in both")
# simulate forward from prices alone (every two days), all units, 200 paths; compare committed capacity
grid_d = [d for i, d in enumerate(days[1:], 1) if (d - days[i - 1]).days == 2 and d in PD.index and not np.isnan(PD.rc.get(d, np.nan))]; W = pd.DataFrame({"rc": PD.rc.rolling(2).mean(), "mn": PD.mn.rolling(2).min()}).reindex(days); obsK = (on.values * cap.values[None]).sum(1); S = np.zeros((200, len(days))); st = np.tile(on.values[0], (200, 1)); S[:, 0] = (st * cap.values[None]).sum(1); one = np.zeros(len(days)); one[0] = obsK[0]
for i in range(1, len(days)):
    r_, m_ = W.rc.iloc[i], W.mn.iloc[i]; r_ = 1.0 if np.isnan(r_) else r_; m_ = 0.4 if np.isnan(m_) else m_; x = np.array([1.0, np.clip(r_, 0, 2.5), np.clip(m_, -1, 2), float(days[i].dayofweek >= 5)]); po, pf = sig(x @ b_on), sig(x @ b_off); u_ = rng.random(st.shape); st = np.where(st, u_ > pf, u_ < po); S[:, i] = (st * cap.values[None]).sum(1); prev = on.values[i - 1]; one[i] = (np.where(prev, 1 - pf, po) * cap.values).sum()
ax = axs[3]; q = np.percentile(S, [5, 50, 95], axis=0); ax.fill_between(days, q[0], q[2], color=BLUE, alpha=0.18, linewidth=0, label="simulated from prices alone (5-95%)"); ax.plot(days, q[1], color=BLUE, linewidth=1.2); ax.plot(days, pd.Series(obsK).rolling(3, center=True, min_periods=1).mean(), color=INK, linewidth=1.3, label="observed"); ax.set_ylabel("lignite capacity committed, GW"); ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.tick_params(axis="x", labelrotation=30)
cf = np.corrcoef(q[1], obsK)[0, 1] ** 2; c1 = np.corrcoef(one[1:], obsK[1:])[0, 1] ** 2; naive = np.corrcoef(obsK[:-1], obsK[1:])[0, 1] ** 2; inside = float(((obsK >= q[0]) & (obsK <= q[2])).mean()); fin(ax, f"Prices alone explain {100 * cf:.0f}% of committed capacity;\nfrom the state two days before {100 * c1:.0f}% (unchanged: {100 * naive:.0f}%)")
print(f"committed capacity GW: mean {obsK.mean():.1f}, sd {obsK.std():.2f}; free-run median R2 {cf:.2f}, observed inside the simulated 5-95% band on {100 * inside:.0f}% of days; two-days-ahead from the observed state R2 {c1:.2f}, MAE {np.abs(one[1:] - obsK[1:]).mean():.2f} GW; carrying the state unchanged R2 {naive:.2f}, MAE {np.abs(obsK[:-1] - obsK[1:]).mean():.2f} GW")
fig.suptitle("The commitment state of lignite units: how switching on and off depends on the price, and a two-state model per unit run forward (22 units, every second day)", x=0.125, ha="left", fontsize=10.5, fontweight="bold", color=INK, y=1.04); fig.savefig(f"{SP}/out/commit_look.png", dpi=100, bbox_inches="tight"); print("figure ok")
