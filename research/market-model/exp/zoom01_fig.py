"""Pictures for zoom levels 0 and 1 (notes section 20)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
OR = LinearSegmentedColormap.from_list("o", ["#f6d3c2", "#eb6834", "#7a2b0b"])
D = pd.read_parquet(f"{SP}/out/zoom1_rows.parquet"); Z = np.load(f"{SP}/out/zoom1_post.npz"); K = Z["K"]; kn = Z["kn"]; km = np.median(kn, 0); c0 = float(Z["c0"]); sig = float(np.median(Z["sigma"])); nu = float(np.median(Z["nu"])); tr = D[D.tr]; ho = D[D.ho]; rng = np.random.default_rng(0); out = {}
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig = plt.figure(figsize=(18.5, 13.8)); gs = fig.add_gridspec(3, 12, hspace=0.42, wspace=2.6); HO0 = pd.Timestamp("2025-12-01", tz="Europe/Berlin")
# level 0
ax = fig.add_subplot(gs[0, 0:3]); m = D[D.tr | D.ho].resample("MS").agg(p=("p", "median"), L=("L", "median"), y=("y", "median")); ax.plot(m.index, m.L / 10, color=NEUT, linewidth=2, label="gas-fired cost"); ax.plot(m.index, m.p / 10, color=ORANGE, linewidth=2, label="price"); ax.axvspan(HO0, m.index[-1], color="#f1f0ea", zorder=0)
ax.set_ylabel("monthly median, ct/kWh"); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "Level 0  Price follows gas-fired cost\n(shaded: the held-out year)")
ax = fig.add_subplot(gs[0, 3:6]); ax.plot(m.index, m.y, color=ORANGE, linewidth=2, label="price / gas-fired cost"); ax.axhline(c0, color=BLUE, linewidth=2, label=f"level 0: one number, {c0:.2f}"); ax.axvspan(HO0, m.index[-1], color="#f1f0ea", zorder=0); ax.set_ylim(0, 1.6); ax.set_ylabel("monthly median ratio"); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.legend(frameon=False, loc="lower left", fontsize=8); clean(ax)
ttl(ax, "Level 0  The ratio stays of order one,\nalso in the year gas doubled"); out["level 0 monthly ratio range, fitting / held out"] = (float(m.y[m.index < HO0].min()), float(m.y[m.index < HO0].max()), float(m.y[m.index >= HO0].min()), float(m.y[m.index >= HO0].max()))
ax = fig.add_subplot(gs[0, 6:9]); sub = tr.iloc[rng.choice(len(tr), 5000, replace=False)]; ax.scatter(sub.rl, sub.y, s=4, color=NEUT, alpha=0.5, linewidths=0); ax.axhline(c0, color=BLUE, linewidth=2); ax.set_xlim(-12, 75); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("German residual load, GW"); ax.set_ylabel("price / gas-fired cost"); clean(ax); ttl(ax, "Level 0  What it leaves out:\nthe ratio depends on residual load")
ax = fig.add_subplot(gs[0, 9:12]); pk = Z["prior_kn"]
for i in range(70): ax.plot(K, pk[i], color=BLUE, linewidth=0.7, alpha=0.35)
ax.set_xlim(-12, 75); ax.set_ylim(-0.6, 4.2); ax.set_xlabel("German residual load, GW"); ax.set_ylabel("price / gas-fired cost"); clean(ax); ttl(ax, "Level 1  Before data: curves its priors\nallow (70 draws)"); out["level 1 prior: k at 30 / 70 GW, 5-50-95%"] = tuple(float(v) for v in np.percentile(pk[:, 4], [5, 50, 95])) + tuple(float(v) for v in np.percentile(pk[:, 8], [5, 50, 95]))
# level 1 fit and hold-out
xx = np.linspace(-10, 75, 200); kq = np.percentile(np.array([np.interp(xx, K, k_) for k_ in kn[::8]]), [5, 50, 95], axis=0); tq = stats.t.ppf([0.05, 0.25, 0.75, 0.95], nu) * sig
for c_, (dat, lab, colr) in enumerate(((tr, "fitting period, Mar 2024 - Nov 2025", None), (ho, "held-out year, Dec 2025 - Nov 2026", "gas"))):
    ax = fig.add_subplot(gs[1, 3 * c_:3 * c_ + 3]); sub = dat.iloc[rng.choice(len(dat), 5000, replace=False)]
    if colr is None: ax.scatter(sub.rl, sub.y, s=4, color=NEUT, alpha=0.5, linewidths=0)
    else: sc = ax.scatter(sub.rl, sub.y, s=4, c=sub.gas, cmap=OR, alpha=0.7, linewidths=0, vmin=30, vmax=100); cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.ax.set_title("gas", fontsize=8, color=MUTED); cb.outline.set_visible(False)
    ax.fill_between(xx, kq[1] + tq[0], kq[1] + tq[3], color=BLUE, alpha=0.10, linewidth=0); ax.fill_between(xx, kq[1] + tq[1], kq[1] + tq[2], color=BLUE, alpha=0.18, linewidth=0); ax.plot(xx, kq[1], color=BLUE, linewidth=2)
    ax.set_xlim(-12, 75); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("German residual load, GW"); ax.set_ylabel("price / gas-fired cost"); clean(ax)
    ttl(ax, "Level 1  One curve, fitted Mar 2024 - Nov 2025\n(bands: where 50% and 90% of hours should fall)" if c_ == 0 else "Level 1  Same curve, held-out year (gas doubled):\nthe shape holds, the level sits low")
    r = dat.y - np.interp(dat.rl, K, km); out[f"level 1 coverage of the 50% / 90% band, {lab}"] = (float(np.mean((r > tq[1]) & (r < tq[2]))), float(np.mean((r > tq[0]) & (r < tq[3])))); out[f"level 1 mean / median residual, {lab}"] = (float(r.mean()), float(r.median()))
# replicated against observed quantiles by load band
ax = fig.add_subplot(gs[1, 6:9]); bands = np.arange(-10, 71, 10); cen = bands[:-1] + 5; qs = [0.05, 0.25, 0.5, 0.75, 0.95]; mod = np.array([np.interp(c, K, km) + stats.t.ppf(qs, nu) * sig for c in cen]); obs = np.array([tr.y[(tr.rl >= a) & (tr.rl < a + 10)].quantile(qs).values if ((tr.rl >= a) & (tr.rl < a + 10)).sum() > 50 else [np.nan] * 5 for a in bands[:-1]])
ax.fill_between(cen, mod[:, 0], mod[:, 4], color=BLUE, alpha=0.10, linewidth=0, label="model: 5-95%"); ax.fill_between(cen, mod[:, 1], mod[:, 3], color=BLUE, alpha=0.2, linewidth=0, label="model: 25-75%"); ax.plot(cen, mod[:, 2], color=BLUE, linewidth=2)
for i, q in enumerate(qs): ax.plot(cen, obs[:, i], "-o", color=INK, linewidth=0.9, markersize=3.5, label="observed 5, 25, 50, 75, 95%" if i == 0 else None)
ax.set_xlim(-12, 75); ax.set_ylim(-0.9, 2.6); ax.set_xlabel("German residual load, GW (10 GW bands)"); ax.set_ylabel("price / gas-fired cost"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "Level 1  Spread, model against observed:\none noise size is wrong at both ends"); out["level 1 observed 5-95% width by band"] = tuple(float(v) for v in (obs[:, 4] - obs[:, 0])); out["level 1 model 5-95% width"] = float(mod[0, 4] - mod[0, 0])
# day-level check
ax = fig.add_subplot(gs[1, 9:12]); r = tr.y.values - np.interp(tr.rl.values, K, km); rep = stats.t.rvs(nu, size=(200, len(r)), random_state=1) * sig; bl = (("below 15 GW", tr.rl.values < 15), ("15-35 GW", (tr.rl.values >= 15) & (tr.rl.values < 35)), ("above 35 GW", tr.rl.values >= 35))
def T_(res, msk):
    g_ = pd.Series(np.where(msk, res, np.nan)).groupby(tr.day.values); v = g_.mean()[g_.count() >= 4]; return float(np.sqrt(np.mean(v ** 2)))
for i, (nm, msk) in enumerate(bl):
    o = T_(r, msk); t_ = np.array([T_(rep[k_], msk) for k_ in range(200)]); lo, md, hi = np.percentile(t_, [5, 50, 95]); ax.plot([lo, hi], [i, i], color=BLUE, linewidth=5, solid_capstyle="round", alpha=0.35, label="replicated (independent hours)" if i == 0 else None); ax.scatter([md], [i], s=40, color=BLUE, zorder=3, edgecolor=SURF, linewidth=0.8); ax.scatter([o], [i], s=64, color=INK, zorder=4, edgecolor=SURF, linewidth=0.8, label="observed" if i == 0 else None); ax.annotate(f"{o / md:.1f} x", (o, i), xytext=(8, -3), textcoords="offset points", fontsize=8, color=MUTED); out[f"level 1 day-level error {nm}: observed / replicated"] = (o, float(md))
ax.set_yticks(range(3)); ax.set_yticklabels([b_[0] for b_ in bl]); ax.set_ylim(-0.6, 3.2); ax.set_xlim(0, None); ax.set_xlabel("typical size of a whole-day error (ratio units)"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax); ttl(ax, "Level 1  Errors come as whole days, 2-3 times\nwhat independent hours would give")
# residuals against what is left out, one scale
D["r"] = D.y - np.interp(D.rl, K, km); D["moy"] = D.index.month; D["t"] = (D.index.year - 2024) * 12 + D.index.month; cand = (("hour of day", "hour", np.arange(-0.5, 24, 1), 0), ("month of the year", "moy", np.arange(0.5, 13, 1), 3), ("calendar month, Mar 2024 on", "t", np.arange(2.5, 36, 1), 6), ("gas price, EUR/MWh", "gas", np.array([25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 110]), 0), ("neighbours' residual load, GW from their mean", "nbs", np.arange(-45, 46, 10), 3), ("temperature, C", "teff", np.arange(-6, 31, 4), 6))
axes = []
for i, (nm, col, edges, _) in enumerate(cand):
    ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2], sharey=axes[0] if axes else None); axes.append(ax); sp_ = {}
    for dat, colr, lab in ((D[D.tr], NEUT, "fitting period"), (D[D.ho], ORANGE, "held-out year")):
        k_ = pd.cut(dat[col], edges); g_ = dat.groupby(k_, observed=True)["r"]; md = g_.median(); q1 = g_.quantile(0.25); q3 = g_.quantile(0.75); n_ = g_.size(); keep = n_ >= 60; xc = np.array([iv.mid for iv in md.index])[keep.values]
        ax.fill_between(xc, q1[keep], q3[keep], color=colr, alpha=0.18, linewidth=0); ax.plot(xc, md[keep], "-o", color=colr, linewidth=1.8, markersize=3.5, label=lab); sp_[lab] = float(md[keep].max() - md[keep].min()) if keep.sum() > 1 else np.nan
    ax.axhline(0, color=INK, linewidth=0.7); ax.set_xlabel(nm, fontsize=8.5); clean(ax); out[f"residual spread across bins, {nm}: fitting / held out"] = (sp_["fitting period"], sp_["held-out year"]); ax.set_title(f"{sp_['fitting period']:.2f} | {sp_['held-out year']:.2f}", loc="left", fontsize=9, color=MUTED)
    if i == 0: ax.set_ylabel("level 1 residual, ratio units\n(median and middle half)"); ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    else: plt.setp(ax.get_yticklabels(), visible=False)
axes[0].set_ylim(-0.45, 0.45)
fig.text(0.125, 0.345, "Level 1  What it leaves out, on one scale. Number above each panel: how far the binned medians range, fitting period | held-out year.", fontsize=9.5, fontweight="bold", color=INK)
fig.savefig(f"{SP}/out/zoom_01.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/zoom_01.png")
for k_, v in out.items(): print(f"  {k_}: " + (", ".join(f"{float(x):.3f}" for x in v) if isinstance(v, tuple) else f"{v:.3f}"))
