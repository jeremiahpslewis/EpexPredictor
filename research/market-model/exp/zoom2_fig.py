"""Pictures for zoom level 2 (notes 20.4): the daily cycle as a shift of the load."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
OR = LinearSegmentedColormap.from_list("o", ["#f6d3c2", "#eb6834", "#7a2b0b"])
D = pd.read_parquet(f"{SP}/out/zoom2_rows.parquet"); Z = np.load(f"{SP}/out/zoom2_post.npz"); Z1 = np.load(f"{SP}/out/zoom1_post.npz"); K = Z["K"]; kn = Z["kn"]; km = np.median(kn, 0); sig = float(np.median(Z["sigma"])); nu = float(np.median(Z["nu"])); rng = np.random.default_rng(0); out = {}
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig = plt.figure(figsize=(18.5, 13.8)); gs = fig.add_gridspec(3, 12, hspace=0.42, wspace=2.6); hh = np.arange(24)
ax = fig.add_subplot(gs[0, 0:3])
for i in range(50): ax.plot(hh, Z["prior_ah"][i], color=BLUE, linewidth=0.7, alpha=0.3)
ax.set_xlabel("hour of day"); ax.set_ylabel("shift of the load, GW"); ax.set_ylim(-13, 13); clean(ax); ttl(ax, "Level 2  Before data: daily cycles its priors\nallow (50 draws)")
ax = fig.add_subplot(gs[0, 3:6]); q = np.percentile(Z["ah"], [5, 50, 95], axis=0); ax.fill_between(hh, q[0], q[2], color=BLUE, alpha=0.2, linewidth=0); ax.plot(hh, q[1], "-o", color=BLUE, linewidth=2, markersize=4); ax.axhline(0, color=INK, linewidth=0.7); qd = np.percentile(Z["ad"], [5, 50, 95], axis=0)
ax.annotate(f"Saturday {qd[1, 0]:+.1f} GW, Sunday or holiday {qd[1, 1]:+.1f} GW", (0.03, 0.04), xycoords="axes fraction", fontsize=8, color=MUTED); ax.set_xlabel("hour of day"); ax.set_ylabel("shift of the load at which the hour is priced, GW"); ax.set_ylim(-13, 13); clean(ax)
ttl(ax, "Level 2  The daily cycle it finds: ramps priced\nhigher, midday lower (line and 90% band)"); out["hour shift range"] = (float(q[1].min()), float(q[1].max())); out["widest 90% band of an hour shift"] = float((q[2] - q[0]).max())
ax = fig.add_subplot(gs[0, 6:9]); xx = np.linspace(-10, 80, 200)
for z_, col, lab in ((Z1, NEUT, "level 1"), (Z, BLUE, "level 2")):
    kq = np.percentile(np.array([np.interp(xx, z_["K"], k_) for k_ in z_["kn"][::8]]), [5, 50, 95], axis=0); ax.fill_between(xx, kq[0], kq[2], color=col, alpha=0.25, linewidth=0); ax.plot(xx, kq[1], color=col, linewidth=2, label=lab)
ax.set_xlim(-12, 80); ax.set_ylim(-0.5, 4.2); ax.set_xlabel("(effective) residual load, GW"); ax.set_ylabel("price / gas-fired cost"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "Level 2  The coarse picture stayed put below\n60 GW; the unsupported top end came down")
out["k level 1 at 20..60"] = tuple(float(v) for v in np.median(Z1["kn"], 0)[3:8]); out["k level 2 at 20..60"] = tuple(float(v) for v in km[3:8])
ax = fig.add_subplot(gs[0, 9:12]); ho = D[D.ho]; sub = ho.iloc[rng.choice(len(ho), 5000, replace=False)]; tq = stats.t.ppf([0.05, 0.25, 0.75, 0.95], nu) * sig; kmx = np.interp(xx, K, km)
sc = ax.scatter(sub.x, sub.y, s=4, c=sub.gas, cmap=OR, alpha=0.7, linewidths=0, vmin=30, vmax=100); cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.ax.set_title("gas", fontsize=8, color=MUTED); cb.outline.set_visible(False)
ax.fill_between(xx, kmx + tq[0], kmx + tq[3], color=BLUE, alpha=0.10, linewidth=0); ax.fill_between(xx, kmx + tq[1], kmx + tq[2], color=BLUE, alpha=0.18, linewidth=0); ax.plot(xx, kmx, color=BLUE, linewidth=2); ax.set_xlim(-12, 80); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("effective residual load, GW"); ax.set_ylabel("price / gas-fired cost"); clean(ax); ttl(ax, "Level 2  Held-out year against the curve\n(fitted on the earlier period)")
for lab, dat in (("fitting", D[D.tr]), ("held out", ho)): out[f"coverage 50% / 90%, {lab}"] = (float(np.mean((dat.res > tq[1]) & (dat.res < tq[2]))), float(np.mean((dat.res > tq[0]) & (dat.res < tq[3])))); out[f"mean / median residual, {lab}"] = (float(dat.res.mean()), float(dat.res.median()))
# is the hour effect a load shift? residual by hour within load bands, fitting period, level 1 against level 2
tr = D[D.tr]; bl = (("below 20 GW", tr.rl < 20), ("20-40 GW", (tr.rl >= 20) & (tr.rl < 40)), ("above 40 GW", tr.rl >= 40)); axb = []
for i, (nm, m) in enumerate(bl):
    ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3], sharey=axb[0] if axb else None); axb.append(ax); g1 = tr[m].groupby("hour")["res1"].median(); g2 = tr[m].groupby("hour")["res"].median(); n_ = tr[m].groupby("hour").size(); kp = n_ >= 40
    ax.plot(g1.index[kp], g1[kp], "-o", color=NEUT, linewidth=1.8, markersize=3.5, label="level 1"); ax.plot(g2.index[kp], g2[kp], "-o", color=BLUE, linewidth=1.8, markersize=3.5, label="level 2"); ax.axhline(0, color=INK, linewidth=0.7); ax.set_xlabel("hour of day"); clean(ax)
    out[f"hour residual range {nm}: level 1 / level 2"] = (float(g1[kp].max() - g1[kp].min()), float(g2[kp].max() - g2[kp].min())); ttl(ax, f"Level 2  Hour-of-day residual, {nm}:\nrange {g1[kp].max() - g1[kp].min():.2f} -> {g2[kp].max() - g2[kp].min():.2f}")
    if i == 0: ax.set_ylabel("median residual, ratio units"); ax.legend(frameon=False, loc="upper left", fontsize=8)
axb[0].set_ylim(-0.45, 0.45)
ax = fig.add_subplot(gs[1, 9:12]); r = tr.res.values; rep = stats.t.rvs(nu, size=(200, len(r)), random_state=1) * sig; b3 = (("below 15 GW", tr.rl.values < 15), ("15-35 GW", (tr.rl.values >= 15) & (tr.rl.values < 35)), ("above 35 GW", tr.rl.values >= 35))
def T_(res, msk):
    g_ = pd.Series(np.where(msk, res, np.nan)).groupby(tr.day.values); v = g_.mean()[g_.count() >= 4]; return float(np.sqrt(np.mean(v ** 2)))
for i, (nm, msk) in enumerate(b3):
    o = T_(r, msk); t_ = np.array([T_(rep[k_], msk) for k_ in range(200)]); lo, md, hi = np.percentile(t_, [5, 50, 95]); ax.plot([lo, hi], [i, i], color=BLUE, linewidth=5, solid_capstyle="round", alpha=0.35, label="replicated (independent hours)" if i == 0 else None); ax.scatter([md], [i], s=40, color=BLUE, zorder=3, edgecolor=SURF, linewidth=0.8); ax.scatter([o], [i], s=64, color=INK, zorder=4, edgecolor=SURF, linewidth=0.8, label="observed" if i == 0 else None); ax.annotate(f"{o / md:.1f} x", (o, i), xytext=(8, -3), textcoords="offset points", fontsize=8, color=MUTED); out[f"day-level error {nm}: observed / replicated"] = (o, float(md))
ax.set_yticks(range(3)); ax.set_yticklabels([b_[0] for b_ in b3]); ax.set_ylim(-0.6, 3.2); ax.set_xlim(0, None); ax.set_xlabel("typical size of a whole-day error (ratio units)"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax); ttl(ax, "Level 2  Whole-day errors are unchanged:\nstill the largest thing the noise misses")
D["moy"] = D.index.month; D["t"] = (D.index.year - 2024) * 12 + D.index.month; cand = (("hour of day", "hour", np.arange(-0.5, 24, 1)), ("month of the year", "moy", np.arange(0.5, 13, 1)), ("calendar month, Mar 2024 on", "t", np.arange(2.5, 36, 1)), ("gas price, EUR/MWh", "gas", np.array([25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 110])), ("neighbours' residual load, GW from their mean", "nbs", np.arange(-45, 46, 10)), ("temperature, C", "teff", np.arange(-6, 31, 4)))
axes = []
for i, (nm, col, edges) in enumerate(cand):
    ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2], sharey=axes[0] if axes else None); axes.append(ax); sp_ = {}; sp1 = {}
    for dat, colr, lab in ((D[D.tr], NEUT, "fitting period"), (D[D.ho], ORANGE, "held-out year")):
        k_ = pd.cut(dat[col], edges); g_ = dat.groupby(k_, observed=True)["res"]; md = g_.median(); q1 = g_.quantile(0.25); q3 = g_.quantile(0.75); n_ = g_.size(); keep = n_ >= 60; xc = np.array([iv.mid for iv in md.index])[keep.values]
        ax.fill_between(xc, q1[keep], q3[keep], color=colr, alpha=0.18, linewidth=0); ax.plot(xc, md[keep], "-o", color=colr, linewidth=1.8, markersize=3.5, label=lab); sp_[lab] = float(md[keep].max() - md[keep].min()); m1 = dat.groupby(k_, observed=True)["res1"].median(); sp1[lab] = float(m1[keep].max() - m1[keep].min())
    ax.axhline(0, color=INK, linewidth=0.7); ax.set_xlabel(nm, fontsize=8.5); clean(ax); out[f"residual range, {nm}: level 1 -> level 2, fitting | held out"] = (sp1["fitting period"], sp_["fitting period"], sp1["held-out year"], sp_["held-out year"]); ax.set_title(f"{sp1['fitting period']:.2f} -> {sp_['fitting period']:.2f} | {sp1['held-out year']:.2f} -> {sp_['held-out year']:.2f}", loc="left", fontsize=9, color=MUTED)
    if i == 0: ax.set_ylabel("level 2 residual, ratio units\n(median and middle half)"); ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    else: plt.setp(ax.get_yticklabels(), visible=False)
axes[0].set_ylim(-0.45, 0.45)
fig.text(0.125, 0.345, "Level 2  What it leaves out, on the same scale as before. Above each panel: range of the binned medians, level 1 -> level 2, fitting period | held-out year.", fontsize=9.5, fontweight="bold", color=INK)
fig.savefig(f"{SP}/out/zoom_02.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/zoom_02.png")
for k_, v in out.items(): print(f"  {k_}: " + (", ".join(f"{float(x):.3f}" for x in v) if isinstance(v, tuple) else f"{v:.3f}"))
