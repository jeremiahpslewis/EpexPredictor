"""Where the market model stands after the units fix (M0d, notes 21.10): storage block, held-out mean shocks, error with and without the carried state
against the reduced-form curve, and the target slots against the candidate. Numbers in panels C and D are those printed by market_horizon.py."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"; LB, LO = "#a9c9ef", "#f7c3ad"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="y"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); H = pd.read_parquet(f"{SP}/out/market0d_rows.parquet"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market0d_post.npz").items() if np.ndim(v) == 1}; H["r"] = H.p / H.L; H["wk"] = ((H.index.tz_convert("UTC") - pd.Timestamp("2024-01-01", tz="UTC")) // pd.Timedelta(days=7)).astype(int)
def binned(df, xcol, ycol, edges, nboot=300, minn=80):
    x = df[xcol].values; y = df[ycol].values; b = np.digitize(x, edges) - 1; nb = len(edges) - 1; uw, wi = np.unique(df.wk.values, return_inverse=True); idx = [np.where(wi == k)[0] for k in range(len(uw))]
    med = np.array([np.median(y[b == i]) if (b == i).sum() >= minn else np.nan for i in range(nb)]); B = np.full((nboot, nb), np.nan)
    for j in range(nboot):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(uw), len(uw))]); bb, yy = b[ii], y[ii]
        for i in range(nb):
            m = bb == i
            if m.sum() >= minn // 2: B[j, i] = np.median(yy[m])
    lo, hi = np.nanpercentile(B, [5, 95], axis=0); lo[np.isnan(med)] = np.nan; hi[np.isnan(med)] = np.nan; return med, lo, hi
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 4, wspace=0.30)
ax = fig.add_subplot(gs[0, 0]); eR = np.r_[-0.6, -0.2, -0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.1, 2.6]; cen = 0.5 * (eR[:-1] + eR[1:]); rr = np.linspace(-0.6, 2.6, 200)
for df, col, lab in ((H[H.tr], NEUT, "fitting period"), (H[~H.tr], ORANGE, "held-out year")):
    med, lo, hi = binned(df, "r", "S", eR); ax.fill_between(cen, lo, hi, color=col, alpha=0.18, linewidth=0); ax.plot(cen, med, "o", color=col, markersize=4, label=lab)
ax.plot(rr, th["K_st"] * (2 * norm.cdf((rr - th["m_st"]) / th["s_st"]) - 1), color=BLUE, linewidth=2.2, label="storage block"); ax.axhline(0, color=INK, linewidth=0.6); ax.set_xlabel("price / gas-plus-CO2 cost"); ax.set_ylabel("pumped storage, generation minus pumping, GW"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax, "both")
ttl(ax, "Storage as its own block: one curve in\nprice / cost; the held-out year sits on it")
ax = fig.add_subplot(gs[0, 1]); blocks = (("thermal", "eT"), ("net imports", "eX"), ("storage", "eS"), ("unmeasured part", "eps")); y = np.arange(4)[::-1]
for i, (nm, c) in enumerate(blocks):
    for df, col, off in ((H[H.tr], NEUT, 0.14), (H[~H.tr], ORANGE, -0.14)):
        wm = df.groupby("wk")[c].mean().values; bs = [np.mean(rng.choice(wm, len(wm))) for _ in range(400)]; lo, hi = np.percentile(bs, [5, 95]); ax.plot([lo, hi], [y[i] + off] * 2, color=col, linewidth=4, alpha=0.45, solid_capstyle="round"); ax.scatter([df[c].mean()], [y[i] + off], s=38, color=col, zorder=3, edgecolor=SURF, linewidth=0.8, label=("fitting period" if col == NEUT else "held-out year") if i == 0 else None)
ax.axvline(0, color=INK, linewidth=0.8); ax.set_yticks(y); ax.set_yticklabels([b[0] for b in blocks]); ax.set_xlabel("mean observed minus curve, GW (5-95% over weeks)"); ax.set_xlim(-4.6, 4.8); ax.legend(frameon=False, loc="center right", fontsize=8); clean(ax, "x"); ho = H[~H.tr]
ttl(ax, f"Held-out year by block: imports {ho.eX.mean():+.1f} GW\noff their curve, storage on it")
ax = fig.add_subplot(gs[0, 2]); grp = ("first version\n(rest in EUR)", "units fixed", "reduced form\n(level 1)"); none = (3.23, 2.67, 2.27); st = (1.82, 1.88, 1.93); x = np.arange(3); w = 0.3
for xi, a, b, c1, c2 in zip(x, none, st, (LB, LB, LO), (BLUE, BLUE, ORANGE)):
    ax.bar(xi - w / 2 - 0.02, a, w, color=c1, linewidth=0); ax.bar(xi + w / 2 + 0.02, b, w, color=c2, linewidth=0); ax.text(xi - w / 2 - 0.02, a + 0.05, f"{a:.2f}", ha="center", fontsize=8.5, color=INK); ax.text(xi + w / 2 + 0.02, b + 0.05, f"{b:.2f}", ha="center", fontsize=8.5, color=INK)
ax.bar([np.nan], [np.nan], color=LB, label="curves only"); ax.bar([np.nan], [np.nan], color=BLUE, label="with last week's state carried"); ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylim(0, 3.7); ax.set_ylabel("price error two days ahead, ct/kWh"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax)
ttl(ax, "Units fix: 0.56 ct off the bare model; with\nthe state carried all three are within 0.11 ct")
ax = fig.add_subplot(gs[0, 3]); per = ("winter", "summer", "autumn"); mm = (1.39, 1.39, 2.70); l1 = (1.54, 1.38, 2.49); cd = (0.67, 0.79, 1.31); w = 0.24
for j, (v, col, lab) in enumerate(((mm, BLUE, "market model + state (true load)"), (l1, ORANGE, "reduced form + state (true load)"), (cd, AQUA, "candidate (forecast load)"))):
    ax.bar(x + (j - 1) * (w + 0.02), v, w, color=col, linewidth=0, label=lab)
    for xi, a in zip(x, v): ax.text(xi + (j - 1) * (w + 0.02), a + 0.04, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylim(0, 3.2); ax.set_ylabel("night slots of D+2, price error, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ttl(ax, "Target slots: level with the reduced form,\n75-110% behind the candidate")
fig.suptitle("Market model after the units fix (M0d): what transfers, what the carried state is worth, and where it stands on the target", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market0d_where.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0d_where.png")
