"""Figure for notes 21.88-21.89: the predictive distributions over the whole day and at the peaks."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ("top", "right")]
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
Z = np.load(f"{SP}/out/day_dist.npz"); TAU = np.array([.05, .10, .50, .90, .95]); rng = np.random.default_rng(0); PERS = (("winter", "winter, used", 3), ("summer", "summer, used", 3), ("autumn", "autumn, used", 7), ("winterf", "winter, fresh", 3), ("summerf", "summer, fresh", 3)); RT = (("Ap", "market model, shift + load uncertainty", BLUE), ("P", "Timestral on the price", ORANGE), ("V", "average of the two, quantile by quantile", AQUA))
def pin(Q, a): d = a[None] - Q; return np.maximum(TAU[:, None, None] * d, (TAU[:, None, None] - 1) * d)
fig = plt.figure(figsize=(19.5, 10.6)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.24, height_ratios=[1, 1.05])
# reliability, all slots and forecast-peak slots
for c_, (lab, peak) in enumerate((("all slots of the day", False), ("the tenth of slots with the highest forecast (known at issue)", True))):
    ax = fig.add_subplot(gs[0, c_]); nom = [0.05, 0.10, 0.10, 0.05]; x = np.arange(4); w = 0.26; res = {}
    for k_, (k, nm, col) in enumerate(RT):
        cnt = np.zeros(4); n = 0
        for p, _, _ in PERS:
            a = Z[f"{p}_a"].astype(float); Q = Z[f"{p}_{k}"].astype(float); m = (Q[2] >= np.quantile(Q[2], 0.9)) if peak else np.ones_like(a, bool); cnt += [(a < Q[0])[m].sum(), (a < Q[1])[m].sum(), (a > Q[3])[m].sum(), (a > Q[4])[m].sum()]; n += m.sum()
        res[k] = cnt / n; ax.bar(x + (k_ - 1) * w, cnt / n, w * 0.92, color=col, linewidth=0, label=nm)
    for i, v in enumerate(nom): ax.hlines(v, i - 0.45, i + 0.45, color=INK, linewidth=1.4, linestyle=(0, (4, 2)), label="nominal" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(["below the\n5% quantile", "below the\n10% quantile", "above the\n90% quantile", "above the\n95% quantile"]); ax.set_ylabel("share of slots"); ax.set_ylim(0, 0.16); clean(ax); ax.grid(False, axis="x")
    if c_ == 0: ax.legend(frameon=False, loc="upper center", fontsize=8); ttl(ax, f"Over the whole day both are close to nominal; the price route\nis below its 10% quantile too often ({res['P'][1]:.0%} against 10%)")
    else: ttl(ax, f"Where a peak is forecast: market model on target ({res['Ap'][3]:.1%} above\nits 95% quantile), price route too high ({res['P'][3]:.1%} above, {res['P'][1]:.0%} below its 10%)")
ax = fig.add_subplot(gs[0, 2]); y = np.arange(5)[::-1]
for k, off, col, lab in (("Ap", 0.17, BLUE, "market model minus price route"), ("V", -0.17, AQUA, "average minus price route")):
    for yy, (p, nm, blk) in zip(y, PERS):
        a = Z[f"{p}_a"].astype(float); d = pin(Z[f"{p}_{k}"].astype(float), a).mean((0, 2)) - pin(Z[f"{p}_P"].astype(float), a).mean((0, 2)); base = pin(Z[f"{p}_P"].astype(float), a).mean(); nb = len(d) // blk; v = d[:nb * blk].reshape(nb, blk).mean(1); lo, hi = np.percentile([v[rng.integers(0, nb, nb)].mean() for _ in range(4000)], [5, 95]) / base * 100; ax.hlines(yy + off, lo, hi, color=col, linewidth=2.4); ax.plot(d.mean() / base * 100, yy + off, "o", color=col, markersize=6.5, label=lab if yy == y[0] else None); ax.text(hi + 0.6, yy + off - 0.07, f"{d.mean() / base * 100:+.0f}%", fontsize=8, color=INK)
ax.axvline(0, color=INK, linewidth=0.7); ax.set_yticks(y); ax.set_yticklabels([s[1] for s in PERS]); ax.set_xlabel("difference in mean pinball loss over five quantile levels, whole day, % (bar: 5-95%)"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax); ax.grid(False, axis="y"); ax.set_ylim(-0.7, 4.9); ax.set_xlim(-28, 30); ttl(ax, "The quantile average beats the price route in all five\nsets (clearly in two); the market model alone only in winter")
# fan chart: the summer day with the highest evening price
a = Z["summer_a"].astype(float); j = int(np.argmax(a[:, 64:88].max(1))); t = np.arange(96) / 4
for c_, (k, nm, col) in enumerate(RT):
    ax = fig.add_subplot(gs[1, c_]); Q = Z[f"summer_{k}"].astype(float)[:, j]; ax.fill_between(t, Q[0], Q[4], color=col, alpha=0.18, linewidth=0, label="5-95%"); ax.fill_between(t, Q[1], Q[3], color=col, alpha=0.3, linewidth=0, label="10-90%"); ax.plot(t, Q[2], color=col, linewidth=1.8, label="median"); ax.plot(t, a[j], color=INK, linewidth=1.5, label="actual"); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel("hour of the target day"); ax.set_ylabel("ct/kWh"); ax.set_ylim(min(-6, a[j].min() - 2), max(a[j].max(), Z["summer_P"][4, j].max(), Z["summer_Ap"][4, j].max()) + 2); clean(ax)
    if c_ == 0: ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    pk = slice(64, 88); ttl(ax, f"{nm[0].upper() + nm[1:]}\nevening peak {a[j, pk].max():.0f} ct; its 95% quantile there reaches {Q[4, pk].max():.0f} ct, median {Q[2, pk].max():.0f}")
fig.text(0.125, 0.475, f"One day in detail: the summer day with the highest evening price in the used set (day {j + 1} of 50)", fontsize=10, fontweight="bold", color=INK)
fig.suptitle("Predictive distributions over the whole target day, forecast at midnight the day before (five sets pooled in the top row)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.955); fig.savefig(f"{SP}/out/day_dist.png", dpi=95, bbox_inches="tight"); print("ok", j)
