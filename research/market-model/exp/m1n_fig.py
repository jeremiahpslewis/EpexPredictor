"""M1n (notes 21.47-21.48): neighbours in the import block. Coefficients, the neighbour term forecast at midnight against the published one, key metric."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ("top", "right")]
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
P = np.load(f"{SP}/out/market1n_post.npz"); kn = P["kn"]; B = np.median(kn, 0) / P["NSD"]; NM = ["France", "Netherlands", "Belgium", "Austria", "Poland", "Czech Rep.", "Switzerland", "Denmark"]
fig, axs = plt.subplots(1, 4, figsize=(19.5, 4.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1, 0.45, 1.2])); axs[2].axis("off"); axs = [axs[0], axs[1], axs[3]]
ax = axs[0]; q = np.percentile(kn, [5, 50, 95], 0); o = np.argsort(q[1]); y = np.arange(8)
ax.hlines(y, q[0][o], q[2][o], color=BLUE, linewidth=2.2); ax.plot(q[1][o], y, "o", color=BLUE, markersize=6); ax.axvline(0, color=INK, linewidth=0.7); ax.set_yticks(y); ax.set_yticklabels([NM[i] for i in o]); ax.set_xlabel("GW of German net imports per sd of the\nneighbour's residual load (90% band, too narrow: hours treated as independent)"); clean(ax)
ttl(ax, "A short neighbour means fewer imports:\nthe split between countries is only loosely pinned")
ax = axs[1]; T, Fc = [], []
for per, col in (("winter", BLUE), ("summer", YEL), ("autumn", ORANGE)):
    Z = np.load(f"{SP}/out/nbmid2_{per}.npz"); t = ((Z["true"].astype(float) - P["NMU"]) @ B)[:, :24].reshape(len(Z["true"]), 6, 4).mean(2).ravel(); f = ((Z["fc"].astype(float) - P["NMU"]) @ B)[:, :24].reshape(len(Z["fc"]), 6, 4).mean(2).ravel(); ax.plot(t, f, "o", color=col, markersize=3, alpha=0.55, label=per, markeredgewidth=0); T.append(t); Fc.append(f)
T = np.concatenate(T); Fc = np.concatenate(Fc); lim = (T.min() - 0.5, T.max() + 0.5); ax.plot(lim, lim, color=INK, linewidth=0.7); ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel("neighbour term from the published figures, GW (hours 00-06)"); ax.set_ylabel("forecast at midnight the day before, GW"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ttl(ax, f"The neighbour term is forecastable:\nerror {np.abs(Fc - T).mean():.2f} GW against a spread of {T.std():.1f} GW")
ax = axs[2]; ROWS = (("Merit order M1 + layer", (0.69, 0.83, 1.50), NEUT), ("M1n, neighbours carried from the day before", (0.70, 0.82, 1.45), NEUT), ("M1n, neighbours forecast (panel Timestral)", (0.65, 0.73, 1.40), BLUE), ("M1n, neighbours' published figures given", (0.66, 0.73, 1.39), AQUA), ("M1 + Timestral price", (0.54, 0.64, 1.15), NEUT), ("M1n forecast + Timestral price", (0.53, 0.60, 1.15), BLUE), ("Timestral price alone", (0.51, 0.65, 1.14), ORANGE)); yy = np.arange(len(ROWS))[::-1]; xo = (0.0, 1.0, 2.0)
for yv, (lab, v, col) in zip(yy, ROWS):
    for k in range(3): ax.barh(yv, v[k] / 1.8 * 0.9, 0.6, left=xo[k], color=col, linewidth=0); ax.text(xo[k] + v[k] / 1.8 * 0.9 + 0.02, yv, f"{v[k]:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5); ax.set_xticks([0.2, 1.2, 2.2]); ax.set_xticklabels(["winter", "summer", "autumn"]); ax.set_xlim(0, 3.05); ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ("top", "right", "bottom")]; ax.set_xlabel("MAE 00:00-06:00, ct/kWh (bars to one scale)")
ttl(ax, "Forecast neighbours cut the model's night error by 6%, 12%, 7%;\nblended with Timestral price the gain survives only in summer")
fig.suptitle("Level M1n: the neighbours' residual loads enter the import block (key metric: price 00:00-06:00, forecast at midnight one day ahead, 146 nights)", x=0.06, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04); fig.savefig(f"{SP}/out/market1n_neighbours.png", dpi=105, bbox_inches="tight"); print("figure ok", round(float(np.abs(Fc - T).mean()), 2), round(float(T.std()), 2))
