"""Fit assessment, figure 3: the form of the gas response in raw data (P10, P11)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
GREY = LinearSegmentedColormap.from_list("g", ["#dcdbd5", "#8f8e87", "#2b2a27"])
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
fig = plt.figure(figsize=(16.6, 5.2)); gs = fig.add_gridspec(1, 3, wspace=0.42)
R = pd.read_csv(f"{SP}/out/fit_p10.csv"); z = np.load(f"{SP}/out/fit_p10.npz"); ax = fig.add_subplot(gs[0, 0]); cr = float(z["LB"] / z["LA"])
ax.axhline(cr, color=NEUT, linewidth=1.4, linestyle=(0, (5, 2)), label=f"gas-fired cost rose by this factor ({cr:.2f})")
ax.plot(R.x, R.cand, color=BLUE, linewidth=2, label="candidate's gas term alone"); ax.plot(R.x, R.prior, color=ORANGE, linewidth=2, label="gas term alone, with the tight prior")
ax.errorbar(R.x, R.ratio, yerr=[R.ratio - R.lo, R.hi - R.ratio], fmt="o", color=INK, markersize=6, markeredgecolor=SURF, elinewidth=1.4, capsize=0, label="observed (5-95% over days)", zorder=4)
ax.set_ylim(1.0, 3.2); ax.set_xlabel("effective residual load, GW"); ax.set_ylabel("price in Jul-Sep 2026 / price in Jul-Sep 2025"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax)
ax.set_title("P10  A year apart, prices from 25 GW up rose by the\nsame factor as gas-fired cost, at every load level", loc="left", fontsize=9.5, fontweight="bold", color=INK)
D = pd.read_parquet(f"{SP}/out/fit_p11.parquet"); b = np.load(f"{SP}/out/fit_p11.npz"); be, lo, hi = b["b"], b["lo"], b["hi"]
for c_, (xcol, ycol, e_, l_, h_, xl, ttl) in enumerate((("slow", "part_slow", be[1], lo[1], hi[1], "slow part: 28-day mean of log gas-fired cost", "P11  Slow moves in gas-fired cost reach the price about\none for one (one gas rise; 0.85-1.0 is the honest range)"), ("fast", "part_fast", be[2], lo[2], hi[2], "fast part: the day's deviation from that mean (log)", "P11  A day's deviation reaches it at about\nthree quarters"))):
    ax = fig.add_subplot(gs[0, c_ + 1]); x = D[xcol] - (D[xcol].mean() if xcol == "slow" else 0.0); y = D[ycol] - (be[0] if xcol == "slow" else 0.0)
    sc = ax.scatter(x, y, c=D.mo, cmap=GREY, s=10, linewidths=0, alpha=0.85); xs = np.array([x.quantile(0.01), x.quantile(0.99)])
    ax.plot(xs, xs * 1.0, color=NEUT, linewidth=1.4, linestyle=(0, (5, 2)), label="one for one"); ax.plot(xs, xs * e_, color=BLUE, linewidth=2.2, label=f"fitted: {e_:.2f}")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, ticks=[5, 9, 13, 17, 21]); cb.ax.set_yticklabels(["May 25", "Sep 25", "Jan 26", "May 26", "Sep 26"]); cb.outline.set_visible(False)
    ax.set_xlabel(xl); ax.set_ylabel("day's price level against the load curve (log)"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ax.set_title(ttl, loc="left", fontsize=9.5, fontweight="bold", color=INK)
    if xcol == "fast": ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.45, 0.45)
fig.savefig(f"{SP}/out/fit_gas_3.png", dpi=110, bbox_inches="tight"); print("figure:", f"{SP}/out/fit_gas_3.png")
