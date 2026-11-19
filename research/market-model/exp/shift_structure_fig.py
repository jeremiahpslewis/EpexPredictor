import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
BLUE, ORANGE, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
sets = ["winter", "summer", "autumn", "winter,\nfresh", "summer,\nfresh"]
share = np.array([[46, 44, 10], [34, 56, 10], [34, 57, 9], [37, 52, 11], [33, 57, 11]], float)
ts = {"level": [3.04, 3.48, 2.70, 3.09, 3.05], "hourly profile": [2.95, 4.47, 3.47, 3.69, 4.02], "within hour": [1.45, 1.93, 1.37, 1.71, 1.75]}
st = {"level": [3.23, 3.99, 3.36, 4.29, 3.72], "hourly profile": [3.70, 5.76, 4.00, 4.36, 4.95], "within hour": [1.66, 2.38, 1.48, 1.93, 2.09]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(1, 4, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]}, facecolor=SURF)
a = ax[0]; a.set_facecolor(SURF); left = np.zeros(5); y = np.arange(5)[::-1]
for k, (nm, col) in enumerate((("daily level", GREY), ("hourly profile", BLUE), ("within hour", ORANGE))):
    a.barh(y, share[:, k], left=left, color=col, height=0.62, edgecolor=SURF, linewidth=2, label=nm)
    for i in range(5): a.text(left[i] + share[i, k] / 2, y[i], f"{share[i, k]:.0f}%", ha="center", va="center", color="white" if k else INK, fontsize=9)
    left += share[:, k]
a.set_yticks(y); a.set_yticklabels(sets); a.set_xlim(0, 100); a.set_xticks([]); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=9)
a.set_title("Where the shift forecast is wrong:\nmostly the hourly profile, not the daily level", loc="left", fontsize=10.5, fontweight="bold")
for s in a.spines.values(): s.set_visible(False)
for a, nm in zip(ax[1:], ts):
    a.set_facecolor(SURF); x = np.arange(5); w = 0.36
    a.bar(x - w / 2 - 0.02, ts[nm], w, color=BLUE, label="Timestral"); a.bar(x + w / 2 + 0.02, st[nm], w, color=ORANGE, label="structural form")
    for i in range(5): a.text(x[i] + w / 2 + 0.02, st[nm][i] + 0.06, f"+{100 * (st[nm][i] / ts[nm][i] - 1):.0f}%", ha="center", fontsize=8.5, color=INK2)
    a.set_xticks(x); a.set_xticklabels(sets, fontsize=8.5); a.set_ylim(0, 6.4); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True)
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
    a.tick_params(length=0); a.set_title(f"{nm.capitalize()}: the structural form\nis worse in every set", loc="left", fontsize=10.5, fontweight="bold")
ax[1].set_ylabel("error of this part (RMSE, GW)"); ax[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.text(0.008, 0.012, "Supply shift of the market model, forecast one day ahead on five backtest sets (226 days). Structural forms fitted on the 56 days before each issue.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{SP}/out/shift_structure.png", dpi=200, facecolor=SURF)
