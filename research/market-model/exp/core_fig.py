import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
BLUE, ORANGE, AQUA, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
sets = ["winter", "summer", "autumn", "winter, fresh", "summer, fresh"]
day = [87, 65, 58, 59, 49]; night = [73, 76, 78, 76, 47]; left = [65, 54, 63, 65, 54]
J = {"core with carry-over taken out first": ([5.7, 4.6, 1.4, 9.2, 12.2], [25.2, 20.3, 6.8, 19.5, 27.4], ORANGE), "smooth core taken out first": ([4.4, 3.5, 0.9, -0.3, 8.7], [13.0, -1.5, 1.0, 2.8, 5.5], BLUE)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1, 1]})
a = ax[0]; a.set_facecolor(SURF); y = np.arange(5)[::-1]; h = 0.34
a.barh(y + h / 2 + 0.02, [100] * 5, h, color=GRID); a.barh(y - h / 2 - 0.02, [100] * 5, h, color=GRID)
a.barh(y + h / 2 + 0.02, day, h, color=BLUE, label="whole day"); a.barh(y - h / 2 - 0.02, night, h, color=AQUA, label="night, 0 to 6 h")
for i in range(5): a.text(day[i] - 2, y[i] + h / 2 + 0.02, f"{day[i]}%", ha="right", va="center", color="white", fontsize=9); a.text(night[i] - 2, y[i] - h / 2 - 0.02, f"{night[i]}%", ha="right", va="center", color="white", fontsize=9)
a.set_yticks(y); a.set_yticklabels(sets); a.set_xlim(0, 100); a.set_xticks([0, 50, 100]); a.set_xticklabels(["7-day mean\nof the shift", "", "Timestral on\nthe whole shift"], fontsize=9); a.tick_params(length=0)
for s in a.spines.values(): s.set_visible(False)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
a.set_title("The state core alone closes half to most\nof the price-error gap (no Timestral)", loc="left", fontsize=10.5, fontweight="bold")
for a, k, nm in ((ax[1], 0, "Whole day: joined with Timestral,\nwithin a few percent of it"), (ax[2], 1, "Night: the full core costs 7 to 27%,\nthe smooth core is level in four sets")):
    a.set_facecolor(SURF); x = np.arange(5); w = 0.36
    for m, (lab, v) in enumerate(J.items()):
        b = a.bar(x + (m - 0.5) * (w + 0.04), v[k], w, color=v[2], label=lab)
        for i in range(5): a.text(x[i] + (m - 0.5) * (w + 0.04), v[k][i] + (0.6 if v[k][i] >= 0 else -1.8), f"{v[k][i]:+.0f}", ha="center", fontsize=8.5, color=INK2)
    a.axhline(0, color=INK2, lw=1); a.set_xticks(x); a.set_xticklabels([s.replace(", ", ",\n") for s in sets], fontsize=8.5); a.set_ylim(-4, 30); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"): a.spines[s].set_visible(False)
    a.set_title(nm, loc="left", fontsize=10.5, fontweight="bold")
ax[1].set_ylabel("price error vs. Timestral on the whole shift, %", fontsize=9); ax[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.text(0.008, 0.012, "Market model route, five backtest sets (226 days), forecasts at midnight for the day that follows. \nState core: usual level per quarter hour, demand response, carry-over of the last hour and of yesterday. Smooth core: without the carry-over and the working day / weekend split.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{SP}/out/core_ledger.png", dpi=200, facecolor=SURF)
