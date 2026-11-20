import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = Path(sys.argv[1]); R = json.load(open(D / "demo.json")); h = np.arange(96) / 4
BLUE, ORANGE, AQUA, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURF)
def style(a, ylab):
    a.set_facecolor(SURF); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0); a.set_ylabel(ylab); a.set_xticks([0, 6, 12, 18, 24]); a.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
b = R["base"]
a = ax[0, 0]; style(a, "price, EUR/MWh"); a.step(h, b["price_A"], where="post", color=BLUE, lw=2); a.step(h, b["price_B"], where="post", color=GREY, lw=2)
a.text(23.9, b["price_A"][-1] + 4, "zone A", color=BLUE, ha="right", fontsize=9.5); a.text(23.9, b["price_B"][-1] - 9, "zone B", color=INK2, ha="right", fontsize=9.5)
a.set_title("One draw of a full day: zone A falls below zero at midday;\nzone B is priced by hydro at night and imports A's solar at midday", loc="left", fontsize=10.5, fontweight="bold")
a = ax[0, 1]; style(a, "flow A to B, MW"); a.axhline(1000, color=INK2, lw=1, ls="--"); a.axhline(-1000, color=INK2, lw=1, ls="--"); a.step(h, b["flow"], where="post", color=BLUE, lw=2)
a.text(0.2, 1060, "limit", color=INK2, fontsize=9); a.set_ylim(-1300, 1300)
a.set_title(f"The line is at its limit in {b['at_limit']} of 96 quarters:\nA imports at night and exports at midday", loc="left", fontsize=10.5, fontweight="bold")
a = ax[1, 0]; style(a, "inventory, MWh"); a.axhline(6000, color=INK2, lw=1, ls="--"); a.axhline(0, color=INK2, lw=1)
a.plot(np.arange(97) / 4, b["soc_A"], color=BLUE, lw=2); a.text(0.2, 6150, "capacity 6,000 MWh", color=INK2, fontsize=9); a.set_ylim(-300, 6800)
a.text(24, b["soc_A"][-1] + 200, f"closes at {b['closing_soc']['A']:,.0f}", color=BLUE, ha="right", fontsize=9.5)
a.set_title("Zone A storage bids only what its inventory allows:\nit charges at midday and discharges when prices are high", loc="left", fontsize=10.5, fontweight="bold")
a = ax[1, 1]; style(a, "change in B's price, EUR/MWh"); a.axhline(0, color=INK2, lw=1)
d1 = np.array(R["cf_atc"]["price_B"]) - np.array(b["price_B"]); d2 = np.array(R["book2_cf_atc"]["price_B"]) - np.array(b["price_B"])
a.step(h, d1, where="post", color=BLUE, lw=2); a.step(h, d2, where="post", color=ORANGE, lw=2)
lo_, hi_ = min(d1.min(), d2.min()) - 4, max(d1.max(), d2.max()) + 4; a.set_ylim(lo_, hi_)
a.text(0.3, hi_ - 0.12 * (hi_ - lo_), f"book 2: mean {d2.mean():+.1f}", color=ORANGE, fontsize=9.5); a.text(0.3, lo_ + 0.08 * (hi_ - lo_), f"book 1: mean {d1.mean():+.1f}", color=BLUE, fontsize=9.5)
a.set_title(f"Two books fit the same day exactly, yet halving the line\nmoves B's price by up to {np.abs(d1).max():.0f} or {np.abs(d2).max():.0f} EUR/MWh in a quarter", loc="left", fontsize=10.5, fontweight="bold")
fig.text(0.008, 0.012, "Synthetic two-zone day, 96 quarter hours, one posterior-style draw (research/coupled-auction/demo.py). Book 2 = book 1 plus 1,500 MW of buy and sell orders 3 EUR/MWh either side of each cleared price.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D / "vslice_day.png", dpi=170, facecolor=SURF)
