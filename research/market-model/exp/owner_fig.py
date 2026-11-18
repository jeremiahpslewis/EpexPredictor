import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
# output in GW at a margin of 30-60 EUR/MWh over the fuel's cost, August to early November (owner_look.py)
G = [("RWE lignite (5.7 GW)", 4.3, 3.4), ("EnBW lignite (1.7 GW)", 1.3, 0.9), ("Saale lignite (0.8 GW)", 0.5, 0.4), ("LEAG lignite (6.0 GW)", 3.4, 3.5), ("EnBW hard coal (2.4 GW)", 1.2, 1.1), ("Other hard coal owners (4.5 GW)", 1.9, 2.1), ("RWE gas, Emsland (1.9 GW)", 0.8, 0.5), ("Other large gas blocks (6.3 GW)", 3.0, 3.1)]
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, a = plt.subplots(figsize=(9.2, 4.9)); y = np.arange(len(G))[::-1]; h = 0.36
a.barh(y + h / 2 + 0.02, [g[1] for g in G], height=h, color=ORANGE, label="Aug-Nov 2025"); a.barh(y - h / 2 - 0.02, [g[2] for g in G], height=h, color=BLUE, label="Aug-Nov 2026")
for yy, g in zip(y, G):
    a.text(g[1] + 0.05, yy + h / 2 + 0.02, f"{g[1]:.1f}", va="center", color=MUTED, fontsize=8.5); a.text(g[2] + 0.05, yy - h / 2 - 0.02, f"{g[2]:.1f}  ({g[2] - g[1]:+.1f})", va="center", color=INK, fontsize=8.5)
a.set_yticks(y); a.set_yticklabels([g[0] for g in G]); a.set_xlabel("output, GW"); a.set_xlim(0, 5.2); a.grid(True, axis="x", color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True); a.legend(frameon=False, loc="lower right")
for s_ in ("top", "right", "left"): a.spines[s_].set_visible(False)
a.tick_params(axis="y", length=0); a.set_title("Output by owner at the same margin: price 30-60 EUR/MWh above the fuel's cost", loc="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{SP}/out/owner_fig.png", dpi=120); print("ok")
