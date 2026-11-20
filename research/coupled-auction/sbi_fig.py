import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = Path(sys.argv[1]); R = json.load(open(D / "sbi_results.json"))
BLUE, ORANGE, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
lab = {"markup": ("Markup (EUR/MWh)", "recovered"), "netting": ("Netting share", "not identified, as it should be"), "flex_wtp": ("Flexible demand, willingness to pay", "partly")}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.9), facecolor=SURF)
for a, nm in zip(ax, ["markup", "netting", "flex_wtp"]):
    p, o = R["params"][nm], R["observed"][nm]; t, m = np.array(p["truth"]), np.array(p["post_mean"])
    a.set_facecolor(SURF); a.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    lo, hi = min(t.min(), m.min()), max(t.max(), m.max()); a.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    a.scatter(t, m, s=10, color=BLUE, alpha=0.5, lw=0)
    a.errorbar([o["truth"]], [o["median"]], yerr=[[o["median"] - o["lo"]], [o["hi"] - o["median"]]], fmt="o", color=ORANGE, ms=7, capsize=4, lw=2)
    a.set_xlabel("true value"); a.set_ylabel("posterior mean")
    a.set_title(f"{lab[nm][0]}: {lab[nm][1]}\ncontraction {p['contraction_mean']:.2f}, 90% intervals cover {p['coverage90']:.0%}", loc="left", fontsize=10, fontweight="bold")
ax[0].text(0.03, 0.96, "orange: the observed dataset,\nposterior median and 90% interval", transform=ax[0].transAxes, va="top", fontsize=9, color=ORANGE)
fig.text(0.008, 0.012, f"BayesFlow 2.1.0, coupling flow with a DeepSet over five synthetic days; {R['n_test']} held-out datasets (blue). Dashed: posterior mean equals truth.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(D / "sbi_recovery.png", dpi=170, facecolor=SURF)
