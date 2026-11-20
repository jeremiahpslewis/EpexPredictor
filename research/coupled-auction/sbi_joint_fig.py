"""Figure of the joint recovery and its counterfactuals (notes 21.160). Usage: sbi_joint_fig.py DIR -> DIR/sbi_joint.png"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path(sys.argv[1]); C = json.load(open(D / "cf2.json")); BY = C["draws_by_set"]
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#e8613c", "#222222", "#555555", "#e6e6e6", "#fbfbf9"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.facecolor": BG, "figure.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.9))
P = {t: np.load(D / f"post2_{t}.npz") for t in ("noavail", "avail")}; truth = P["avail"]["truth"]
a = ax[0]
for t, col, lab in (("noavail", ORANGE, "prices and flows only"), ("avail", BLUE, "plus published availability")):
    d = P[t]["draws"][::4]; a.scatter(d[:, 2], d[:, 1], s=6, color=col, alpha=0.35, lw=0, label=lab)
f = np.linspace(0.0, 0.14, 50); a.plot(f, truth[1] * (1 - truth[2]) / (1 - f), color=INK2, lw=1, ls="--")
a.text(0.098, 0.985, "same offered product\nas the truth", color=INK2, fontsize=8.5, ha="right")
a.scatter([truth[2]], [truth[1]], marker="x", color=INK, s=60, zorder=5); a.text(truth[2] + 0.003, truth[1] - 0.012, "truth", color=INK, fontsize=9)
a.set_xlabel("mean forced-outage share"); a.set_ylabel("offered share of available thermal (pi)"); a.set_xlim(0, 0.14); a.set_ylim(0.84, 1.0); a.legend(frameon=False, loc="lower left", fontsize=8.5)
a.set_title("Prices alone leave a ridge; published\navailability cuts it short", loc="left", fontweight="bold", fontsize=10.5)
for k, (j, lab, title) in enumerate(((3, "change in B's mean price, ATC 1000 -> 500", "The transmission effect follows the offered\nproduct, which prices pin only to +-3%"),
                                     (4, "change in A's mean price, gas 40 -> 52", "So does the gas effect: availability\nhalves its spread"))):
    a = ax[k + 1]
    for t, col in (("posterior_noavail", ORANGE), ("posterior_avail", BLUE)):
        A = np.array([d for d in BY[t] if len(d) == 7]); a.scatter(A[:, 1] * (1 - A[:, 2]), A[:, j], s=12, color=col, alpha=0.6, lw=0)
    T = np.array(BY["truth"][0]); a.scatter([T[1] * (1 - T[2])], [T[j]], marker="x", color=INK, s=60, zorder=5)
    s = C["posterior_noavail"]["atc_dB" if j == 3 else "gas_dA"]; s2 = C["posterior_avail"]["atc_dB" if j == 3 else "gas_dA"]; pr = C["prior"]["atc_dB" if j == 3 else "gas_dA"]
    a.text(0.02, 0.04, f"90%: prior {pr[0]:+.1f} to {pr[2]:+.1f}\nprices only {s[0]:+.1f} to {s[2]:+.1f}\nwith availability {s2[0]:+.1f} to {s2[2]:+.1f}\ntruth {T[j]:+.1f}",
           transform=a.transAxes, fontsize=8.5, color=INK2, va="bottom")
    a.set_xlabel("offered product: pi x (1 - forced-outage share)"); a.set_ylabel(lab); a.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
for a in ax: a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
fig.text(0.01, 0.01, "Synthetic: 5,000 datasets of five days from the slice; markup, pi and forced-outage share free, the rest at the truth. Posteriors for one observed dataset; "
         "counterfactuals averaged over three common days, 120 posterior draws each.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(D / "sbi_joint.png", dpi=170)
