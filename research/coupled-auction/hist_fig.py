"""Figure of the historical prior-predictive check (hist_window.py output). Usage: hist_fig.py DIR -> DIR/hist_window.png"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path(sys.argv[1]); R = pd.read_parquet(D / "hist_window.parquet").tz_convert("Europe/Berlin"); S = json.load(open(D / "hist_window.json"))
R = R.reindex(pd.date_range(R.index.min(), R.index.max(), freq="15min"))      # the skipped clock-change day stays a gap
cap = int((R.p95 > 600).sum())
BLUE, ORANGE, INK2, GRID, BG = "#2a78d6", "#e8613c", "#555555", "#e6e6e6", "#fbfbf9"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.facecolor": BG, "figure.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(11, 9.6), sharex=True)
x = R.index
a = ax[0]; a.fill_between(x, R.p05, R.p95, step="post", color=BLUE, alpha=0.18, lw=0); a.step(x, R.p50, where="post", color=BLUE, lw=1.4)
a.step(x, R.obs_price, where="post", color=ORANGE, lw=1.4); a.set_ylabel("EUR/MWh"); a.set_ylim(-60, 600)
a.text(0.995, 0.93, f"band clipped at 600 in {cap} quarters (some draws reach the 4,000 cap)", color=INK2, transform=a.transAxes, ha="right", fontsize=8.5)
a.text(0.005, 0.93, "observed DE-LU price", color=ORANGE, transform=a.transAxes); a.text(0.005, 0.84, "slice, prior median and 90% band", color=BLUE, transform=a.transAxes)
a.set_title(f"Observed prices sit {S['price_bias_median']:+.0f} EUR/MWh above the slice's prior median and inside its 90% band in only "
            f"{S['price_inside_90']:.0%} of quarters", loc="left", fontweight="bold", fontsize=10.5)
a = ax[1]; a.axhspan(-1000, 1000, color=BLUE, alpha=0.18, lw=0); a.step(x, R.obs_netpos, where="post", color=ORANGE, lw=1.4); a.axhline(0, color=INK2, lw=0.6)
a.set_ylabel("MW, + = export"); a.text(0.005, 0.9, "observed day-ahead net position", color=ORANGE, transform=a.transAxes)
a.text(0.005, 0.81, "what the slice's one line allows (+-1,000 MW)", color=BLUE, transform=a.transAxes)
a.set_title(f"Germany's net position leaves the slice's +-1 GW in {S['obs_netpos_abs_over_1000']:.0%} of quarters and is an export on average "
            f"({S['obs_netpos_mean']:+,} MW)", loc="left", fontweight="bold", fontsize=10.5)
a = ax[2]; a.fill_between(x, R.gen_p05 / 1000, R.gen_p95 / 1000, step="post", color=BLUE, alpha=0.18, lw=0); a.step(x, R.gen_p50 / 1000, where="post", color=BLUE, lw=1.4)
a.step(x, R.obs_gen / 1000, where="post", color=ORANGE, lw=1.4); a.step(x, R.load / 1000, where="post", color=INK2, lw=0.9, ls="--"); a.set_ylabel("GW")
a.text(0.005, 0.9, "observed scheduled generation", color=ORANGE, transform=a.transAxes); a.text(0.005, 0.81, "slice: awards + self-served, median and 90% band", color=BLUE, transform=a.transAxes)
a.text(0.005, 0.72, "load forecast (input)", color=INK2, transform=a.transAxes)
a.set_title(f"Scheduled generation exceeds the slice's by {S['gen_bias_median_MW'] / 1000:+.1f} GW on average, while net exports average {S['obs_netpos_mean'] / 1000:+.1f} GW",
            loc="left", fontweight="bold", fontsize=10.5)
for a in ax: a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
fig.text(0.01, 0.005, f"{S['days']} delivery days from {x.min():%d %b %Y}, {S['draws_per_day']} prior draws per day; zone A fed DE-LU's public day-ahead inputs "
         f"\n(ENTSO-E load, wind and solar forecasts; THE gas {S['gas_range'][0]:.0f}-{S['gas_range'][1]:.0f} EUR/MWh; carbon 75). Zone B is synthetic. Prior-predictive check, not a fit.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(D / "hist_window.png", dpi=170)
