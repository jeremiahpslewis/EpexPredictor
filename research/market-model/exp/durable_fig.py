import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; Z = np.load(f"{SP}/out/durable.npz", allow_pickle=True); B = pd.read_csv(f"{SP}/out/durable_blocks.csv", index_col=0)
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
def style(a):
    a.set_facecolor(SURF); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0)
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
mo = pd.PeriodIndex(Z["months"], freq="M").to_timestamp(); fitend = pd.Timestamp("2025-12-01"); n = len(mo) - 1
a = ax[0]; style(a); a.axvspan(mo[0], fitend, color=GRID, alpha=0.6, lw=0); a.axhline(0, color=INK2, lw=1); a.plot(mo[:n], Z["shift_full"][:n], color=BLUE, lw=2, marker="o", ms=3.5)
a.text(mo[1], 10.2, "fit window", color=INK2, fontsize=9); a.text(fitend + pd.Timedelta(days=12), 10.2, "after", color=INK2, fontsize=9); a.set_ylim(-14, 12); a.set_ylabel("monthly mean of the supply shift, GW"); a.set_xticks(pd.to_datetime(["2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"])); a.set_xticklabels(["Jul\n2024", "Jan\n2025", "Jul\n2025", "Jan\n2026", "Jul\n2026"], fontsize=8.5)
a.set_title("The shift swings around zero in the fit window,\nthen drifts: 10 GW too much supply by late 2026", loc="left", fontsize=10.5, fontweight="bold")
a = ax[1]; style(a); hy = list(B.index); x = np.arange(len(hy)); a.axvspan(-0.4, 3.5, color=GRID, alpha=0.6, lw=0); a.axhline(0, color=INK2, lw=1)
for c, col in (("net imports", ORANGE), ("lignite", GREY), ("hard coal", INK2), ("gas", BLUE)):
    a.plot(x, B[c], color=col, lw=2, marker="o", ms=4); a.text(x[-1] + 0.12, B[c].iloc[-1] + {"lignite": -0.28, "hard coal": 0.28}.get(c, 0), c, color=col if col != GREY else INK2, va="center", fontsize=9)
a.set_xticks(x); a.set_xticklabels([h[:4] + "\n" + h[4:] for h in hy], fontsize=8.5); a.set_xlim(-0.4, len(hy) + 0.6); a.set_ylabel("actual output minus the curve, GW")
a.set_title("Net imports age steadily, lignite and hard coal\nonce prices rise; gas holds", loc="left", fontsize=10.5, fontweight="bold")
a = ax[2]; style(a); q = [str(s) for s in Z["q"]]; x = np.arange(len(q)); fz = np.clip(Z["frozen_shape"], -30, 60); rf = np.clip(Z["refit"], -30, 60); a.axhline(0, color=INK2, lw=1); a.axvspan(-0.5, 7.5 - 0.33, color=GRID, alpha=0.6, lw=0)
a.plot(x, fz, color=BLUE, lw=2, marker="o", ms=4, label="fitted on the first year, never again"); a.plot(x[1:], rf[1:], color=ORANGE, lw=2, marker="o", ms=4, label="refitted on the last 85 days")
a.set_xticks(x[::2]); a.set_xticklabels([s[:4] + "\n" + s[4:] for s in q[::2]], fontsize=8.5); a.set_ylim(-32, 62); a.set_ylabel("share of the shift's variance explained, %"); a.legend(frameon=False, fontsize=9, loc="lower left")
a.set_title("Static terms of the shift: frozen is as good as\nrefitted for 18 months, then neither explains much", loc="left", fontsize=10.5, fontweight="bold")
fig.text(0.008, 0.012, "German day-ahead market, March 2024 to November 2026. Curves fitted once on March 2024 to November 2025 (shaded). Middle: mean by half year, at the observed price. Right: hour-of-day pattern, within-hour and demand terms; level set per quarter; values below -30 % are drawn at -30 %.", fontsize=8, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{SP}/out/durable.png", dpi=200, facecolor=SURF)
