"""Figure for notes 21.98-21.99: thermal units within the day."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ("top", "right")]
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
T = pd.read_parquet(f"{SP}/out/unit_hours.parquet"); T["su"] = pd.to_datetime(T.day.astype(str).str[:10]).dt.month.between(4, 9).values; FU = (("lignite", "lignite", NEUT), ("coal", "hard coal", INK), ("gas", "gas", ORANGE)); h = np.arange(24) + 0.5
fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.0), gridspec_kw=dict(wspace=0.22, width_ratios=[1.1, 1, 1.2])); ax = axs[0]
for f, lab, col in FU:
    for su, ls in ((True, "-"), (False, (0, (4, 2)))): g = T[(T.fuel == f) & (T.su == su)].groupby("hod").run_cap.mean(); ax.plot(h, g.values, color=col, linewidth=2, linestyle=ls, label=f"{lab}, {'April-September' if su else 'October-March'}")
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(0, None); ax.set_xlabel("hour of day"); ax.set_ylabel("capacity of units that are running, GW"); ax.legend(frameon=False, loc="lower left", fontsize=7.5, ncol=2); clean(ax)
g = T[(T.fuel == "gas") & T.su].groupby("hod").run_cap.mean(); ttl(ax, f"Gas cycles within the day (summer: {g[12]:.1f} GW running at noon,\n{g[19]:.1f} at 19h); lignite and coal hardly move")
ax = axs[1]; w = 0.28
for k, (f, lab, col) in enumerate(FU): S = T[(T.fuel == f) & T.su]; g = S.groupby("hod").starts.sum() / S.day.nunique(); ax.bar(np.arange(24) + (k - 1) * w + 0.5, g.values, w * 0.95, color=col, linewidth=0, label=lab)
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel("hour of day (April-September)"); ax.set_ylabel("capacity started in that hour, GW per day"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ax.grid(False, axis="x")
S = T[T.su]; tot = S.groupby("fuel").starts.sum() / S.day.nunique(); late = {f: S[(S.fuel == f) & S.hod.between(14, 19)].starts.sum() / S[S.fuel == f].day.nunique() for f in ("lignite", "coal", "gas")}; ttl(ax, f"Start-ups are an afternoon event: between 14 and 20h\ngas {late['gas']:.1f} GW a day, hard coal {late['coal']:.1f}, lignite {late['lignite']:.1f}")
ax = axs[2]; J = pd.read_csv(f"{SP}/out/unit_days.csv", index_col=0); J.index = pd.to_datetime(J.index, utc=True).tz_convert("Europe/Berlin"); S = J[J.month.between(4, 9)]; q = S.trough.quantile([0.33, 0.67]).values; grp = (("shallow", S.trough <= q[0]), ("middle", (S.trough > q[0]) & (S.trough <= q[1])), ("deep", S.trough > q[1])); T["dayk"] = pd.to_datetime(T.day, utc=True).dt.tz_convert("Europe/Berlin"); x0 = 0; BLK = (("before dawn 03-06h", 3, 5, 0.3), ("midday 10-15h", 10, 15, 0.6), ("evening 17-20h", 17, 20, 1.0)); keep = {}
for f, lab, col in FU:
    X = T[T.fuel == f]
    for i, (nm, m) in enumerate(grp):
        for k, (bl, h0, h1, al) in enumerate(BLK):
            v = X[X.hod.between(h0, h1)].groupby("dayk").run_cap.mean().reindex(S.index[m]).mean(); keep[(f, nm, k)] = v; ax.bar(x0 + i + (k - 1) * 0.27, v, 0.25, color=col, alpha=al, linewidth=0, label=bl if (f == "lignite" and i == 0) else None); ax.text(x0 + i + (k - 1) * 0.27, v + 0.12, f"{v:.1f}", ha="center", fontsize=7)
    ax.text(x0 + 1, 11.6, lab, ha="center", fontsize=9.5, fontweight="bold", color=INK); x0 += 3.6
ax.set_xticks([i + k * 3.6 for k in range(3) for i in range(3)]); ax.set_xticklabels(["shallow", "middle", "deep"] * 3, fontsize=8); ax.set_xlabel("depth of the same day's midday trough (April-September, evening demand 27-30 GW)"); ax.set_ylabel("capacity of running units, GW"); ax.set_ylim(0, 13.5); ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=3); clean(ax); ax.grid(False, axis="x")
ttl(ax, f"On deep-trough days less is running already before dawn: lignite {keep[('lignite', 'shallow', 0)] - keep[('lignite', 'deep', 0)]:.1f} GW,\ngas {keep[('gas', 'shallow', 0)] - keep[('gas', 'deep', 0)]:.1f} GW less. The evening gap is that same gap, not a failure to restart")
fig.suptitle("Thermal units within the day, from per-unit output (117 units, every second day, Mar 2024 - Nov 2026): the evening shortfall is capacity that is off for days, not units held back or failing to restart", x=0.05, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.03); fig.savefig(f"{SP}/out/units_day.png", dpi=100, bbox_inches="tight"); print("ok")
