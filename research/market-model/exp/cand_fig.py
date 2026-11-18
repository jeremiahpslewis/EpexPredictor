"""Figure: (top) night error of the candidate's two parts and of the trees by load range; (bottom) the supply side: the three states, what they are worth
in the index model, and the candidate with and without them."""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
CFG = [("winter", "hw2", "vw", "win_ts", "median", "two_winter_ts"), ("summer", "hs2", "vs", "ssn_ts", "median", "two_summer_ts"), ("autumn", "ha", "va", "tsl_123_8k", "vint", "two_autumn_ts_d1")]
ld = lambda f, k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float); bins = [-60, 5, 15, 25, 35, 45, 100]; labs = ["< 5", "5-15", "15-25", "25-35", "35-45", "> 45"]
fig = plt.figure(figsize=(13.2, 8.2)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.24); axs = []
for c, (period, run, vrun, tsf, tsk, two) in enumerate(CFG):
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float)[:, :32]; tags = [str(x)[:10] for x in I["cutoffs"]]; s = ld(f"structfc_{run}_ts_dyn2xxh")[:, :32]; m = ld(tsf, tsk)[:, :32]; t = ld(two)[:, :32]
    x = np.stack([np.load(f"{SP}/out/struct_{run}/{g}.npz")["x_ts"][:32] for g in tags]); k = np.digitize(x, bins[1:-1])
    ax = fig.add_subplot(gs[0, c], sharey=axs[0] if axs else None); axs.append(ax)
    for v, col, lab in ((s, BLUE, "structural curve + its layer"), (m, ORANGE, "Timestral price"), (t, AQUA, "trees (yardstick)")):
        y = [np.abs(v - a)[k == i].mean() if (k == i).sum() >= 20 else np.nan for i in range(6)]; ax.plot(range(6), y, "-o", color=col, linewidth=2, markersize=6, markeredgecolor=SURF, markeredgewidth=1, label=lab)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{l}\n{(k == i).sum() // 4} h" for i, l in enumerate(labs)], fontsize=8); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_title(f"{period}", loc="left", fontsize=10, fontweight="bold", color=INK)
    if c == 0: ax.set_ylabel("night MAE, ct/kWh"); ax.legend(frameon=False, loc="upper right", fontsize=8)
    if c == 1: ax.set_xlabel("effective residual load of the forecast hour, GW (hours in the range below)")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axs[0].set_ylim(0, None)
# bottom left: the states
import oc, ocx2
ix = pd.DatetimeIndex(oc.S.idx[: oc.n15 // 4 * 4: 4]); d = pd.DataFrame({"lignite + hard coal": ocx2.SOL, "gas": ocx2.GASY, "neighbours' nuclear": ocx2.NUC}, index=ix).loc["2025-06-01":].resample("W").mean(); d.index = d.index.tz_localize(None)
b1 = fig.add_subplot(gs[1, 0])
for (nm, col, dy) in (("neighbours' nuclear", AQUA, 7), ("lignite + hard coal", BLUE, 7), ("gas", ORANGE, -11)):
    b1.plot(d.index, d[nm], color=col, linewidth=2, label=nm)
b1.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.52), fontsize=8)
b1.set_ylabel("available capacity as measured, GW"); b1.set_ylim(0, 70); b1.grid(True, axis="y", color=GRID, linewidth=0.6); b1.set_axisbelow(True); b1.set_title("Supply side, as known three days back", loc="left", fontsize=10, fontweight="bold", color=INK)
import matplotlib.dates as mdates; b1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9))); b1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); b1.set_xlim(d.index[0], d.index[-1])
def night(v, period): a = np.load(f"{SP}/out/inputs_{period}.npz")["actual"].astype(float); return float(np.abs(v - a)[:, :32].mean())
def bars(ax, pairs, labels, cols, title, ylab):
    w = 0.36
    for i, (period, vals) in enumerate(pairs):
        for k_, v in enumerate(vals):
            ax.bar(i + (k_ - 0.5) * (w + 0.03), v, width=w, color=cols[k_], linewidth=0, label=labels[k_] if i == 0 else None); ax.annotate(f"{v:.2f}", (i + (k_ - 0.5) * (w + 0.03), v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p for p, _ in pairs]); ax.set_ylabel(ylab); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left", fontsize=8); ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK); ax.set_ylim(0, max(max(v) for _, v in pairs) * 1.3)
b2 = fig.add_subplot(gs[1, 1]); bars(b2, [(p, (night(ld(f"ocxs_cn_{p}_ts"), p), night(ld(f"ocxs_c_{p}_ts"), p))) for p in ("winter", "summer", "autumn")], ("without the supply side", "with it"), (NEUT, BLUE), "Index model: what it is worth", "night MAE, ct/kWh")
b3 = fig.add_subplot(gs[1, 2])
if all(os.path.exists(f"{SP}/out/structfc_{vr}_ts_dyn2xxh.npz") for _, _, vr, *_ in CFG):
    pairs = []
    for period, run, vrun, tsf, tsk, two in CFG:
        cv = 0.5 * (ld(f"structfc_{vrun}_ts_dyn2xxh") + ld(tsf, tsk)); np.savez_compressed(f"{SP}/out/candv_{period}.npz", vint=cv.astype(np.float32), cutoffs=np.load(f"{SP}/out/inputs_{period}.npz")["cutoffs"]); pairs.append((period, (night(ld(f"cand_{period}"), period), night(cv, period))))
    bars(b3, pairs, ("candidate as it is", "with the supply side in its curve"), (NEUT, BLUE), "Candidate: the same terms in its curve", "night MAE, ct/kWh")
else: b3.axis("off")
for a_ in (b1, b2, b3):
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
fig.suptitle("Top: where each forecaster is strong at night (Timestral loads). Bottom: the supply-side terms from the offer-curve work.", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.955)
out = f"{SP}/out/cand_fig.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); print("figure:", out)
