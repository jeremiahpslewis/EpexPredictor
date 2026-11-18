"""Merit order with and without block orders across the target day (notes 21.31): error and bias by hour, midnight-issue load forecast."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; CM = {"winter": "cmw", "summer": "cms", "autumn": "cma"}; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(2, 3, figsize=(17.5, 8.4), gridspec_kw=dict(hspace=0.4, wspace=0.2)); h = np.arange(24)
for i, per in enumerate(CM):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :96]; ld = lambda f, k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float)[:, :96]; S = (("merit order", ld(f"structfc_m1m_{per}_price", "vint_ts"), BLUE, "-"), ("merit order with block orders", ld(f"structfc_m1hm_{per}_price", "vint_ts"), INK, (0, (4, 2))), ("candidate's curve", ld(f"structfc_{CM[per]}_ts_dyn2xxh"), ORANGE, "-")); hr = lambda x: x.reshape(len(x), 24, 4).mean(2).mean(0)
    for row, fn, yl in ((0, lambda v: hr(np.abs(v - a)), "mean absolute error, ct/kWh"), (1, lambda v: hr(v - a), "forecast minus actual, ct/kWh")):
        ax = axs[row, i]
        for nm, v, col, ls in S: ax.plot(h + 0.5, fn(v), color=col, linestyle=ls, linewidth=1.9, label=nm)
        ax.axvspan(0, 6, color=NEUT, alpha=0.15, linewidth=0); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("hour of the target day (shaded: the key-metric window)"); ax.set_ylabel(yl); ax.grid(True, color=GRID, linewidth=0.6); [ax.spines[s].set_visible(False) for s in ("top", "right")]
        if row == 1: ax.axhline(0, color=INK, linewidth=0.6)
        if row == 0 and i == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
    n1, nh = np.abs(S[0][1] - a)[:, :24].mean(), np.abs(S[1][1] - a)[:, :24].mean(); d1, dh = np.abs(S[0][1] - a)[:, 24:].mean(), np.abs(S[1][1] - a)[:, 24:].mean()
    axs[0, i].set_title(f"{per.capitalize()}: block orders make the night worse ({n1:.2f} to {nh:.2f} ct)\nand the rest of the day {'better' if dh < d1 else 'no better'} ({d1:.2f} to {dh:.2f})", loc="left", fontsize=9.5, fontweight="bold", color=INK); axs[1, i].set_title(f"{per.capitalize()}: bias by hour", loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig.suptitle("Merit order with and without block orders over the target day (load forecast issued at midnight; with the layer)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.97); fig.savefig(f"{SP}/out/market1h_day.png", dpi=105, bbox_inches="tight"); print("figure ok")
