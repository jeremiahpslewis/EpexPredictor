"""Figure for the gas-share prior on the candidate's curve: (A) one fit, the price curve at the gas price it was fitted on and at a much higher one, with and
without the prior; (B) the gas share at 30 GW across all origins; (C) the frozen test in autumn, curve alone and with the online layer."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(13.2, 8.4)); gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.5)
# A
ax = fig.add_subplot(gs[0, :2]); T = "2026-09-18"; zh = np.load(f"{SP}/out/struct_fh4/{T}.npz"); zg = np.load(f"{SP}/out/struct_fg4/{T}.npz"); x = np.linspace(0, 62, 200)
Ia = np.load(f"{SP}/out/inputs_autumn.npz"); g0 = float(Ia["gas"][[str(c)[:10] for c in Ia["cutoffs"]].index(T)])
for z, col, nm in ((zh, BLUE, "as the candidate"), (zg, ORANGE, "with the prior")):
    b = np.interp(x, z["grid"], z["b_curve"]); h = np.interp(x, z["grid"], z["h_curve"])
    ax.plot(x, (b + g0 * h) / 10, color=col, linewidth=2, label=f"{nm}, at that day's gas price ({g0:.0f})"); ax.plot(x, (b + (g0 - 40) * h) / 10, color=col, linewidth=2, linestyle=(0, (5, 2)), label=f"{nm}, if gas were 40 lower")
ax.set_xlabel("effective residual load, GW"); ax.set_ylabel("price, ct/kWh"); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left", fontsize=8); ax.set_ylim(0, None)
ax.set_title("One fit, 18 Sep 2026: same today, different if gas moves", loc="left", fontsize=10, fontweight="bold", color=INK)
# B
bx = fig.add_subplot(gs[0, 2:]); eua = np.load(f"{SP}/out/eua_15min.npz")["eua"]
for runs, col, nm in ((("hw2", "hs2", "ha"), BLUE, "as the candidate"), (("kw", "ks", "ka"), ORANGE, "with the prior")):
    first = True
    for period, run in zip(("winter", "summer", "autumn"), runs):
        I = np.load(f"{SP}/out/inputs_{period}.npz"); t_, v_ = [], []
        for c in I["cutoffs"]:
            f = f"{SP}/out/struct_{run}/{str(c)[:10]}.npz"
            if not os.path.exists(f): continue
            z = np.load(f); g = float(z["gref"]); b = float(np.interp(30.0, z["grid"], z["b_curve"])); h = float(np.interp(30.0, z["grid"], z["h_curve"])); t_.append(pd.Timestamp(str(c)[:10])); v_.append(g * h / (b + g * h))
        bx.plot(t_, v_, color=col, linewidth=2, label=nm if first else None); first = False
bx.axhline(0.665, color=NEUT, linewidth=1, linestyle=(0, (4, 3))); bx.annotate("target from 2021-2023: about 0.66", (pd.Timestamp("2026-11-08"), 0.665), xytext=(0, 9), textcoords="offset points", fontsize=8, color=MUTED, ha="right")
bx.set_ylabel("share at the window's mean gas price"); bx.set_ylim(0.2, 0.9); bx.grid(True, axis="y", color=GRID, linewidth=0.6); bx.set_axisbelow(True); bx.legend(frameon=False, loc="lower right", fontsize=8); bx.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); bx.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
bx.set_title("Gas-priced share of the price at 30 GW, fit by fit", loc="left", fontsize=10, fontweight="bold", color=INK)
# C
I = np.load(f"{SP}/out/inputs_autumn.npz"); a = I["actual"].astype(float); night = lambda f, k: float(np.abs(np.load(f"{SP}/out/{f}.npz")[k].astype(float) - a)[:, :32].mean())
for c, (key, ttl) in enumerate((("curve", "Autumn, frozen curve alone"), ("vint", "Autumn, frozen curve with its online layer"))):
    cx = fig.add_subplot(gs[1, 2 * c:2 * c + 2]); w = 0.34; ref = night("structfc_ha_ts_dyn2xxh", key)
    for i, (d, lab) in enumerate((("j", "fit of 6 Jan"), ("m", "fit of 31 Mar"))):
        for k_, (arm, col, nm) in enumerate((("h", BLUE, "as the candidate"), ("g", ORANGE, "with the prior"))):
            v = night(f"structfc_z{arm}4{d}_a_ts_dyn2xxh", key); cx.bar(i + (k_ - 0.5) * (w + 0.03), v, width=w, color=col, linewidth=0, label=nm if i == 0 else None); cx.annotate(f"{v:.2f}", (i + (k_ - 0.5) * (w + 0.03), v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    cx.axhline(ref, color=NEUT, linewidth=1.2, linestyle=(0, (4, 3)), label=f"refitted at every origin: {ref:.2f}")
    cx.set_xticks([0, 1]); cx.set_xticklabels(["fit of 6 Jan", "fit of 31 Mar"]); cx.set_xlim(-0.6, 1.6); cx.set_ylabel("night MAE, ct/kWh"); cx.set_ylim(0, 4.6); cx.grid(True, axis="y", color=GRID, linewidth=0.6); cx.set_axisbelow(True); cx.set_title(ttl, loc="left", fontsize=10, fontweight="bold", color=INK)
    cx.legend(frameon=False, loc="upper left", fontsize=8)
for a_ in fig.axes:
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
out = f"{SP}/out/gs_fig.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); print("figure:", out)
