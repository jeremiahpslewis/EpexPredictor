"""The candidate's layer on the market model, in price and in GW, and the surplus-end level (notes 21.15-21.17)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"; LB = "#a9c9ef"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="y"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 4, wspace=0.30)
# (a) the low end of the curve moved sideways
ax = fig.add_subplot(gs[0, 0]); R = pd.read_parquet(f"{SP}/out/market0e_rows.parquet"); e = np.arange(-14, 22, 3.0); cen = e[:-1] + 1.5
for g0, col, lab in ((R[R.tr], NEUT, "fitting period"), (R[~R.tr], ORANGE, "held-out year")):
    md = [g0[(g0.D >= a) & (g0.D < a + 3)].p.median() if ((g0.D >= a) & (g0.D < a + 3)).sum() > 40 else np.nan for a in e[:-1]]; mm = [g0[(g0.D >= a) & (g0.D < a + 3)].p0.median() if ((g0.D >= a) & (g0.D < a + 3)).sum() > 40 else np.nan for a in e[:-1]]
    ax.plot(cen, md, "-o", color=col, linewidth=1.8, markersize=4, label=f"observed, {lab}"); ax.plot(cen, mm, color=col, linewidth=1.4, linestyle=(0, (4, 2)), label=f"model as fitted, {lab}")
ax.axhline(0, color=INK, linewidth=0.6); ax.set_xlabel("demand left for thermal, imports and storage, GW"); ax.set_ylabel("price, EUR/MWh (median)"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax, "both")
ttl(ax, "Held-out year: the low end sits 5-8 GW to the\nleft; zero prices come at much lower demand")
# (b) ladder, summer first 14 h, true load
ax = fig.add_subplot(gs[0, 1]); lab = ("curves\nonly", "+ layer\nin price", "+ layer\nin GW", "+ surplus-end\nlevel"); v = (3.07, 1.67, 1.52, 1.25); x = np.arange(4)
ax.bar(x, v, 0.55, color=[LB, BLUE, BLUE, BLUE], linewidth=0)
for xi, a in zip(x, v): ax.text(xi, a + 0.05, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
ax.axhline(1.14, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 2))); ax.text(1.5, 0.98, "candidate's curve + layer: 1.14", ha="center", fontsize=8, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5)); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0, 3.5); ax.set_ylabel("first 14 h of D+2, summer, price error, ct/kWh"); clean(ax)
ttl(ax, "Summer, true load: each step toward the\ncandidate's curve (model frozen at Nov 2025)")
# (c) by residual load, summer + autumn, true load
ax = fig.add_subplot(gs[0, 2]); bands = ((-40, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 80)); E = {k: [[] for _ in bands] for k in ("c", "p", "g")}
for per, run in (("summer", "hs2"), ("autumn", "ha")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :56]; rl = I["rl_true"].astype(float)[:, :56]
    for k, f, key in (("c", f"structfc_{run}_known_dyn2xxh", "vint"), ("p", f"structfc_mm0_{per}_known", "vint"), ("g", f"structfc_mg0e_l1_{per}_price", "vint")):
        er = np.abs(np.load(f"{SP}/out/{f}.npz")[key].astype(float)[:, :56] - a)
        for i, (lo, hi) in enumerate(bands): E[k][i].append(er[(rl >= lo) & (rl < hi)])
x = np.arange(len(bands)); w = 0.26
for j, (k, col, lb) in enumerate((("p", LB, "market model, layer in price"), ("g", BLUE, "market model, surplus end + layer in GW"), ("c", ORANGE, "candidate's curve + layer"))):
    ax.bar(x + (j - 1) * (w + 0.02), [np.concatenate(E[k][i]).mean() for i in range(len(bands))], w, color=col, linewidth=0, label=lb)
ax.set_xticks(x); ax.set_xticklabels(["< 5", "5-15", "15-25", "25-35", "35-45", "> 45"]); ax.set_xlabel("residual load of the target slot, GW"); ax.set_ylabel("price error, ct/kWh (summer and autumn, first 14 h)"); ax.legend(frameon=False, loc="upper right", fontsize=7.5); clean(ax)
lo_p, lo_g, lo_c = (np.concatenate(E[k][0]).mean() for k in ("p", "g", "c")); ttl(ax, f"Below 5 GW the error falls from {lo_p:.1f} to {lo_g:.1f} ct\n(candidate {lo_c:.1f}); the rest is in the middle")
# (d) forecast mode, night slots, demand built from the published residual load (the series Timestral forecasts)
ax = fig.add_subplot(gs[0, 3]); per = ("winter", "summer", "autumn"); ser = (("market model + layer", BLUE, (0.81, 1.06, 1.73)), ("plain curve + layer", NEUT, (0.84, 1.05, 1.70)), ("candidate's curve + layer", ORANGE, (0.86, 0.89, 1.59)), ("market model + Timestral, 50/50", LB, (0.66, 0.83, 1.36)), ("candidate (its curve + Timestral)", AQUA, (0.67, 0.79, 1.31))); x = np.arange(3); w = 0.155
for j, (lb, col, v) in enumerate(ser):
    ax.bar(x + (j - 2) * (w + 0.012), v, w, color=col, linewidth=0, label=lb)
    for xi, a_ in zip(x, v): ax.text(xi + (j - 2) * (w + 0.012), a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=6.8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylim(0, 2.5); ax.set_ylabel("night slots of D+2, price error, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax)
ttl(ax, "Timestral's load forecast: level with a plain curve;\nin the blend, level with the candidate")
fig.suptitle("The market model with the candidate's online layer: learning the error as a supply shift in GW, and a step in supply near a price of zero", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market0e_layer.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0e_layer.png", "| below 5 GW:", round(lo_p, 2), round(lo_g, 2), round(lo_c, 2))
