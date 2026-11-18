"""Memory in the shocks (notes 21.11-21.12): what each likelihood says the supply response is, whether whole days now vary as much as real ones,
a strip of simulated against real prices, and the price two days ahead from the model's own memory."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ["NOPHI"] = "1"; import market0m as Mm; B = Mm.B
SP = B.SP; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="y"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(1); med = lambda f: {k: float(np.median(v)) for k, v in np.load(f).items() if np.ndim(v) == 1}; T0, T1, T2 = med(f"{SP}/out/market0d_post.npz"), med(f"{SP}/out/market0m0_post.npz"), med(f"{SP}/out/market0m_post.npz"); A = Mm.grid(B.H[B.TR]); v = A["v"]; L = A["L"]; D = A["D"]
slope = lambda t: (t["K_T"] * 0.3989 / t["s_T"], t["x1"], 2 * t["K_st"] * 0.3989 / t["s_st"])
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 4, wspace=0.30)
ax = fig.add_subplot(gs[0, 0]); x = np.arange(3); w = 0.25; SL = [slope(t) for t in (T0, T1, T2)]
for j, (sl, col, lab) in enumerate(zip(SL, (NEUT, BLUE, ORANGE), ("hours independent", "memory from day to day", "memory from hour to hour"))):
    ax.bar(x + (j - 1) * (w + 0.02), sl, w, color=col, linewidth=0, label=lab)
    for xi, a in zip(x, sl): ax.text(xi + (j - 1) * (w + 0.02), a + 0.4, f"{a:.0f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["thermal", "imports", "storage"]); ax.set_ylabel("GW per unit of price / cost, at the block's centre"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax)
ttl(ax, f"Supply response under three likelihoods;\nthe hour-to-hour fit (thermal {SL[2][0]:.0f} GW) is rejected")
# spread of the day's mean price error: observed against simulated from each fitted model
def daysd(p, p0): r = np.where(v, (p - p0) / L, np.nan); return float(np.nanstd(np.nanmean(r, 1)))
res = []
for th, nm, simf in ((T0, "hours\nindependent", lambda t: B.simulate(t, D.reshape(-1), L.reshape(-1), rng)[0].reshape(D.shape)), (T1, "memory from\nday to day", lambda t: Mm.simulate(t, D, L, rng)[0]), (T2, "memory from\nhour to hour", lambda t: Mm.simulate(t, D, L, rng)[0])):
    p0 = B.clear(th, D.reshape(-1), L.reshape(-1))[0].reshape(D.shape); obs = daysd(A["p"], p0); sims = [daysd(simf(th), p0) for _ in range(20)]; res.append((nm, obs, np.percentile(sims, [5, 50, 95]))); print(nm.replace("\n", " "), "sd of the day's mean price error (price / cost): observed", round(obs, 3), "simulated", np.round(res[-1][2], 3))
    if "day to day" in nm: strip = [simf(th) for _ in range(2)]
ax = fig.add_subplot(gs[0, 1])
for i, (nm, obs, q) in enumerate(res):
    ax.plot([i, i], [q[0], q[2]], color=BLUE, linewidth=6, alpha=0.4, solid_capstyle="round", label="simulated from the fitted model (5-95%)" if i == 0 else None); ax.scatter([i], [q[1]], s=42, color=BLUE, zorder=3, edgecolor=SURF, linewidth=0.8); ax.scatter([i], [obs], s=70, color=INK, zorder=4, edgecolor=SURF, linewidth=0.8, marker="D", label="observed" if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in res]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, None); ax.set_ylabel("sd of the day's mean price error, price / cost"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ttl(ax, f"Whole days: simulated spread is {res[0][2][1] / res[0][1]:.0%} of real with\nindependent hours, {res[1][2][1] / res[1][1]:.0%} with day-to-day memory")
ax = fig.add_subplot(gs[0, 2]); d0, nd = 420, 14; strip = strip[:1]; t = np.arange(nd * 24) / 24.0
for s_ in strip: ax.plot(t, (s_[d0:d0 + nd] / L[d0:d0 + nd]).reshape(-1), color=BLUE, linewidth=1.0, alpha=0.8, label="simulated" if s_ is strip[0] else None)
ax.plot(t, np.where(v[d0:d0 + nd], A["p"][d0:d0 + nd] / L[d0:d0 + nd], np.nan).reshape(-1), color=INK, linewidth=1.0, label="real"); ax.set_xlabel(f"days from {pd.Timestamp(A['index'][d0 * 24]).strftime('%-d %B %Y')}"); ax.set_ylabel("price / gas-plus-CO2 cost"); ax.set_ylim(-1.0, 2.6); ax.legend(frameon=False, loc="lower left", fontsize=8, ncol=2, bbox_to_anchor=(0.0, -0.01)); clean(ax, "both")
ttl(ax, "Two weeks, real and simulated (day-to-day\nmemory): same range, rougher hours")
ax = fig.add_subplot(gs[0, 3]); grp = ("own\nmemory", "last 7 days carried\n(same data)", "plain curve,\nerror carried"); fitv = (1.47, 1.51, 1.47); hov = (1.96, 1.90, 1.93); w = 0.3
for xi, a, b in zip(x, fitv, hov):
    ax.bar(xi - w / 2 - 0.02, a, w, color=NEUT, linewidth=0); ax.bar(xi + w / 2 + 0.02, b, w, color=ORANGE, linewidth=0); ax.text(xi - w / 2 - 0.02, a + 0.03, f"{a:.2f}", ha="center", fontsize=8.5, color=INK); ax.text(xi + w / 2 + 0.02, b + 0.03, f"{b:.2f}", ha="center", fontsize=8.5, color=INK)
ax.bar([np.nan], [np.nan], color=NEUT, label="fitting period"); ax.bar([np.nan], [np.nan], color=ORANGE, label="held-out year"); ax.set_xticks(x); ax.set_xticklabels(grp, fontsize=8); ax.set_ylim(0, 2.5); ax.set_ylabel("hourly price error two days ahead, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ttl(ax, "Two days ahead, the model's own memory is\nno better than carrying the last week")
fig.suptitle("Memory in the market model's shocks: what it reveals about the supply response, and what it is worth (true load given)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market0m_memory.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0m_memory.png")
