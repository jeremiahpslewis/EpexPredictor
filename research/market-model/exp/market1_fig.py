"""Merit order by fuel (M1, notes 21.27): each fuel's output against price / its own cost, fitting period and held-out year; and the stack at two cost levels."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market1 as M
SP = M.SP; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="both"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}; R["wk"] = ((R.index.tz_convert("UTC") - pd.Timestamp("2024-01-01", tz="UTC")) // pd.Timedelta(days=7)).astype(int)
def binned(df, xcol, ycol, edges, nboot=200, minn=80):
    x = df[xcol].values; y = df[ycol].values; b = np.digitize(x, edges) - 1; nb = len(edges) - 1; uw, wi = np.unique(df.wk.values, return_inverse=True); idx = [np.where(wi == k)[0] for k in range(len(uw))]; med = np.array([np.median(y[b == i]) if (b == i).sum() >= minn else np.nan for i in range(nb)]); Bt = np.full((nboot, nb), np.nan)
    for j in range(nboot):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(uw), len(uw))]); bb, yy = b[ii], y[ii]
        for i in range(nb):
            m = bb == i
            if m.sum() >= minn // 2: Bt[j, i] = np.median(yy[m])
    lo, hi = np.nanpercentile(Bt, [5, 95], axis=0); lo[np.isnan(med)] = np.nan; hi[np.isnan(med)] = np.nan; return med, lo, hi
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 4, wspace=0.28); e = np.r_[-0.4, -0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.1, 2.8]; cen = 0.5 * (e[:-1] + e[1:]); rr = np.linspace(-0.4, 2.8, 200); NAME = dict(l="Lignite", c="Hard coal", g="Gas"); tr, ho = R[R.tr], R[~R.tr]
for i, f in enumerate(M.FU):
    ax = fig.add_subplot(gs[0, i])
    for df, col, lab in ((tr, NEUT, "fitting period"), (ho, ORANGE, "held-out year")):
        med, lo, hi = binned(df, f"rc_{f}", M.QC[f], e); ax.fill_between(cen, lo, hi, color=col, alpha=0.18, linewidth=0); ax.plot(cen, med, "o", color=col, markersize=4, label=lab)
    ax.plot(rr, M.CAP[f] * (th[f"phi_{f}"] + (1 - th[f"phi_{f}"]) * th[f"kap_{f}"] * M.norm.cdf((rr - 1) / th[f"s_{f}"])), color=BLUE, linewidth=2.2, label="block as fitted"); ax.set_xlabel(f"price / {NAME[f].lower()}'s own cost"); ax.set_ylabel("output, GW"); ax.set_ylim(0, None); clean(ax)
    if i == 0: ax.legend(frameon=False, loc="lower right", fontsize=8)
    d = float(ho[f"d_{f}"].mean()); cost_txt = {"l": f"cost = {th['a_l']:.0f} + CO2", "c": f"cost = {th['a_c']:.0f} + 0.85 CO2", "g": f"centred at {th['m_g']:.1f} x a modern plant's cost"}[f]
    fin = {"l": "on its block below cost, about 2 GW short above", "c": "on its block below cost, about 2 GW short above", "g": "above its block: it fills what the others leave"}[f]; ttl(ax, f"{NAME[f]} ({cost_txt}), held-out year:\n{fin}")
ax = fig.add_subplot(gs[0, 3]); pp = np.linspace(-20, 320, 300)
for (gas, co2, col, lab) in ((35.0, 70.0, NEUT, "gas 35, CO2 70 (fitting period)"), (90.0, 75.0, ORANGE, "gas 90, CO2 75 (autumn 2026)")):
    L = 1.85 * gas + 0.37 * co2; q = M.fuels(th, pp, np.full(len(pp), L), np.full(len(pp), co2)); ax.plot(q["l"], pp, color=col, linewidth=1.2, linestyle=(0, (1, 2))); ax.plot(q["l"] + q["c"], pp, color=col, linewidth=1.2, linestyle=(0, (4, 2))); ax.plot(q["l"] + q["c"] + q["g"], pp, color=col, linewidth=2.2, label=lab)
ax.set_xlabel("thermal output, GW (dotted: lignite; dashed: + coal; solid: + gas)"); ax.set_ylabel("price, EUR/MWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "The stack at two gas prices: only the gas part\nmoves; lignite and coal stay where they are")
fig.suptitle("Merit order by fuel (M1): each block against its own cost, and whether the held-out year, with gas far dearer, sits on it", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market1_fuels.png", dpi=105, bbox_inches="tight"); print("figure ok")
