"""Market model M0c, one more picture (notes 21.7): where does the held-out gap sit?
Rest of system R = realised residual load - thermal - net imports is observed. Does its relation to price transfer
between the fitting period and the held-out year in EUR, or in price / cost? And are the two supply curves' shocks
centred in the held-out year? Bands: whole weeks resampled with replacement (hours within a week stay together)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); H = pd.read_parquet(f"{SP}/out/market0_rows.parquet"); H["R"] = H.rlreal - H["T"] - H.X; H["r"] = H.p / H.L; H["stor"] = H.psgen + H.pump; H["gap"] = H.R - H.other - H.stor
H["wk"] = (H.index.tz_convert("UTC").asi8 // (168 * 3600 * 10**6 if str(H.index.dtype).find("[us") > 0 else 168 * 3600 * 10**9)).astype(int)
tr, ho = H[H.tr], H[~H.tr]; print("hours", len(tr), len(ho), "| cost median", round(tr.L.median(), 1), round(ho.L.median(), 1), "| price median", round(tr.p.median(), 1), round(ho.p.median(), 1))
def binned(df, xcol, ycol, edges, nboot=300, minn=80):
    """medians of ycol in bins of xcol, with 5-95% band from resampling whole weeks"""
    x = df[xcol].values; y = df[ycol].values; w = df.wk.values; b = np.digitize(x, edges) - 1; nb = len(edges) - 1; uw, wi = np.unique(w, return_inverse=True)
    med = np.array([np.median(y[b == i]) if (b == i).sum() >= minn else np.nan for i in range(nb)]); idx_by_w = [np.where(wi == k)[0] for k in range(len(uw))]; B = np.full((nboot, nb), np.nan)
    for j in range(nboot):
        ii = np.concatenate([idx_by_w[k] for k in rng.integers(0, len(uw), len(uw))]); bb, yy = b[ii], y[ii]
        for i in range(nb):
            m = bb == i
            if m.sum() >= minn // 2: B[j, i] = np.median(yy[m])
    lo, hi = np.nanpercentile(B, [5, 95], axis=0); lo[np.isnan(med)] = np.nan; hi[np.isnan(med)] = np.nan; return med, lo, hi
def draw(ax, edges, cols=("R",), xcol="p", sets=None, lab=True):
    cen = 0.5 * (edges[:-1] + edges[1:]); res = {}
    for df, col, nm in sets:
        med, lo, hi = binned(df, xcol, cols[0], edges); ax.fill_between(cen, lo, hi, color=col, alpha=0.18, linewidth=0); ax.plot(cen, med, "-o", color=col, linewidth=1.8, markersize=3.5, label=nm if lab else None); res[nm] = (med, lo, hi)
    clean(ax); return cen, res
sets = ((tr, NEUT, "fitting period (Mar 2024 - Nov 2025)"), (ho, ORANGE, "held-out year (Dec 2025 - Nov 2026)")); eP = np.r_[-60, -20, -1, 1, 20, 40, 60, 80, 100, 120, 140, 170, 200, 260]; eR = np.r_[-0.6, -0.2, -0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.6]
fig = plt.figure(figsize=(18.5, 13.4)); gs = fig.add_gridspec(3, 4, hspace=0.46, wspace=0.32); out = {}
def gapstat(res):
    (a, alo, ahi), (b, blo, bhi) = res[sets[0][2]], res[sets[1][2]]; d = b - a; ok = ~np.isnan(d); return float(np.nanmean(np.abs(d))), float(np.nanmean(d)), int(((blo > ahi) | (bhi < alo))[ok].sum()), int(ok.sum())
# row 0: rest of system and its parts against EUR price ; row 1: against price / cost
for row, (xcol, edges, xl, xlim) in enumerate((("p", eP, "price, EUR/MWh", (-60, 260)), ("r", eR, "price / gas-plus-CO2 cost", (-0.6, 2.6)))):
    for c_, (col, nm, yl) in enumerate((("R", "Rest of system", (4, 16)), ("other", "Other plant (biomass, hydro, nuclear, ...)", (6, 10)), ("stor", "Pumped storage, generation minus pumping", (-5, 5)), ("gap", "Balance gap (nothing measured)", (0, 8)))):
        ax = fig.add_subplot(gs[row, c_]); cen, res = draw(ax, edges, (col,), xcol, sets, lab=(row == 0 and c_ == 0)); g = gapstat(res); out[(col, xcol)] = g; ax.set_xlim(*xlim); ax.set_ylim(*yl); ax.set_xlabel(xl); ax.set_ylabel("GW")
        ttl(ax, f"{nm}\nheld-out minus fitting: {g[1]:+.2f} GW on average, bands apart in {g[2]} of {g[3]} bins")
        if row == 0 and c_ == 0: ax.legend(frameon=False, loc="lower right", fontsize=8)
# row 2: the three shocks in the held-out year by residual-load band, and the supply quantities against p/L by period
eL = np.arange(-10, 71, 10)
for c_, (col, nm, yl) in enumerate((("eT", "Thermal shock (observed - curve)", (-8, 8)), ("eX", "Import shock (observed - curve)", (-8, 8)), ("d", "Third shock (rest of system - its curve)", (-8, 8)))):
    ax = fig.add_subplot(gs[2, c_]); cen, res = draw(ax, eL, (col,), "rlreal", sets, lab=False); ax.axhline(0, color=INK, linewidth=0.7); ax.set_xlim(-10, 70); ax.set_ylim(*yl); ax.set_xlabel("residual load, GW (10 GW bands)"); ax.set_ylabel("GW"); mh = float(ho[col].mean()); mt = float(tr[col].mean()); out[(col, "mean")] = (mt, mh)
    ttl(ax, f"{nm}\nmean {mt:+.2f} GW fitting, {mh:+.2f} GW held-out")
ax = fig.add_subplot(gs[2, 3]); yrs = (("2024", NEUT), ("2025", BLUE), ("2026", ORANGE)); cen = 0.5 * (eR[:-1] + eR[1:])
for y_, col in yrs:
    df = H[H.index.year == int(y_)]; med, lo, hi = binned(df, "r", "T", eR); ax.fill_between(cen, lo, hi, color=col, alpha=0.15, linewidth=0); ax.plot(cen, med, "-o", color=col, linewidth=1.8, markersize=3.5, label=y_)
ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 40); ax.set_xlabel("price / gas-plus-CO2 cost"); ax.set_ylabel("thermal output, GW"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "Thermal output against price / cost,\nby calendar year")
fig.suptitle("Market model M0c: where the held-out gap sits. Medians per bin; bands from resampling whole weeks (5-95%)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.93)
fig.savefig(f"{SP}/out/market0_rest.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_rest.png")
for k, v in out.items(): print(k, tuple(round(float(x), 2) for x in v))
# numbers behind the pictures: yearly means of each part at the same price / cost band
for lo_, hi_ in ((0.6, 1.0), (1.0, 1.4), (1.4, 2.0)):
    m = (H.r >= lo_) & (H.r < hi_); print(f"price/cost {lo_}-{hi_}:", {str(y_): {c: round(float(H[m & (H.index.year == y_)][c].median()), 2) for c in ("T", "X", "R", "other", "stor", "gap", "rlreal", "p", "L")} for y_ in (2024, 2025, 2026)})
