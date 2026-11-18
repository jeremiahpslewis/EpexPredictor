"""Market model M0c (notes 21.7): which quantities moved from year to year, and how much of the unexplained rise in price relative to cost they account for.
Months March to October only, because all three years have them. Bands: whole weeks resampled with replacement, 5-95%."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market0 as M
SP = M.SP; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); H = pd.read_parquet(f"{SP}/out/market0_rows.parquet"); S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["g_fr_reservoir", "g_fr_ror", "p_fr", "p_ch", "p_at", "p_be", "p_nl", "p_dk", "p_pl", "p_cz"]].resample("1h").mean(); S.index = S.index.tz_convert("Europe/Berlin"); H = H.join(S, how="left")
H["stor"] = H.psgen + H.pump; H["gap"] = H.rlreal - H["T"] - H.X - H.other - H.stor; H["Tg"] = H["T"] + H.gap; H["r"] = H.p / H.L; H["frh"] = H.g_fr_reservoir + H.g_fr_ror; H["pn"] = H[["p_fr", "p_ch", "p_at", "p_be", "p_nl", "p_dk", "p_pl", "p_cz"]].mean(1) / H.L; H["yr"] = H.index.year; H["mo"] = H.index.month
H["wk"] = ((H.index.tz_convert("UTC") - pd.Timestamp("2024-01-01", tz="UTC")) // pd.Timedelta(days=7)).astype(int); C = H[(H.mo >= 3) & (H.mo <= 10)]; th = {k: float(np.median(np.load(f"{SP}/out/market0_post.npz")[k])) for k in M.PN}; YR = ((2024, NEUT), (2025, BLUE), (2026, ORANGE)); out = {}
def binned(df, xcol, ycol, edges, nboot=300, minn=80, fn=np.median):
    x = df[xcol].values; y = df[ycol].values; b = np.digitize(x, edges) - 1; nb = len(edges) - 1; uw, wi = np.unique(df.wk.values, return_inverse=True); idx = [np.where(wi == k)[0] for k in range(len(uw))]
    med = np.array([fn(y[b == i]) if (b == i).sum() >= minn else np.nan for i in range(nb)]); B = np.full((nboot, nb), np.nan)
    for j in range(nboot):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(uw), len(uw))]); bb, yy = b[ii], y[ii]
        for i in range(nb):
            m = bb == i
            if m.sum() >= minn // 2: B[j, i] = fn(yy[m])
    lo, hi = np.nanpercentile(B, [5, 95], axis=0); lo[np.isnan(med)] = np.nan; hi[np.isnan(med)] = np.nan; return med, lo, hi
fig = plt.figure(figsize=(18.5, 13.8)); gs = fig.add_gridspec(3, 4, hspace=0.48, wspace=0.30); eR = np.r_[-0.6, -0.2, -0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.1]; cen = 0.5 * (eR[:-1] + eR[1:])
# ---- row 0: quantities at the same price / cost, by year (no model involved)
for c_, (col, yl, ylab) in enumerate((("X", (-8, 12), "net imports, GW"), ("Tg", (5, 35), "thermal output + balance gap, GW"), ("other", (6, 10), "other plant, GW"), ("stor", (-5, 5), "pumped storage, GW"))):
    ax = fig.add_subplot(gs[0, c_]); R_ = {}
    for y_, colr in YR:
        med, lo, hi = binned(C[C.yr == y_], "r", col, eR); ax.fill_between(cen, lo, hi, color=colr, alpha=0.16, linewidth=0); ax.plot(cen, med, "-o", color=colr, linewidth=1.8, markersize=3.5, label=str(y_)); R_[y_] = med
    d1 = float(np.nanmean(R_[2025] - R_[2024])); d2 = float(np.nanmean(R_[2026] - R_[2025])); out[col] = (d1, d2); ax.set_xlim(-0.6, 2.1); ax.set_ylim(*yl); ax.set_xlabel("price / gas-plus-CO2 cost"); ax.set_ylabel(ylab); clean(ax)
    if c_ == 0: ax.legend(frameon=False, loc="upper left", fontsize=8, title="March to October of", title_fontsize=8)
    ttl(ax, {"X": f"Net imports at the same price / cost fell\nby {-d1:.1f} GW, then by another {-d2:.1f} GW", "Tg": f"Thermal output plus the unmeasured part:\nchange {d1:+.1f} GW, then {d2:+.1f} GW", "other": f"Other plant (biomass, hydro, waste):\nchange {d1:+.1f} GW, then {d2:+.1f} GW", "stor": f"Pumped storage: the same curve each year\n(change {d1:+.1f}, {d2:+.1f} GW)"}[col])
# ---- row 1: by month of the year, one line per year
mon = np.arange(1, 13); ml = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
def bymonth(ax, col, df=H):
    R_ = {}
    for y_, colr in YR:
        g = df[df.yr == y_]; v = np.array([g[g.mo == m][col].mean() if (g.mo == m).sum() > 300 else np.nan for m in mon]); ax.plot(mon, v, "-o", color=colr, linewidth=1.8, markersize=3.5, label=str(y_)); R_[y_] = v
    ax.set_xticks(mon); ax.set_xticklabels(ml); clean(ax); return R_
ax = fig.add_subplot(gs[1, 0]); R_ = bymonth(ax, "eX"); ax.axhline(0, color=INK, linewidth=0.7); ax.set_ylabel("net imports minus the import curve, GW"); a_, b_ = np.nanmean(R_[2025] - R_[2024]), np.nanmean(R_[2026] - R_[2025]); out["eX by month"] = (a_, b_, int(np.nansum((R_[2025] - R_[2024]) < 0)), int(np.sum(~np.isnan(R_[2025] - R_[2024]))), int(np.nansum((R_[2026] - R_[2025]) < 0)), int(np.sum(~np.isnan(R_[2026] - R_[2025])))); ax.set_ylim(-9, 6.5); ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=3)
ttl(ax, f"Imports against the import curve: lower in\n{out['eX by month'][2]} of {out['eX by month'][3]} months in 2025, {out['eX by month'][4]} of {out['eX by month'][5]} in 2026")
ax = fig.add_subplot(gs[1, 1]); R_ = bymonth(ax, "gap"); ax.set_ylabel("balance gap, GW"); ax.set_ylim(0, 6); g_ = R_[2026] - R_[2025]; out["gap Apr-Oct 2026 vs 2025"] = float(np.nanmean(g_[3:10])); ttl(ax, f"The unmeasured part of the balance\nfell by {-out['gap Apr-Oct 2026 vs 2025']:.1f} GW from April 2026 ...")
ax = fig.add_subplot(gs[1, 2]); R_ = bymonth(ax, "eT"); ax.axhline(0, color=INK, linewidth=0.7); ax.set_ylabel("thermal output minus the thermal curve, GW"); g_ = R_[2026] - R_[2025]; out["eT Apr-Oct 2026 vs 2025"] = float(np.nanmean(g_[3:10])); ttl(ax, f"... while measured thermal output rose\nby {out['eT Apr-Oct 2026 vs 2025']:.1f} GW against its curve")
ax = fig.add_subplot(gs[1, 3]); R_ = bymonth(ax, "frh"); ax.set_ylabel("French hydro output, GW"); ax.set_ylim(0, 12); out["French hydro Mar-Oct"] = tuple(float(np.nanmean(R_[y_][2:10])) for y_ in (2024, 2025, 2026)); fh = out["French hydro Mar-Oct"]; ttl(ax, f"French hydro fell {fh[0] - fh[1]:.1f} GW, then {fh[1] - fh[2]:.1f} GW:\nthe size of the first import drop only")
# ---- row 2: what the import shift does to the model's price, year by year
eL = np.arange(-10, 61, 10); cL = eL[:-1] + 5; fitm = H[H.tr]; oth0 = float(fitm.other.mean()); acc = {}
for c_, (y_, colr) in enumerate(YR):
    g = C[C.yr == y_].copy(); mX = float(g.eX.mean()); mO = float(g.other.mean() - oth0); g["r0"] = g.p0 / g.L; g["rX"] = M.clear(th, g.rlreal.values, g.L.values, eX=mX)[0] / g.L.values; g["rXO"] = M.clear(th, g.rlreal.values, g.L.values, eX=mX, d=-mO)[0] / g.L.values
    ax = fig.add_subplot(gs[2, c_]); med, lo, hi = binned(g, "rlreal", "r", eL); ax.fill_between(cL, lo, hi, color=INK, alpha=0.10, linewidth=0); ax.plot(cL, med, "o", color=INK, markersize=4.5, label="observed")
    for col, cc, ls, lab in (("r0", NEUT, "-", "model, no shocks"), ("rX", BLUE, "-", f"imports shifted {mX:+.1f} GW"), ("rXO", AQUA, (0, (4, 2)), f"and other plant {mO:+.1f} GW")):
        mm = np.array([np.median(g[col].values[(g.rlreal.values >= a) & (g.rlreal.values < a + 10)]) if ((g.rlreal.values >= a) & (g.rlreal.values < a + 10)).sum() >= 80 else np.nan for a in eL[:-1]]); ax.plot(cL, mm, color=cc, linewidth=2, linestyle=ls, label=lab)
    acc[y_] = tuple(float(np.median(g.r - g[k])) for k in ("r0", "rX", "rXO")) + (mX, mO, float(np.median(g.L)))
    ax.set_xlim(-10, 60); ax.set_ylim(-0.3, 2.0); ax.set_xlabel("residual load, GW (10 GW bands)"); ax.set_ylabel("price / gas-plus-CO2 cost"); ax.legend(frameon=False, loc="lower right", fontsize=7.5); clean(ax)
    ttl(ax, f"Mar-Oct {y_}: observed minus model is\n{acc[y_][0]:+.2f} bare, {acc[y_][1]:+.2f} with imports shifted")
ax = fig.add_subplot(gs[2, 3]); xs = np.arange(3)
for j, (lab, cc, mk) in enumerate((("model, no shocks", NEUT, "o"), ("imports shifted", BLUE, "o"), ("imports and other plant shifted", AQUA, "D"))):
    ax.plot(xs, [acc[y_][j] for y_, _ in YR], "-", color=cc, linewidth=1.6, marker=mk, markersize=7, markeredgecolor=SURF, markeredgewidth=0.8, label=lab)
ax.axhline(0, color=INK, linewidth=0.7); ax.set_xticks(xs); ax.set_xticklabels([str(y_) for y_, _ in YR]); ax.set_xlim(-0.4, 2.4); ax.set_ylabel("observed minus model, price / cost"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
rise0 = acc[2026][0] - acc[2024][0]; riseX = acc[2026][1] - acc[2024][1]; riseXO = acc[2026][2] - acc[2024][2]; out["rise 2024->2026 in observed minus model (none, imports, imports+other)"] = (rise0, riseX, riseXO)
ttl(ax, f"Rise against the model in two years: {rise0:.2f};\n{riseX:.2f} left with imports, {riseXO:.2f} with other plant")
fig.suptitle("Which quantities moved from year to year, and what that does to the price level (market model M0c; March to October of each year)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.93)
fig.savefig(f"{SP}/out/market0_level.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_level.png")
for k, v in out.items(): print(k, tuple(round(float(x), 2) for x in np.atleast_1d(v)))
for y_ in acc: print(y_, "observed minus model (none / imports / imports+other), mean eX, other shift, median cost:", tuple(round(x, 3) for x in acc[y_]))
print("neighbours' mean price / German cost, Mar-Oct:", {y_: round(float(C[C.yr == y_].pn.mean()), 3) for y_, _ in YR})
