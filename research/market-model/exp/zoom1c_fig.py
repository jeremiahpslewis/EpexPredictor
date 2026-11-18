"""Pictures for the CO2 weight (notes 20.7)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
D = pd.read_parquet(f"{SP}/out/zoom1c_rows.parquet"); D = D[D.tr | D.ho].copy(); Z = np.load(f"{SP}/out/zoom1c_post.npz"); W = np.load(f"{SP}/out/zoom1c_w.npz"); HO0 = pd.Timestamp("2025-12-01", tz="Europe/Berlin"); rng = np.random.default_rng(0); out = {}
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig = plt.figure(figsize=(17.0, 9.6)); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.26)
m = D.resample("MS").agg(gas=("gas", "mean"), co2=("co2", "mean")); ax = fig.add_subplot(gs[0, 0]); ax.plot(m.index, 1.85 * m.gas / 10, color=BLUE, linewidth=2, label="gas part: 1.85 x gas price"); ax.plot(m.index, 0.37 * m.co2 / 10, color=ORANGE, linewidth=2, label="CO2 part: 0.37 x allowance price"); ax.axvspan(HO0, m.index[-1], color="#f1f0ea", zorder=0)
ax.set_ylabel("ct per kWh of electricity"); ax.set_ylim(0, None); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); sh = 0.37 * m.co2 / (1.85 * m.gas + 0.37 * m.co2); out["CO2 share of gas-fired cost, min / max by month"] = (float(sh.min()), float(sh.max()))
ttl(ax, f"Gas-fired cost is gas plus CO2 allowances; CO2 is\n{100 * sh.min():.0f}-{100 * sh.max():.0f}% of it (shaded: the held-out year)")
ax = fig.add_subplot(gs[0, 1]); ax.plot(m.index, m.co2, color=ORANGE, linewidth=2); ax.axvspan(HO0, m.index[-1], color="#f1f0ea", zorder=0); ax.set_ylabel("CO2 allowance price, EUR per tonne"); ax.set_ylim(40, 95); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); clean(ax)
ttl(ax, "In the fitting period CO2 mostly rose with the\ncalendar, so its effect cannot be told from a trend")
ax = fig.add_subplot(gs[0, 2]); bins = np.linspace(0.2, 2.0, 91)
for w, col, lab in ((Z["w"], NEUT, f"fitting period (gas 28-55): {np.median(Z['w']):.2f}"), (W["w_all"], BLUE, f"whole span: {np.median(W['w_all']):.2f}"), (W["w_ho"], ORANGE, f"held-out year (gas 29-89): {np.median(W['w_ho']):.2f}")):
    ax.hist(w, bins=bins, color=col, alpha=0.75, label=lab, density=True, linewidth=0)
for xv, lab in ((0.37, "gas plant"), (0.85, "hard coal"), (1.0, "lignite")): ax.axvline(xv, color=INK, linewidth=0.9, linestyle=(0, (4, 3))); ax.annotate(lab, (xv, ax.get_ylim()[1] * 0.97), xytext=(3, 0), textcoords="offset points", fontsize=8, color=MUTED, rotation=90, va="top")
ax.set_xlabel("fitted CO2 weight, tonnes per MWh of electricity"); ax.set_yticks([]); ax.legend(frameon=False, loc="center right", fontsize=8); clean(ax); ax.grid(False)
ax.annotate("held-out year with the CO2 price\nheld at its mean: 0.33", (0.37, 0.30), xycoords=("data", "axes fraction"), xytext=(14, 0), textcoords="offset points", fontsize=8, color=MUTED, va="center")
ttl(ax, "The split between gas and the rest matches a gas plant's;\nCO2's own effect cannot be isolated (bands too narrow)")
# residual against carbon by load band, with a noise envelope from whole-week shifts
ax = fig.add_subplot(gs[1, 0]); edges = np.array([56, 62, 66, 70, 73, 76, 79, 82, 85, 90]); cen = (edges[:-1] + edges[1:]) / 2; r = D.r1.values; c = D.co2.values; n = len(D); nw = n // 168
for (nm, msk), col in zip((("below 20 GW", D.rl.values < 20), ("20-40 GW", (D.rl.values >= 20) & (D.rl.values < 40)), ("above 40 GW", D.rl.values >= 40)), (AQUA, BLUE, ORANGE)):
    def bm(res, mk):
        s_ = pd.Series(np.where(mk, res, np.nan)); g_ = s_.groupby(pd.cut(c, edges), observed=False); v = g_.median(); return np.where(g_.count() >= 150, v, np.nan)
    obs = bm(r, msk); env = np.array([bm(np.roll(r, int(k) * 168), np.roll(msk, int(k) * 168)) for k in rng.integers(6, nw - 6, 150)]); lo, hi = np.nanpercentile(env, [5, 95], axis=0)   # residual AND its load band shifted together against carbon
    out[f"bins outside the envelope, {nm}"] = (float(np.nansum((obs > hi) | (obs < lo))), float(np.sum(np.isfinite(obs))))
    ax.fill_between(cen, lo, hi, color=col, alpha=0.13, linewidth=0); ax.plot(cen, obs, "-o", color=col, linewidth=1.9, markersize=4, label=nm); out[f"residual range against carbon, {nm} (observed / envelope width)"] = (float(np.nanmax(obs) - np.nanmin(obs)), float(np.nanmean(hi - lo)))
ax.axhline(0, color=INK, linewidth=0.7); ax.set_ylim(-0.45, 0.45); ax.set_xlabel("CO2 allowance price, EUR per tonne"); ax.set_ylabel("level-1 residual, ratio units (median)"); ax.legend(frameon=False, loc="lower right", fontsize=8); clean(ax)
ttl(ax, "Level-1 residual against the CO2 price, by load band\n(shaded: what whole-week shifts give by chance)")
D["yr"] = D.index.year; D["mo"] = D.index.month; YC = {2024: "#c9c8c2", 2025: "#8f8e87", 2026: "#2b2a27"}
for c_, (col, lab) in enumerate((("r1", "CO2 weight 0.37 (as in levels 0-2)"), ("rw", f"CO2 weight {np.median(Z['w']):.2f} (fitted on the fitting period)"))):
    ax = fig.add_subplot(gs[1, 1 + c_]); t = D.groupby(["yr", "mo"])[col].median().unstack("yr"); n_ = D.groupby(["yr", "mo"]).size().unstack("yr")
    for yr in (2024, 2025, 2026): ax.plot(t.index, t[yr].where(n_[yr] >= 300), "-o", color=YC[yr], linewidth=2, markersize=4, label=str(yr))
    ax.axhline(0, color=INK, linewidth=0.7); ax.set_ylim(-0.45, 0.45); ax.set_xticks(range(1, 13)); ax.set_xlabel("month of the year"); ax.set_ylabel("residual, ratio units (median)"); ax.legend(frameon=False, loc="lower right", fontsize=8, title="year", title_fontsize=8); clean(ax)
    j = t.loc[[7, 8, 9]].mean(); out[f"Jul-Sep residual by year, {lab}"] = tuple(float(j[y_]) for y_ in (2024, 2025, 2026)); ttl(ax, ("Residual by month, one line per year: higher each year\n" if c_ == 0 else "A large CO2 weight closes 2024 to 2025 and\nopens 2026 wider: ") + lab if c_ == 0 else "A large CO2 weight closes the gap 2024 to 2025\nbut leaves 2026 further out")
fig.savefig(f"{SP}/out/zoom_co2.png", dpi=110, bbox_inches="tight"); print("figure:", f"{SP}/out/zoom_co2.png")
for k_, v in out.items(): print(f"  {k_}: " + ", ".join(f"{float(x):.3f}" for x in v))
