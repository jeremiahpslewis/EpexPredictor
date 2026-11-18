"""Market model M0c (notes 21.9): how much of what the long-window model gets wrong is known 1 to 7 days ahead?
The model's curves are fitted once on Mar 2024 - Nov 2025. Here its shocks (GW, observed a day late through the quantities) are carried forward from a
14:00 origin on day D to days D+1 .. D+7, by hour of day, and the market is cleared again. Compared with carrying the price error forward instead.
NOT a forecast test: realised residual load, gas and CO2 of the target hour are used. It isolates the state."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market0 as M
SP = M.SP; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
R = pd.read_parquet(f"{SP}/out/market0_rows.parquet"); th = {k: float(np.median(np.load(f"{SP}/out/market0_post.npz")[k])) for k in M.PN}; R["po"] = (R.p - R.p0) / R.L
full = pd.date_range(R.index.min().tz_convert("UTC").floor("D"), R.index.max().tz_convert("UTC").ceil("D"), freq="1h", tz="UTC"); G = R.copy(); G.index = G.index.tz_convert("UTC"); G = G.reindex(full); loc = G.index.tz_convert("Europe/Berlin"); hod = np.asarray(loc.hour); n = len(G)
COLS = ("eT", "eX", "d", "po"); V = {c: G[c].values for c in COLS}
def state(c, h, N):
    """value carried to target hour t from origin day(t) - h at 14:00: same hour of day, mean of the last N days available at the origin (data to 12:00)"""
    v = V[c]; lag0 = np.where(hod <= 12, 24 * h, 24 * (h + 1)); acc = np.zeros(n); cnt = np.zeros(n); idx = np.arange(n)
    for j in range(N):
        src = idx - lag0 - 24 * j; okk = src >= 0; x = np.where(okk, v[np.clip(src, 0, n - 1)], np.nan); m = ~np.isnan(x); acc[m] += x[m]; cnt[m] += 1
    return np.where(cnt >= max(1, N // 2), acc / np.maximum(cnt, 1), np.nan)
tr = (G.tr == True).values; ho = (G.tr == False).values; night = hod < 8; res = {}
for N in (1, 7):
    for h in range(1, 8):
        sT, sX, sd_, so = (state(c, h, N) for c in COLS); m = ~np.isnan(sT + sX + sd_ + so + G.p.values); pq = np.full(n, np.nan); pq[m] = M.clear(th, G.rlreal.values[m], G.L.values[m], d=sd_[m], eT=sT[m], eX=sX[m])[0]; pp = G.p0.values + so * G.L.values
        for nm, msk in (("fit", tr), ("held", ho)):
            for sub, ms in (("all", np.ones(n, bool)), ("night", night)):
                k = m & msk & ms; res[(N, h, nm, sub)] = tuple(float(np.mean(np.abs(x[k] - G.p.values[k]))) / 10 for x in (G.p0.values, pp, pq)) + (int(k.sum()),)
        if N == 7 and h == 2: keep = (m.copy(), pq.copy(), pp.copy())
for k_, v in res.items():
    if k_[1] in (1, 2, 4, 7): print(k_, tuple(round(x, 2) for x in v[:3]), v[3])
# persistence of the daily mean of each shock
Dm = G[list(COLS)].copy(); Dm.index = loc; Dd = Dm.resample("1D").mean(); ac = {c: [float(Dd[c].corr(Dd[c].shift(k))) for k in range(1, 15)] for c in COLS}; share = {c: float(Dd[c].var() / G[c].var()) for c in COLS}
print("autocorrelation of daily means, lags 1, 2, 4, 7, 14:", {c: [round(ac[c][i], 2) for i in (0, 1, 3, 6, 13)] for c in COLS}); print("share of hourly variance that is the day's mean:", {c: round(v, 2) for c, v in share.items()})
fig = plt.figure(figsize=(18.5, 4.6)); gs = fig.add_gridspec(1, 4, wspace=0.30); hs = np.arange(1, 8)
ax = fig.add_subplot(gs[0, 0]); lg = np.arange(1, 15)
for c, col, lab in (("eX", BLUE, "import shock"), ("eT", ORANGE, "thermal shock"), ("d", AQUA, "third shock (rest of system)"), ("po", INK, "price error of the bare model")):
    ax.plot(lg, ac[c], "-o", color=col, linewidth=1.8, markersize=3.5, linestyle="-" if c != "po" else (0, (4, 2)), label=lab)
ax.set_ylim(0, 1); ax.set_xlabel("days apart"); ax.set_ylabel("correlation of daily means"); ax.legend(frameon=False, loc="lower left", fontsize=8); clean(ax); ttl(ax, f"The shocks are slow: daily means a week\napart correlate {ac['eX'][6]:.2f} (imports), {ac['eT'][6]:.2f} (thermal)")
for c_, (nm, lab) in enumerate((("fit", "fitting period"), ("held", "held-out year"))):
    ax = fig.add_subplot(gs[0, c_ + 1]); a0 = [res[(1, h, nm, "all")][0] for h in hs]
    ax.plot(hs, a0, "-", color=NEUT, linewidth=2, label="curves only (no state)")
    for N, ls in ((1, "-"), (7, (0, (4, 2)))):
        ax.plot(hs, [res[(N, h, nm, "all")][1] for h in hs], "-o", color=ORANGE, linewidth=1.8, markersize=3.5, linestyle=ls, label=f"carry the price error, last {N} day{'s' if N > 1 else ''}")
        ax.plot(hs, [res[(N, h, nm, "all")][2] for h in hs], "-o", color=BLUE, linewidth=1.8, markersize=3.5, linestyle=ls, label=f"carry the three shocks in GW, last {N} day{'s' if N > 1 else ''}")
    ax.set_ylim(0, max(a0 + [res[(1, h, nm, "all")][1] for h in hs]) * 1.12); ax.set_xlabel("days ahead of the 14:00 origin"); ax.set_ylabel("mean absolute price error, ct/kWh"); clean(ax); b1, b7 = res[(7, 1, nm, "all")][2], res[(7, 7, nm, "all")][2]
    if c_ == 0: ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ttl(ax, f"{lab.capitalize()}: {a0[0]:.2f} ct without state,\n{b1:.2f} a day ahead and {b7:.2f} a week ahead with it")
ax = fig.add_subplot(gs[0, 3]); m, pq, pp = keep; k = m & ho; rl = G.rlreal.values; edges = np.arange(-10, 71, 10); cen = edges[:-1] + 5
for x, col, lab in ((G.p0.values, NEUT, "curves only"), (pp, ORANGE, "carry the price error (7 days)"), (pq, BLUE, "carry the shocks in GW (7 days)")):
    ax.plot(cen, [np.mean(np.abs(x[k & (rl >= a) & (rl < a + 10)] - G.p.values[k & (rl >= a) & (rl < a + 10)])) / 10 if (k & (rl >= a) & (rl < a + 10)).sum() > 80 else np.nan for a in edges[:-1]], "-o", color=col, linewidth=1.8, markersize=3.5, label=lab)
ax.set_ylim(0, None); ax.set_xlabel("residual load of the target hour, GW"); ax.set_ylabel("mean absolute price error, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax); ttl(ax, "Held-out year, two days ahead: the state helps\nbelow 40 GW of load and not above")
fig.suptitle("How much of the long-window model's error is known ahead? Shocks carried forward from a 14:00 origin (realised load and fuel prices of the target hour: this isolates the state, it is not a forecast test)", x=0.125, ha="left", fontsize=10.5, fontweight="bold", color=INK, y=1.06)
fig.savefig(f"{SP}/out/market0_horizon.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_horizon.png")
np.save(f"{SP}/out/market0_horizon_res.npy", np.array([(k_[0], k_[1], 0 if k_[2] == "fit" else 1, 0 if k_[3] == "all" else 1) + v for k_, v in res.items()]))

# the same carried state on the candidate's origins and target hours (D+2, hours 0-7 and 0-13). Hourly prices and realised load: both favour this model.
m, pq, pp = keep; day = np.asarray(loc.normalize().tz_localize(None)); PER = {"winter": pd.date_range("2025-12-01", periods=30, freq="3D"), "summer": pd.date_range("2026-03-31", periods=50, freq="3D"), "autumn": pd.date_range("2026-09-02", periods=66, freq="1D")}
for nm, org in PER.items():
    tg = np.isin(day, (org + pd.Timedelta(days=2)).values); row = []
    for hh in (8, 14):
        k = m & tg & (hod < hh); row.append(tuple(round(float(np.mean(np.abs(x[k] - G.p.values[k]))) / 10, 2) for x in (G.p0.values, pp, pq)) + (int(k.sum()),))
    print(f"candidate's origins, {nm}: night (curves only, price error carried, shocks carried, hours) {row[0]} | first 14 h {row[1]}")
