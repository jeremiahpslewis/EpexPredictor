"""Market model M0c (notes 21.9): what the carried state does NOT know. Innovations = shock minus the state carried two days ahead (mean of the last 7 days,
by hour of day), binned against CHANGES in what the model leaves out. Envelope: innovation series shifted by whole weeks. Realised load and fuel prices."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market0 as M
SP = M.SP; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); R = pd.read_parquet(f"{SP}/out/market0_rows.parquet").join(pd.read_parquet(f"{SP}/out/zoom1_rows.parquet")[["nbs", "teff"]], how="left"); th = {k: float(np.median(np.load(f"{SP}/out/market0_post.npz")[k])) for k in M.PN}
R["pdm"] = (R.p - R.p.groupby(R.index.normalize()).transform("mean")) / R.L
full = pd.date_range(R.index.min().tz_convert("UTC").floor("D"), R.index.max().tz_convert("UTC").ceil("D"), freq="1h", tz="UTC"); G = R.copy(); G.index = G.index.tz_convert("UTC"); G = G.reindex(full); loc = G.index.tz_convert("Europe/Berlin"); hod = np.asarray(loc.hour); dow = np.asarray(loc.dayofweek); n = len(G)
def state(v, h=2, N=7):
    lag0 = np.where(hod <= 12, 24 * h, 24 * (h + 1)); acc = np.zeros(n); cnt = np.zeros(n); idx = np.arange(n)
    for j in range(N):
        src = idx - lag0 - 24 * j; x = np.where(src >= 0, v[np.clip(src, 0, n - 1)], np.nan); m = ~np.isnan(x); acc[m] += x[m]; cnt[m] += 1
    return np.where(cnt >= 4, acc / np.maximum(cnt, 1), np.nan)
S = {c: state(G[c].values) for c in ("eT", "eX", "d", "teff", "nbs", "rlreal", "L", "pdm")}; m = ~np.isnan(S["eT"] + S["eX"] + S["d"] + G.p.values); pq = np.full(n, np.nan); pq[m] = M.clear(th, G.rlreal.values[m], G.L.values[m], d=S["d"][m], eT=S["eT"][m], eX=S["eX"][m])[0]
I = {"thermal": G.eT.values - S["eT"], "imports": G.eX.values - S["eX"], "third": G.d.values - S["d"], "price": (G.p.values - pq) / G.L.values, "raw thermal": G.eT.values, "raw imports": G.eX.values, "raw third": G.d.values, "raw price": (G.p.values - G.p0.values) / G.L.values}
C = {"change in temperature, C": (G.teff.values - S["teff"], np.r_[-12, -6, -3, -1, 1, 3, 6, 12]), "change in neighbours' residual load, GW": (G.nbs.values - S["nbs"], np.r_[-40, -20, -10, -4, 4, 10, 20, 40]), "change in residual load, GW": (G.rlreal.values - S["rlreal"], np.r_[-40, -20, -10, -4, 4, 10, 20, 40]),
     "change in gas-plus-CO2 cost, %": (100 * (G.L.values / S["L"] - 1), np.r_[-20, -8, -3, -1, 1, 3, 8, 20]), "day of the week (Mon = 0)": (dow.astype(float), np.arange(-0.5, 7, 1.0)), "hour of the day": (hod.astype(float), np.arange(-0.5, 24, 2.0)), "hour's price minus day's mean (price / cost)": (G.pdm.values, np.r_[-1.2, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.2]), "residual load, GW": (G.rlreal.values, np.arange(-10, 71, 10.0))}
def bins(y, x, e, minn=150):
    b = np.digitize(x, e) - 1; ok = ~np.isnan(y + x); return np.array([np.median(y[ok & (b == i)]) if (ok & (b == i)).sum() >= minn else np.nan for i in range(len(e) - 1)])
def env(y, x, e, nrep=120):
    A = np.array([bins(np.roll(y, int(k) * 168), x, e) for k in rng.integers(4, n // 168 - 4, nrep)]); return np.nanpercentile(A, [5, 95], axis=0)
T = {}
for inm, y in I.items():
    for cnm, (x, e) in C.items():
        med = bins(y, x, e); lo, hi = env(y, x, e); T[(inm, cnm)] = (med, lo, hi, float(np.nanmax(med) - np.nanmin(med)), int(np.nansum((med > hi) | (med < lo))), int(np.sum(~np.isnan(med))))
print(f"{'range of binned medians (bins outside envelope / bins)':60s}" + "".join(f"{k:>22s}" for k in I))
for cnm in C: print(f"{cnm:60s}" + "".join(f"{T[(k, cnm)][3]:12.2f} ({T[(k, cnm)][4]:2d}/{T[(k, cnm)][5]:2d})   " for k in I))
print("sd of hourly values:", {k: round(float(np.nanstd(v)), 3) for k, v in I.items()})
np.save(f"{SP}/out/market0_innov_T.npy", np.array([(a, b, v[3], v[4], v[5]) for (a, b), v in T.items()], dtype=object), allow_pickle=True)
if len(sys.argv) > 1 and sys.argv[1] == "fig":
    PAN = (("price", "change in residual load, GW"), ("price", "residual load, GW"), ("price", "hour of the day"), ("price", "day of the week (Mon = 0)"), ("price", "change in temperature, C"), ("price", "change in gas-plus-CO2 cost, %"), ("price", "hour's price minus day's mean (price / cost)"), ("price", "change in neighbours' residual load, GW"),
           ("thermal", "change in temperature, C"), ("thermal", "hour's price minus day's mean (price / cost)"), ("imports", "change in neighbours' residual load, GW"), ("third", "hour of the day"))
    fig = plt.figure(figsize=(18.5, 13.2)); gs = fig.add_gridspec(3, 4, hspace=0.48, wspace=0.30)
    for i, (inm, cnm) in enumerate(PAN):
        ax = fig.add_subplot(gs[i // 4, i % 4]); x, e = C[cnm]; cen = 0.5 * (e[:-1] + e[1:]); med, lo, hi, rg, no, nb = T[(inm, cnm)]; rmed, _, _, rrg, _, _ = T[("raw " + inm, cnm)]
        ax.fill_between(cen, lo, hi, color=NEUT, alpha=0.3, linewidth=0, label="no relation (whole-week shifts)"); ax.plot(cen, rmed - np.nanmean(rmed), "-", color=NEUT, linewidth=1.6, label="before the state (centred)"); ax.plot(cen, med, "-o", color=BLUE if inm == "price" else ORANGE, linewidth=2, markersize=4, label="after the state"); ax.axhline(0, color=INK, linewidth=0.7)
        ax.set_xlabel(cnm); ax.set_ylabel("price error, price / cost" if inm == "price" else f"{inm} shock, GW"); clean(ax)
        if i in (0, 8): ax.legend(frameon=False, loc="best", fontsize=7.5)
        u = "" if inm == "price" else " GW"; ttl(ax, f"{'Price error' if inm == 'price' else inm.capitalize() + ' shock'}: range {rrg:.2f}{u} before the state,\n{rg:.2f}{u} after; {no} of {nb} bins outside the envelope")
    fig.suptitle("What last week's state does not know. Errors two days ahead after carrying the state, against changes in what the model leaves out (M0c; realised load and fuel prices)", x=0.125, ha="left", fontsize=10.5, fontweight="bold", color=INK, y=0.93)
    fig.savefig(f"{SP}/out/market0_innov.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_innov.png")
