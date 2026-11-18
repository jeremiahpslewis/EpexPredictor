"""Is the German price relative to the cost of gas-fired power a stable function of residual load, 2021-2026?  L = 1.85 gas + 0.37 carbon.
Also: is lignite / hard coal loading organised by the price relative to L, or by the margin over the plant's own cost?"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
A = pd.read_parquet(f"{SP}/out/hist2123.parquet").loc["2021-01-01":"2023-12-31"]; A = A.rename(columns={"p": "price", "resload": "rl", "hcoal": "coal_out", "lignite": "lig_out"})[["price", "rl", "gas", "eua", "lig_out", "coal_out"]]
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1)
B = pd.DataFrame({"price": D["p_de"], "rl": D["rlreal_de"], "gas": D["gas"], "eua": ed.reindex(loc.normalize().tz_localize(None)).values, "lig_out": D["g_de_lignite"], "coal_out": D["g_de_coal"]}, index=D.index)
X = pd.concat([A, B]).sort_index(); X = X[~X.index.duplicated()]; X["L"] = 1.85 * X["gas"] + 0.37 * X["eua"]; X["r"] = X["price"] / X["L"]; lc = X.index.tz_convert("Europe/Berlin")
half = np.array([f"{y}H{1 if m <= 6 else 2}" for y, m in zip(lc.year, lc.month)]); X["half"] = half; X = X.dropna(subset=["price", "rl", "L"])
bins = [(-30, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55, 90)]; hs = sorted(set(X["half"]))
print("median of price / gas-fired cost by residual-load bin (realised residual load, GW); L = mean gas-fired cost of the half-year")
print("            L  gas  " + "".join(f"[{a},{b})".rjust(10) for a, b in bins))
T = {}
for h in hs:
    x = X[X["half"] == h]; row = []
    for a, b in bins:
        k = (x["rl"] >= a) & (x["rl"] < b); row.append(np.median(x["r"][k]) if k.sum() >= 40 else np.nan)
    T[h] = row; print(f"  {h}  {x['L'].mean():4.0f} {x['gas'].mean():4.0f}  " + "".join(f"{v:10.2f}" if np.isfinite(v) else "         ." for v in row))
M = np.array([T[h] for h in hs]); print("  spread across the twelve half-years (sd / mean):  " + "".join(f"{np.nanstd(M[:, i]) / np.nanmean(M[:, i]):10.2f}" for i in range(len(bins))))
print("the same in absolute price (EUR/MWh), sd / mean:    " + "".join(f"{np.nanstd(v) / np.nanmean(v):10.2f}" for v in np.array([[np.median(X[(X['half'] == h) & (X['rl'] >= a) & (X['rl'] < b)]['price']) if ((X['half'] == h) & (X['rl'] >= a) & (X['rl'] < b)).sum() >= 40 else np.nan for h in hs] for a, b in bins])))
# plant loading: own margin against relative price
for nm, col, own in (("lignite", "lig_out", lambda x: x["eua"] * 1.0 + 12), ("hard coal", "coal_out", lambda x: 0.85 * x["eua"] + 28)):
    g = X[col]; avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24); X["u"] = ((g - floor) / (avail - floor)).clip(-0.2, 1.3); X["m"] = X["price"] - own(X)
    print(f"\n{nm}: median loading by half-year. Left: by margin over own cost [0,30), [30,80). Right: by price / gas-fired cost [0.6,0.8), [0.8,1.0), [1.0,1.3)")
    for h in hs:
        x = X[(X["half"] == h) & X["u"].notna()]; f = lambda k: f"{np.median(x['u'][k]):5.2f} ({k.sum():4d})" if k.sum() >= 40 else "      .     "
        print(f"  {h} L {x['L'].mean():4.0f} | " + "  ".join(f((x["m"] >= a) & (x["m"] < b)) for a, b in ((0, 30), (30, 80))) + "  |  " + "  ".join(f((x["r"] >= a) & (x["r"] < b)) for a, b in ((0.6, 0.8), (0.8, 1.0), (1.0, 1.3))))
# figure: ratio against residual load, one line per half-year
SEQ = plt.get_cmap("Blues")(np.linspace(0.3, 1.0, len(hs))); INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8)); e = np.arange(-10, 71, 5.0)
for i, h in enumerate(hs):
    x = X[X["half"] == h]; mid, r_, p_ = [], [], []
    for a, b in zip(e[:-1], e[1:]):
        k = (x["rl"] >= a) & (x["rl"] < b)
        if k.sum() >= 40: mid.append(0.5 * (a + b)); r_.append(np.median(x["r"][k])); p_.append(np.median(x["price"][k]))
    ax[0].plot(mid, p_, "-", color=SEQ[i], linewidth=1.8, label=f"{h[:4]} {'Jan-Jun' if h.endswith('1') else 'Jul-Dec'} (gas {x['gas'].mean():.0f})"); ax[1].plot(mid, r_, "-", color=SEQ[i], linewidth=1.8)
ax[0].set_title("German price against residual load, by half-year", loc="left", fontsize=10, fontweight="bold", color=INK); ax[0].set_ylabel("median price, EUR/MWh"); ax[1].set_title("The same prices divided by the cost of gas-fired power", loc="left", fontsize=10, fontweight="bold", color=INK); ax[1].set_ylabel("median price / (1.85 gas + 0.37 carbon)")
for a in ax:
    a.set_xlabel("German residual load (realised), GW"); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2); ax[1].set_ylim(-0.1, 2.6)
fig.tight_layout(); fig.savefig(f"{SP}/out/ratio_law.png", dpi=110); print("figure:", f"{SP}/out/ratio_law.png")
