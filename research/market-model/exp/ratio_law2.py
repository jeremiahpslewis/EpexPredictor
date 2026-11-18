"""Mid load: is the price relative to gas-fired cost a function of what is left for gas after the cheaper plants (capacity), or do the cheaper plants hold back (offers)?
x = realised residual load minus available nuclear, lignite and hard coal (trailing 14-day maximum of each one's output)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
A = pd.read_parquet(f"{SP}/out/hist2123.parquet").loc["2021-01-01":"2023-12-31"]; A = A.rename(columns={"p": "price", "resload": "rl", "hcoal": "coal_out", "lignite": "lig_out", "nuclear": "nuc_out", "gas_out": "gas_gen"})[["price", "rl", "gas", "eua", "lig_out", "coal_out", "nuc_out", "gas_gen"]]
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1)
B = pd.DataFrame({"price": D["p_de"], "rl": D["rlreal_de"], "gas": D["gas"], "eua": ed.reindex(loc.normalize().tz_localize(None)).values, "lig_out": D["g_de_lignite"], "coal_out": D["g_de_coal"], "nuc_out": 0.0, "gas_gen": D["g_de_gas"]}, index=D.index)
X = pd.concat([A, B]).sort_index(); X = X[~X.index.duplicated()]; X["nuc_out"] = X["nuc_out"].fillna(0.0); X["L"] = 1.85 * X["gas"] + 0.37 * X["eua"]; X["r"] = X["price"] / X["L"]
for c in ("lig", "coal", "nuc"): X[c + "_av"] = X[c + "_out"].rolling(14 * 24, min_periods=7 * 24).max().shift(24)
X["cheap"] = X["lig_av"] + X["coal_av"] + X["nuc_av"]; X["x"] = X["rl"] - X["cheap"]; X["cheap_out"] = X["lig_out"] + X["coal_out"] + X["nuc_out"]
lc = X.index.tz_convert("Europe/Berlin"); X["half"] = np.array([f"{y}H{1 if m <= 6 else 2}" for y, m in zip(lc.year, lc.month)]); X = X.dropna(subset=["price", "rl", "L", "cheap"]); hs = sorted(set(X["half"]))
bins = [(-40, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30), (30, 50)]
print("median price / gas-fired cost by (residual load minus available nuclear + lignite + hard coal), GW")
print("          L   avail   " + "".join(f"[{a},{b})".rjust(10) for a, b in bins))
T = []
for h in hs:
    x = X[X["half"] == h]; row = [np.median(x["r"][(x["x"] >= a) & (x["x"] < b)]) if ((x["x"] >= a) & (x["x"] < b)).sum() >= 40 else np.nan for a, b in bins]; T.append(row)
    print(f"  {h} {x['L'].mean():4.0f}  {x['cheap'].mean():5.1f}   " + "".join(f"{v:10.2f}" if np.isfinite(v) else "         ." for v in row))
M = np.array(T); print("  spread across half-years (sd / mean):" + "".join(f"{np.nanstd(M[:, i]) / abs(np.nanmean(M[:, i])):10.2f}" for i in range(len(bins))))
print("\nloading of the cheaper plants (output / available) and gas output, in the same bins: 2021H2 against 2026H2, equal gas-fired cost")
for h in ("2021H2", "2022H2", "2025H2", "2026H2"):
    x = X[X["half"] == h]; print(f"  {h}: " + "  ".join(f"[{a},{b}) {np.median((x['cheap_out'] / x['cheap'])[(x['x'] >= a) & (x['x'] < b)]):.2f} / gas {np.median(x['gas_gen'][(x['x'] >= a) & (x['x'] < b)]):.1f}" if ((x["x"] >= a) & (x["x"] < b)).sum() >= 40 else f"[{a},{b}) ." for a, b in bins))
SEQ = plt.get_cmap("Blues")(np.linspace(0.3, 1.0, len(hs))); INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
for j, (col, e) in enumerate((("rl", np.arange(-10, 71, 5.0)), ("x", np.arange(-45, 46, 5.0)))):
    for i, h in enumerate(hs):
        x = X[X["half"] == h]; mid, r_ = [], []
        for a, b in zip(e[:-1], e[1:]):
            k = (x[col] >= a) & (x[col] < b)
            if k.sum() >= 40: mid.append(0.5 * (a + b)); r_.append(np.median(x["r"][k]))
        ax[j].plot(mid, r_, "-", color=SEQ[i], linewidth=1.8, label=f"{h[:4]} {'Jan-Jun' if h.endswith('1') else 'Jul-Dec'}")
    ax[j].set_ylim(-0.1, 2.4); ax[j].grid(True, color="#e9e8e3", linewidth=0.6); ax[j].set_axisbelow(True); ax[j].set_ylabel("median price / gas-fired cost")
    for s_ in ("top", "right"): ax[j].spines[s_].set_visible(False)
ax[0].set_title("Against residual load", loc="left", fontsize=10, fontweight="bold", color=INK); ax[0].set_xlabel("German residual load, GW"); ax[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax[1].set_title("Against residual load minus available nuclear, lignite and hard coal", loc="left", fontsize=10, fontweight="bold", color=INK); ax[1].set_xlabel("GW left for gas, storage and imports")
fig.tight_layout(); fig.savefig(f"{SP}/out/ratio_law2.png", dpi=110); print("figure:", f"{SP}/out/ratio_law2.png")
