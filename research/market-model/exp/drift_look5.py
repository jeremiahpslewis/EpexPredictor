"""Heat-led plants: is the output a plant type delivers when it clearly pays (and the output it keeps when it clearly does not) a function of temperature,
the same in every year?  Per day: mean output in the hours with margin >= +30 EUR/MWh ('pays') and in the hours with margin <= -30 ('does not pay')."""
import numpy as np, pandas as pd, live_common as lc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = lc.SP; D = pd.read_parquet(f"{SP}/out/star_data.parquet"); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
F = lc.load()["DE"]; temp = F[[c for c in F.columns if c.startswith("temp_")]].mean(1)
H = D.resample("1h").mean(); H["temp"] = temp.resample("1h").mean().reindex(H.index); loc = H.index.tz_convert("Europe/Berlin"); day = loc.normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); H["eua"] = ed.reindex(day.tz_localize(None)).values
FU = {"lignite": ("g_de_lignite", 1.0, 0.0, 12.0), "hard coal": ("g_de_coal", 0.85, 0.0, 25.0), "gas": ("g_de_gas", 0.37, 1.85, 0.0)}
rows = []
for name, (col, ef, hr, c0) in FU.items():
    m = H["p_de"] - (ef * H["eua"] + hr * H["gas"] + c0); g = H[col]
    df = pd.DataFrame({"day": day, "g": g.values, "m": m.values, "t": H["temp"].values}).dropna()
    for dd, x in df.groupby("day"):
        hi = x[x["m"] >= 30]; lo = x[x["m"] <= -30]
        rows.append({"fuel": name, "day": dd, "temp": x["t"].mean(), "pays": hi["g"].mean() if len(hi) >= 4 else np.nan, "npays": len(hi), "not": lo["g"].mean() if len(lo) >= 3 else np.nan, "mx": x["g"].max(), "mn": x["g"].min(), "wd": dd.weekday()})
R = pd.DataFrame(rows); R["yr"] = R["day"].dt.year; R["per"] = np.where(R["day"] >= pd.Timestamp("2026-08-01", tz="Europe/Berlin"), "Aug-Nov 2026", np.where(R["day"] >= pd.Timestamp("2026-03-01", tz="Europe/Berlin"), "Mar-Jul 2026", "before Mar 2026"))
R.to_parquet(f"{SP}/out/drift_look5.parquet"); tb = [-10, 3, 8, 13, 18, 30]
for name in FU:
    x = R[R["fuel"] == name]; print(f"\n{name}: median output GW (number of days) by daily mean temperature band; 'pays' = hours with margin >= +30, 'does not' = hours with margin <= -30")
    for what in ("pays", "not"):
        for per in ("before Mar 2026", "Mar-Jul 2026", "Aug-Nov 2026"):
            y = x[x["per"] == per]; print(f"   {'when it pays     ' if what == 'pays' else 'when it does not '} {per:16s} " + "  ".join(f"[{a},{b}) " + (f"{y[(y['temp'] >= a) & (y['temp'] < b)][what].median():5.1f} ({y[(y['temp'] >= a) & (y['temp'] < b)][what].notna().sum():3d})" if y[(y['temp'] >= a) & (y['temp'] < b)][what].notna().sum() >= 4 else "    .      ") for a, b in zip(tb[:-1], tb[1:])))
COL = {"before Mar 2026": "#9cc0ee", "Mar-Jul 2026": "#2a78d6", "Aug-Nov 2026": "#0e3568"}; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
for j, name in enumerate(FU):
    for r, (what, ttl) in enumerate((("pays", "output in the hours when running clearly pays"), ("not", "output in the hours when it clearly does not"))):
        a = ax[r, j]
        for per, c in COL.items():
            y = R[(R["fuel"] == name) & (R["per"] == per)].dropna(subset=[what]); a.scatter(y["temp"], y[what], s=9, color=c, alpha=0.75, linewidths=0, label=per)
        a.set_title(f"German {name}: {ttl}", loc="left", fontsize=9.5, fontweight="bold", color=INK); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True); a.set_ylim(0, None)
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
        if r == 1: a.set_xlabel("daily mean temperature, deg C")
        if j == 0: a.set_ylabel("GW (one dot per day)")
ax[0, 0].legend(frameon=False, loc="lower left", markerscale=1.6)
fig.suptitle("Is what a plant type delivers set by the weather as well as by the price?  Daily output against temperature", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{SP}/out/drift_look5.png", dpi=110); print("figure:", f"{SP}/out/drift_look5.png")
