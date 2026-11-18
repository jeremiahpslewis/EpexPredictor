"""One plant type at a time, no market model: is German lignite / hard coal / gas output a stable function of the price once
fuel and carbon costs are taken out?  Utilisation = (output - floor) / (available - floor), available = trailing 14-day max
of output (known a day ahead).  The price each plant sees is a blend of the hour's price and the day's mean (inertia)."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet"); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
H = D.resample("1h").mean(); day = H.index.tz_convert("Europe/Berlin").normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1)
H["eua"] = ed.reindex(day.tz_localize(None)).values; H["pbar"] = H["p_de"].groupby(day).transform("mean").values
FUEL = {"lignite": dict(col="g_de_lignite", lam=0.4, ef=1.0, hr=0.0), "hard coal": dict(col="g_de_coal", lam=0.5, ef=0.85, hr=0.0), "gas": dict(col="g_de_gas", lam=0.35, ef=0.37, hr=1.85)}
PER = [("2024-07", "2024-12"), ("2025-01", "2025-06"), ("2025-07", "2025-12"), ("2026-01", "2026-03"), ("2026-04", "2026-07"), ("2026-08", "2026-11")]
LAB = ["Jul-Dec 2024", "Jan-Jun 2025", "Jul-Dec 2025", "Jan-Mar 2026", "Apr-Jul 2026", "Aug-Nov 2026"]
ym = H.index.tz_convert("Europe/Berlin").strftime("%Y-%m")
sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -50, 50)))
out = {}
for name, f in FUEL.items():
    g = H[f["col"]]; avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24)
    u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3); peff = f["lam"] * H["pbar"] + (1 - f["lam"]) * H["p_de"]
    out[name] = pd.DataFrame({"u": u, "peff": peff, "avail": avail, "floor": floor, "g": g, "gas": H["gas"], "eua": H["eua"]})
# 1. per month: the price at which the plant type is half used (logistic fit of utilisation on the blended price, width free)
def half(df):
    df = df.dropna(); x = df["peff"].values; y = df["u"].clip(0, 1).values
    if len(df) < 150: return np.nan, np.nan
    best = (1e9, np.nan, np.nan)
    for w in (8, 12, 16, 22, 30):
        for m in np.arange(0, 420, 2.0):
            e = np.mean(np.abs(y - sg((x - m) / w)));
            if e < best[0]: best = (e, m, w)
    return best[1], best[2]
rows = []
for m_ in sorted(set(ym)):
    k = ym == m_; r = {"month": m_, "gas price": H["gas"][k].mean(), "eua": H["eua"][k].mean(), "price": H["p_de"][k].mean()}
    for name in FUEL: r[name], r[name + " w"] = half(out[name][k]); r[name + " avail"] = out[name]["avail"][k].mean(); r[name + " floor"] = out[name]["floor"][k].mean()
    rows.append(r)
T = pd.DataFrame(rows).set_index("month"); pd.set_option("display.width", 250); print(T.round(1).to_string())
print("\nregression of the monthly half-utilisation price on carbon price and gas price (months from 2024-07):")
Tm = T.loc["2024-07":].dropna()
for name in FUEL:
    X = np.column_stack([np.ones(len(Tm)), Tm["eua"], Tm["gas price"]]); b, *_ = np.linalg.lstsq(X, Tm[name].values, rcond=None); res = Tm[name].values - X @ b
    Xa = X[:-9]; ba, *_ = np.linalg.lstsq(Xa, Tm[name].values[:-9], rcond=None); fw = Tm[name].values[-9:] - X[-9:] @ ba
    print(f"  {name:9s}: all months  {b[0]:+6.1f} {b[1]:+5.2f} x carbon {b[2]:+5.2f} x gas, residual sd {res.std():4.1f} EUR/MWh | fitted to Jan 2026, then applied to Feb-Nov 2026: {ba[0]:+6.1f} {ba[1]:+5.2f} x carbon {ba[2]:+5.2f} x gas, forward errors {np.round(fw, 0)}")
T.to_csv(f"{SP}/out/drift_look.csv")
# 2. figure: utilisation against the cost-adjusted price, one line per period
SEQ = ["#b9d3f2", "#8fb9ea", "#5f9be0", "#2a78d6", "#1b56a3", "#0e3568"]; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
for j, (name, f) in enumerate(FUEL.items()):
    df = out[name]
    for row, (xlab, x) in enumerate([("price the plant sees, EUR/MWh", df["peff"]), ("price the plant sees minus fuel and carbon cost, EUR/MWh", df["peff"] - f["ef"] * df["eua"] - f["hr"] * df["gas"])]):
        a = ax[row, j]
        for i, (lo, hi) in enumerate(PER):
            k = (ym >= lo) & (ym <= hi) & df["u"].notna().values & x.notna().values
            if k.sum() < 300: continue
            xs = x[k].values; us = df["u"][k].values; e = np.quantile(xs, np.linspace(0.02, 0.98, 15)); mid, med = [], []
            for a_, b_ in zip(e[:-1], e[1:]):
                kk = (xs >= a_) & (xs < b_)
                if kk.sum() > 20: mid.append(np.median(xs[kk])); med.append(np.median(us[kk]))
            a.plot(mid, med, "-o", color=SEQ[i], linewidth=2, markersize=3.5, label=LAB[i])
        a.set_ylim(-0.1, 1.15); a.set_xlabel(xlab); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
        if j == 0: a.set_ylabel("median utilisation of available capacity")
        a.set_title(f"German {name}" + ("" if row == 0 else f": costs taken out ({f['ef']:.2f} t/MWh" + (f", heat rate {f['hr']:.2f})" if f["hr"] else ")")), loc="left", fontsize=10, fontweight="bold", color=INK)
        if row == 0 and j == 0: a.legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Does each plant type respond to price the same way in every period?  Top: raw price.  Bottom: fuel and carbon cost subtracted.", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{SP}/out/drift_look.png", dpi=110); print("figure:", f"{SP}/out/drift_look.png")
