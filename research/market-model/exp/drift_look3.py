import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet"); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
M = pd.read_parquet(f"{SP}/out/de_mix.parquet"); print("de_mix columns:", M.columns.tolist()[:40], M.index.min(), M.index.max())
H = D.resample("1h").mean(); loc = H.index.tz_convert("Europe/Berlin"); day = loc.normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1)
H["eua"] = ed.reindex(day.tz_localize(None)).values; H["pbar"] = H["p_de"].groupby(day).transform("mean").values
ym = loc.strftime("%Y-%m"); A = (ym >= "2024-07") & (ym <= "2026-03"); B = (ym >= "2026-08")
g = H["g_de_lignite"]; avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24)
u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3).values; mh = (H["p_de"] - H["eua"]).values; md = (H["pbar"] - H["eua"]).values
pm1 = H["p_de"].shift(1).values - H["eua"].values; pp1 = H["p_de"].shift(-1).values - H["eua"].values; hr = loc.hour.values
for lab, per in (("early", A), ("late", B)):
    k = per & (mh >= 30) & (mh < 60) & (md >= 40) & np.isfinite(u)
    print(f"\n{lab}: lignite, hours with hour margin 30..60 and day margin >= 40: n {k.sum()}, median utilisation {np.median(u[k]):.2f}, output {np.median(g.values[k]):.1f} GW of available {np.median(avail.values[k]):.1f}; previous hour's margin median {np.nanmedian(pm1[k]):.0f} (5..95% {np.nanpercentile(pm1[k], 5):.0f}..{np.nanpercentile(pm1[k], 95):.0f}), next hour's {np.nanmedian(pp1[k]):.0f} ({np.nanpercentile(pp1[k], 5):.0f}..{np.nanpercentile(pp1[k], 95):.0f}); RL {np.median(H['rl_de'].values[k]):.1f} GW")
    print("   by hour of day (count, median utilisation): " + "  ".join(f"{h}h {(k & (hr == h)).sum()} {np.median(u[k & (hr == h)]):.2f}" for h in range(24) if (k & (hr == h)).sum() >= 8))
    spread = np.abs(pp1 - pm1)
    for a_, b_ in ((0, 20), (20, 60), (60, 1e9)):
        kk = k & (spread >= a_) & (spread < b_); print(f"   neighbours' prices differ by {a_}..{b_ if b_ < 1e8 else ''}: n {kk.sum()}, utilisation {np.median(u[kk]) if kk.sum() else np.nan:.2f}")
    # hours on flat stretches: this hour, previous and next all within 15 EUR/MWh
    for lo_, hi_ in ((10, 30), (30, 60), (60, 120), (120, 1e9)):
        kk = per & (mh >= lo_) & (mh < hi_) & (np.abs(pm1 - mh) < 15) & (np.abs(pp1 - mh) < 15) & np.isfinite(u); print(f"   flat stretches (neighbours within 15), hour margin {lo_}..{hi_ if hi_ < 1e8 else ''}: n {kk.sum()}, utilisation {np.median(u[kk]) if kk.sum() else np.nan:.2f}")
# time series: two weeks each
INK, MUTED, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(4, 2, figsize=(14, 9.5), sharex="col")
for c, (t0, ttl) in enumerate((("2025-10-06", "6-19 October 2025 (gas 33 EUR/MWh)"), ("2026-10-05", "5-18 October 2026 (gas 86 EUR/MWh)"))):
    s = H.loc[pd.Timestamp(t0, tz="Europe/Berlin"):pd.Timestamp(t0, tz="Europe/Berlin") + pd.Timedelta(days=14)]; x = s.index.tz_convert("Europe/Berlin")
    ax[0, c].plot(x, s["p_de"], color=ORANGE, linewidth=1.3, label="day-ahead price"); ax[0, c].plot(x, s["eua"] * 1.0 + 12, color=MUTED, linewidth=1, linestyle="--", label="carbon cost of lignite + 12")
    ax[0, c].set_ylabel("EUR/MWh"); ax[0, c].set_title(ttl, loc="left", fontsize=10, fontweight="bold", color=INK); ax[0, c].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for r, (col, nm) in enumerate((("g_de_lignite", "lignite output"), ("g_de_coal", "hard coal output"), ("g_de_gas", "gas output")), start=1):
        ax[r, c].plot(x, s[col], color=BLUE, linewidth=1.3); ax[r, c].set_ylabel("GW"); ax[r, c].set_ylim(0, None); ax[r, c].text(0.01, 0.9, nm, transform=ax[r, c].transAxes, color=INK, fontsize=9, fontweight="bold")
    for r in range(4):
        ax[r, c].grid(True, color="#e9e8e3", linewidth=0.6); ax[r, c].set_axisbelow(True)
        for s_ in ("top", "right"): ax[r, c].spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{SP}/out/drift_look3.png", dpi=105); print("figure:", f"{SP}/out/drift_look3.png")
