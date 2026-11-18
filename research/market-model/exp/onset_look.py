"""Date the change in solid-fuel dispatch: per week, the shift in EUR/MWh that puts that week's utilisation-against-margin points on the reference curve
(reference = logistic fitted to Jul 2024 - Dec 2025). German lignite and hard coal; Polish and Czech lignite as controls."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); G = pd.read_parquet(f"{SP}/out/plcz_gen.parquet").reindex(D.index); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); eua = ed.reindex(loc.normalize().tz_localize(None)).values
sg = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50))); wk = np.asarray((loc.year - 2024) * 12 + loc.month - 7); ref = (loc >= "2024-07-01") & (loc < "2026-01-01")
SER = [("German lignite", D["g_de_lignite"], "p_de", 1.0, 12.0), ("German hard coal", D["g_de_coal"], "p_de", 0.85, 25.0), ("Polish lignite", G["pl_lignite"], "p_pl", 1.0, 12.0), ("Czech lignite", G["cz_lignite"], "p_cz", 1.0, 12.0)]
out = {}
for name, g, pcol, ef, c0 in SER:
    avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24); u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3).values; m = (D[pcol] - ef * eua - c0).values; ok = np.isfinite(u) & np.isfinite(m)
    best = (1e9,)
    for L in (0.8, 0.85, 0.9, 0.95):
        for m0 in np.arange(-30, 41, 2.0):
            for w in (6, 9, 12, 16, 22, 30):
                e = np.mean(np.abs(u[ok & ref] - L * sg((m[ok & ref] - m0) / w)))
                if e < best[0]: best = (e, L, m0, w)
    _, L, m0, w = best; dl = np.arange(-40, 121, 2.0); rows = []
    for k in range(int(wk.max()) + 1):
        kk = ok & (wk == k) & (m > m0 - 3 * w - 40)
        if kk.sum() < 300 or (m[kk] > m0).sum() < 80: rows.append(np.nan); continue
        rows.append(dl[np.argmin([np.mean(np.abs(u[kk] - L * sg((m[kk] - d_ - m0) / w))) for d_ in dl])])
    out[name] = np.array(rows); print(f"{name}: reference curve plateau {L}, half-way margin {m0:+.0f}, width {w}")
t = pd.date_range("2024-07-01", periods=len(rows), freq="MS"); T = pd.DataFrame(out, index=t); T["gas"] = D["gas"].groupby(wk).mean().reindex(range(len(rows))).values; T["price DE"] = D["p_de"].groupby(wk).mean().reindex(range(len(rows))).values
pd.set_option("display.width", 200); print(T.loc["2025-09-01":].round(0).to_string()); T.to_csv(f"{SP}/out/onset_look.csv")
COL = {"German lignite": "#2a78d6", "German hard coal": "#eb6834", "Polish lignite": "#8a8983", "Czech lignite": "#c9c8c2"}; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, a = plt.subplots(figsize=(11, 4.6))
for name in ("Czech lignite", "Polish lignite", "German hard coal", "German lignite"): a.plot(T.index, T[name], "-o", color=COL[name], linewidth=2, markersize=3, label=name)
a.axhline(0, color=MUTED, linewidth=0.8); a.set_ylabel("EUR/MWh"); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True); a.legend(frameon=False, loc="upper left", ncol=4)
for s_ in ("top", "right"): a.spines[s_].set_visible(False)
a.set_title("How much more than in 2024-25 a plant type asks before it runs, per month (carbon cost already subtracted)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{SP}/out/onset_look.png", dpi=110); print("figure:", f"{SP}/out/onset_look.png")
