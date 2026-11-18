"""Figure for the state layer: (top) one origin's filter path through days without evidence; (bottom) predicted against realised level for every origin."""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
import oc
from ocx_state import *
BLUE, ORANGE, INK, MUTED, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
TH = np.load(f"{SP}/out/ocxw{SFX}_th.npz"); dates = fit_dates(); fh = [hx(t) for t in dates]
per = sys.argv[1] if len(sys.argv) > 1 else "summer"; I = np.load(f"{SP}/out/inputs_{per}.npz"); Dg = np.load(f"{SP}/out/ocxs{SFX}_{per}_diag.npz"); th = tuple(TH[per])
# origin for the top panel: the one whose last 45 days have the most days without evidence (ties: D+1 missing first)
best = None
for j in range(len(I["cuts"])):
    h0 = int(I["cuts"][j]) // 4; A = get_fit(dates[max(i for i in range(len(dates)) if fh[i] <= h0 - 48)]); zh = hist_evidence(A, h0, float(I["gas"][j]), EUA[h0 - 48]); zz = np.r_[zh, Dg["z1_ts"][j]]; sc = np.isnan(zz[-45:]).sum() + 2 * np.isnan(zz[-1]) + 0.5 * np.isnan(zz[-8:]).sum()
    if best is None or sc > best[0]: best = (sc, j, zz)
_, j, zz = best; radd = np.zeros(len(zz)); radd[-1] = Dg["v_load"][j]; k = kf2(zz, th, radd=radd); (s2, v2), (s3, v3) = ahead(k, th, 2); D2 = pd.Timestamp(str(I["cutoffs"][j])[:10]) + pd.Timedelta(days=1) if str(I["cutoffs"][j])[11:13] in ("22", "23") else pd.Timestamp(str(I["cutoffs"][j])[:10])
days = pd.date_range(end=D2 - pd.Timedelta(days=1), periods=len(zz), freq="D"); NSHOW = 45; sl = slice(len(zz) - NSHOW, len(zz))
fig = plt.figure(figsize=(13.2, 8.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[30 * 3, 50 * 3, 66], hspace=0.42, wspace=0.08)
ax = fig.add_subplot(gs[0, :]); xs = list(days[sl]) + [D2]; mu = np.r_[k["zp"][sl], s2]; sd = np.sqrt(np.r_[k["S0"][sl], v2])
ax.fill_between(xs, mu - 1.645 * sd, mu + 1.645 * sd, color=BLUE, alpha=0.14, linewidth=0, label="90% interval of the state before the day is seen"); ax.plot(xs, mu, color=BLUE, linewidth=2, label="state predicted one day ahead")
ax.scatter(days[sl], zz[sl], s=26, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=0.8, label="day's evidence (median implied index shift)")
first = True
for d_, z_ in zip(xs, np.r_[zz[sl], np.nan]):
    if np.isnan(z_): ax.axvspan(d_ - pd.Timedelta(hours=12), d_ + pd.Timedelta(hours=12), color="#d9d8d2", alpha=0.6, linewidth=0, label="no evidence: predicted through" if first else None); first = False
if np.isfinite(Dg["z2"][j]): ax.scatter([D2], [Dg["z2"][j]], s=46, facecolor=SURF, edgecolor=ORANGE, linewidth=1.8, zorder=4, label="what D+2 turned out to be")
ax.axhline(0, color="#c9c8c2", linewidth=0.8); lo_, hi_ = np.nanmin(np.r_[zz[sl], mu - 1.645 * sd]), np.nanmax(np.r_[zz[sl], mu + 1.645 * sd, Dg["z2"][j]]); ax.set_ylim(lo_ - 6.5, hi_ + 2.5)
ax.annotate("D+2: the forecast day", (D2, hi_ + 0.4), ha="right", fontsize=8, color=MUTED)
if np.isfinite(zz[-1]): ax.annotate("D+1: price known,\nloads are forecasts", (days[-1], zz[-1]), xytext=(-46, 22), textcoords="offset points", ha="center", fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color="#c9c8c2"))
ax.set_ylabel("index shift, GW"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=8, columnspacing=1.4, handletextpad=0.5)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title(f"One origin (D+2 = {D2:%d %b %Y}): updated on days with evidence, carried forward with a wider interval on days without", loc="left", fontsize=10, fontweight="bold", color=INK)
axs = []
for c, p in enumerate(("winter", "summer", "autumn")):
    a = fig.add_subplot(gs[1, c], sharey=axs[0] if axs else None); axs.append(a); Ip = np.load(f"{SP}/out/inputs_{p}.npz"); Dp = np.load(f"{SP}/out/ocxs{SFX}_{p}_diag.npz"); r_ = float(TH[p][3]); t_ = pd.to_datetime([str(x)[:10] for x in Ip["cutoffs"]]) + pd.Timedelta(days=1)
    sdp = np.sqrt(Dp["sd2_ts"] ** 2 + r_); a.fill_between(t_, Dp["s2_ts"] - 1.645 * sdp, Dp["s2_ts"] + 1.645 * sdp, color=BLUE, alpha=0.14, linewidth=0, label="90% predictive interval for the day's evidence"); a.plot(t_, Dp["s2_ts"], color=BLUE, linewidth=2, label="state predicted for D+2 at the origin")
    a.scatter(t_, Dp["z2"], s=16, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=0.6, label="what D+2 turned out to be"); a.axhline(0, color="#c9c8c2", linewidth=0.8); ok = np.isfinite(Dp["z2"]) & np.isfinite(Dp["s2_ts"]); cov = 100 * np.mean(np.abs((Dp["z2"] - Dp["s2_ts"]) / sdp)[ok] < 1.645)
    a.set_title(f"{p}: {ok.sum()} origins, {cov:.0f}% inside", loc="left", fontsize=9.5, fontweight="bold", color=INK); a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b")); a.grid(True, axis="y", color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    if c == 0: a.set_ylabel("index shift, GW"); a.legend(frameon=False, loc="upper left", fontsize=7.5)
    else: plt.setp(a.get_yticklabels(), visible=False)
axs[0].set_ylim(-13, 16)
fig.suptitle("Level of the index model as a latent state (Timestral loads). Bottom row: the prediction made at each origin against the outcome.", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.96)
out = f"{SP}/out/state_fig{SFX}.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); print("figure:", out, "| top panel origin", j, str(I["cutoffs"][j])[:10], "missing days in the last 45:", int(np.isnan(zz[-45:]).sum()))
