"""Back of the envelope on strategic withholding (notes 21.36): idle in-the-money capacity by fuel against tightness and against the incentive;
lignite units by operator in two October fortnights."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market1 as M
SP = M.SP; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def fin(ax, t):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.set_title(t, loc="left", fontsize=9.3, fontweight="bold", color=INK)
rng = np.random.default_rng(0); R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}; z = np.load(f"{SP}/out/avail_states.npz"); AV = pd.DataFrame(z["av"][:, :2], index=pd.DatetimeIndex(z["index"]).tz_localize("UTC"), columns=["avs", "avg"]).resample("1h").mean(); AV.index = AV.index.tz_convert(R.index.tz); R = R.join(AV, how="left").dropna(subset=["avs", "avg"])
cl = th["a_l"] + R.co2; cc = th["a_c"] + 0.85 * R.co2; R["solid"] = R.lig + R.coal; R["idle_s"] = (R.avs - R.solid).clip(lower=0); R["idle_g"] = (R.avg - R.gasg).clip(lower=0); R["itm_s"] = R.p > 1.2 * np.maximum(cl, cc); R["itm_g"] = R.p > 1.2 * R.L
sup = lambda p: M.supply(th, p, R.L.values, R.co2.values); R["dpdq"] = 2.0 / np.maximum(sup(R.p.values + 1) - sup(R.p.values - 1), 1e-3)          # EUR/MWh per GW taken out
R["gain_s"] = 0.5 * R.lig * R.dpdq - (R.p - cl); R["gain_g"] = 0.3 * R.gasg * R.dpdq - (R.p - R.L)                                  # largest owner: about half of lignite, about 30% of gas
print(f"hours {len(R)}; published available capacity vs output: lignite + coal available {R.avs.mean():.1f} GW, output {R.solid.mean():.1f}; gas available {R.avg.mean():.1f}, output {R.gasg.mean():.1f}  (gas output above its 'available' in {100 * (R.gasg > R.avg).mean():.0f}% of hours: the gas availability series covers only part of the fleet)")
print(f"price rise per GW taken out (EUR/MWh): 5-50-95% {R.dpdq.quantile([.05, .5, .95]).round(1).tolist()}; in-the-money hours: solid {100 * R.itm_s.mean():.0f}%, gas {100 * R.itm_g.mean():.0f}%; hours where holding back 1 GW of lignite would pay the largest owner: {100 * (R.gain_s > 0).mean():.0f}% (of in-the-money hours {100 * (R.gain_s[R.itm_s] > 0).mean():.0f}%)")
fig, axs = plt.subplots(1, 4, figsize=(19, 4.9), gridspec_kw=dict(wspace=0.27)); eD = np.r_[0, 10, 15, 20, 25, 30, 35, 40, 50, 70]; cen = 0.5 * (eD[:-1] + eD[1:])
def bm(x, y, e, minn=60): b = np.digitize(x, e) - 1; return np.array([np.mean(y[b == i]) if (b == i).sum() >= minn else np.nan for i in range(len(e) - 1)])
def envl(x, y, e, n=150): A = np.array([bm(x, np.roll(y, int(k) * 168), e) for k in rng.integers(4, len(y) // 168 - 4, n)]); return np.nanpercentile(A, [5, 95], axis=0)
ax = axs[0]; g = R[R.itm_s]
for lab, m_, col in (("fitting period", g.tr.values, NEUT), ("held-out year", ~g.tr.values, ORANGE)): ax.plot(cen, bm(g.D.values[m_], g.idle_s.values[m_], eD), "-o", color=col, linewidth=1.8, markersize=4, label=lab)
ax.set_xlabel("demand left for thermal, imports, storage, GW"); ax.set_ylabel("available, in the money, not running, GW"); ax.legend(frameon=False, fontsize=8); a_ = bm(g.D.values, g.idle_s.values, eD); fin(ax, f"Lignite + coal: idle in-the-money capacity falls\nas the market tightens ({np.nanmax(a_):.1f} to {a_[~np.isnan(a_)][-1]:.1f} GW)")
ax = axs[1]; g = R[R.itm_s]; er = np.r_[1.2, 1.4, 1.6, 1.9, 2.3, 3.0, 6.0]; cr = 0.5 * (er[:-1] + er[1:]); rc = (g.p / np.maximum(cl, cc)[g.index]).values; ax.plot(cr, bm(rc, g.idle_s.values, er, 40), "-o", color=BLUE, linewidth=1.8, markersize=4, label="idle in-the-money capacity, GW"); ax2v = bm(rc, (0.5 * g.lig * g.dpdq).values, er, 40); ax3v = bm(rc, (g.p - cl[g.index]).values, er, 40)
ax.set_xlabel("price / cost of lignite and coal (the dearer of the two)"); ax.set_ylabel("available, in the money, not running, GW"); fin(ax, f"Holding back 1 GW earns its owner about {np.nanmean(ax2v):.0f} EUR/MWh\non the rest and gives up {np.nanmin(ax3v):.0f} or more: it does not pay")
ax = axs[2]; g = R[R.itm_g]
for lab, m_, col in (("fitting period", g.tr.values, NEUT), ("held-out year", ~g.tr.values, ORANGE)): ax.plot(cen, bm(g.D.values[m_], (g.gasg / 33.0).values[m_], eD), "-o", color=col, linewidth=1.8, markersize=4, label=lab)
ax.set_xlabel("demand left for thermal, imports and storage, GW"); ax.set_ylabel("gas output / 33 GW installed, hours with price > 1.2 x cost"); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 1); a_ = bm(g.D.values, (g.gasg / 33.0).values, eD); fin(ax, f"Gas in the money: {100 * np.nanmin(a_):.0f}-{100 * np.nanmax(a_):.0f}% of installed capacity\nruns, more as the market tightens")
ax = axs[3]; OP = {"LEAG": ("Boxberg", "Jänschwalde", "Schwarze Pumpe", "Lippendorf"), "RWE": ("Neurath", "Niederaußem", "Weisweiler"), "other": ("Schkopau",)}; rows = []
for yr, f in ((2025, "units_lignite_2025-10-06"), (2026, "units_lignite_2026-10-05")):
    U = pd.read_parquet(f"{SP}/out/{f}.parquet"); U.index = U.index.tz_convert(R.index.tz); J = U.join(R[["p", "co2"]], how="inner"); itm = J.p > 1.2 * (th["a_l"] + J.co2)
    for u in U.columns:
        op = next((k for k, v in OP.items() if any(n in u for n in v)), "other"); x = J[u][itm]; rows.append((yr, op, u, float(x.mean()), float((x < 0.6).mean()), float((J[u] > 0.05).mean()), int(itm.sum())))
T = pd.DataFrame(rows, columns=["yr", "op", "unit", "lf", "part", "on", "n"]); print(T.groupby(["yr", "op"])[["lf", "part", "on"]].mean().round(2).to_string(), "| in-the-money hours:", T.groupby("yr").n.first().to_dict())
for j, (op, col) in enumerate((("LEAG", BLUE), ("RWE", ORANGE), ("other", NEUT))):
    for k, yr in enumerate((2025, 2026)):
        t = T[(T.op == op) & (T.yr == yr)]; ax.scatter(np.full(len(t), j + (k - 0.5) * 0.3) + rng.uniform(-0.05, 0.05, len(t)), t.lf, s=30, color=col, alpha=0.4 if yr == 2025 else 0.95, linewidths=0, label=f"{op} {yr}")
ax.set_xticks(range(3)); ax.set_xticklabels(["LEAG", "RWE", "other"]); ax.set_ylim(0, 1.05); ax.set_ylabel("load factor of each lignite unit, in-the-money hours"); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left"); m25, m26 = T[T.yr == 2025].lf.mean(), T[T.yr == 2026].lf.mean(); fin(ax, f"Lignite units in the money, two October\nfortnights: load factor {m25:.2f} (2025), {m26:.2f} (2026)")
fig.suptitle("Back of the envelope: can strategic withholding be seen? Capacity that is available, in the money and not running, against tightness and against the incentive", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04); fig.savefig(f"{SP}/out/withhold.png", dpi=100, bbox_inches="tight"); print("figure ok")
