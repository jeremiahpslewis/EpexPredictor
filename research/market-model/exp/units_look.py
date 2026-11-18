"""Lignite unit by unit (notes 21.41): load factors in in-the-money hours by operator and month, off against part-load, part-load by hour of day,
and how much of the merit-order model's lignite shock the units' states account for. Uses whatever months are downloaded."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def fin(ax, t):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.set_title(t, loc="left", fontsize=9.3, fontweight="bold", color=INK)
P = []
for area in ("DE_50HZ", "DE_AMPRION"):
    fs = sorted(glob.glob(f"{SP}/entsoe/units_all/{area}_lignite_*.parquet")); x = pd.concat([pd.read_parquet(f) for f in fs]).sort_index(); x = x[~x.index.duplicated()]; x.columns = [c.split(" | ")[0] for c in x.columns]; x = x.T.groupby(level=0).sum(min_count=1).T; P.append(x)
U = (P[0].join(P[1], how="outer") / 1000.0); months = sorted({i.strftime("%Y-%m") for i in U.index}); print(f"{len(U)} hours on {U.index.normalize().nunique()} days, {U.shape[1]} units, months {months[0]} .. {months[-1]} ({len(months)})")
OP = lambda u: "LEAG" if any(k in u for k in ("Boxberg", "Jänschwalde", "Schwarze Pumpe", "Lippendorf")) else ("RWE" if any(k in u for k in ("Neurath", "Niederaußem", "Weisweiler")) else "other")
cap = U.quantile(0.995); live = cap[cap > 0.05].index; U = U[live]; cap = cap[live]; R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}; U.index = U.index.tz_convert(R.index.tz); J = U.join(R[["p", "co2", "lig", "d_l", "D"]], how="inner"); cl = th["a_l"] + J.co2; itm = (J.p > 1.2 * cl).values
print("capacity seen per operator GW:", {o: round(float(cap[[u for u in live if OP(u) == o]].sum()), 1) for o in ("LEAG", "RWE", "other")}, "| units' sum against the fuel-level lignite output: ratio", round(float(J[live].sum(1).mean() / J.lig.mean()), 3), "correlation", round(float(np.corrcoef(J[live].sum(1).fillna(0), J.lig)[0, 1]), 3), "| in-the-money hours", int(itm.sum()))
lf = J[live] / cap; off = (lf < 0.05) | lf.isna(); part = (lf >= 0.05) & (lf < 0.85); mo = J.index.strftime("%Y-%m"); fig, axs = plt.subplots(1, 4, figsize=(19.5, 4.9), gridspec_kw=dict(wspace=0.27)); COL = {"LEAG": BLUE, "RWE": ORANGE, "other": NEUT}
ax = axs[0]; T = {}
for o in ("LEAG", "RWE"):
    us = [u for u in live if OP(u) == o and off[u].values[itm].mean() < 0.9]; g = pd.DataFrame({"m": mo[itm], "lf": (np.nansum(J[us].values[itm], 1) / cap[us].sum())}); s = g.groupby("m").lf.mean(); n = g.groupby("m").lf.size(); s = s[n >= 24]; T[o] = s; ax.plot(pd.to_datetime(s.index), s.values, "-o", color=COL[o], linewidth=1.8, markersize=3.5, label=f"{o} ({cap[us].sum():.1f} GW seen)")
ax.set_ylim(0, 1); ax.set_ylabel("output / capacity of the operator's lignite units,\nhours with price > 1.2 x cost"); ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.tick_params(axis="x", labelrotation=30)
y = lambda o, yr: float(T[o][[i for i in T[o].index if i.startswith(str(yr))]].mean()) if any(i.startswith(str(yr)) for i in T[o].index) else np.nan; fin(ax, "In the money, by operator and month: load factor\n" + "; ".join(f"{o} " + " / ".join(f"{y(o, yr):.2f}" for yr in (2024, 2025, 2026) if not np.isnan(y(o, yr))) for o in ("LEAG", "RWE")))
ax = axs[1]; offgw = (off.values * cap.values[None]).sum(1); partgw = (np.where(part.values, (1 - lf.fillna(0).values), 0.0) * cap.values[None]).sum(1)
# published outage messages, per station: unavailable MW by hour, planned and unplanned (not cancelled; per resource the largest active message)
O = pd.read_parquet(f"{SP}/entsoe/unavail_gen_DE_LU.parquet"); O = O[O.plant_type.astype(str).str.contains("ignite") & (O.docstatus.astype(str) != "Cancelled")].copy(); O["un"] = (O.nominal_power.astype(float) - O.avail_qty.astype(float)).clip(lower=0) / 1000.0; STN = lambda u: next(k for k in ("Boxberg", "Jänschwalde", "Schwarze Pumpe", "Lippendorf", "Neurath", "Niederaußem", "Weisweiler", "Schkopau") if k in u)
hrs = J.index; PUB = {}
for (stn, kind), g in O.groupby([O.production_resource_name.map(STN), O.businesstype]):
    tot = np.zeros(len(hrs))
    for rid, gg in g.groupby("production_resource_id"):
        m = np.zeros(len(hrs))
        for st_, en_, un_ in zip(gg.start, gg.end, gg.un): i0, i1 = hrs.searchsorted(st_), hrs.searchsorted(en_); m[i0:i1] = np.maximum(m[i0:i1], un_)
        tot += m
    PUB[(stn, kind)] = tot
cat = {"planned": np.zeros(len(hrs)), "unplanned": np.zeros(len(hrs)), "none": np.zeros(len(hrs))}; byop = {o: {k: np.zeros(len(hrs)) for k in cat} for o in ("LEAG", "RWE", "other")}
for stn in ("Boxberg", "Jänschwalde", "Schwarze Pumpe", "Lippendorf", "Neurath", "Niederaußem", "Weisweiler", "Schkopau"):
    us = [u for u in live if stn in u and cap[u] > 0 and off[u].values[itm].mean() < 0.9]   # units closed for good are left out
    if not us: continue
    gap = np.clip(cap[us].sum() - J[us].fillna(0).sum(1).values, 0, None); pl = np.minimum(gap, PUB.get((stn, "Planned maintenance"), 0.0)); upl = np.minimum(gap - pl, PUB.get((stn, "Unplanned outage"), 0.0)); o = OP(us[0])
    for k, v_ in (("planned", pl), ("unplanned", upl), ("none", gap - pl - upl)): cat[k] += v_; byop[o][k] += v_
g = pd.DataFrame({"m": mo[itm], **{k: v_[itm] for k, v_ in cat.items()}}).groupby("m").mean(); x = pd.to_datetime(g.index); bot = np.zeros(len(g))
for k, col, lb in (("planned", NEUT, "published planned outage"), ("unplanned", BLUE, "published unplanned outage"), ("none", ORANGE, "no outage message")): ax.bar(x, g[k], 20, bottom=bot, color=col, linewidth=0, label=lb); bot += g[k].values
ax.set_ylabel("lignite capacity not producing, in-the-money hours, GW"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.tick_params(axis="x", labelrotation=30); tot_ = g.sum(1).mean(); fin(ax, f"Not producing while in the money: {tot_:.1f} GW, of which\n{g.none.mean():.1f} GW has no outage message")
print("not producing while in the money, GW (planned / unplanned / no message):", {o: tuple(round(float(v_[itm].mean()), 2) for v_ in byop[o].values()) for o in byop}); yrs = J.index.year.values
for yr in sorted(set(yrs)): print("  ", yr, {o: round(float(byop[o]["none"][itm & (yrs == yr)].mean()), 2) for o in byop}, "no-message GW; in-the-money hours", int((itm & (yrs == yr)).sum()))
dayitm = (J.p.groupby(J.index.normalize()).transform("mean") > 1.2 * cl).values; print(f"no-message idle GW: all in-the-money hours {cat['none'][itm].mean():.2f}; hours of days whose MEAN price is 20% above cost {cat['none'][dayitm].mean():.2f} ({int(dayitm.sum())} hours); in-the-money hours of other days {cat['none'][itm & ~dayitm].mean():.2f}"); summ = np.isin(J.index.month.values, (4, 5, 6, 7, 8, 9)); print(f"   April-September {cat['none'][itm & summ].mean():.2f} GW, October-March {cat['none'][itm & ~summ].mean():.2f} GW; whole-day-in-the-money, by year:", {int(yr): round(float(cat['none'][dayitm & (J.index.year.values == yr)].mean()), 2) for yr in (2024, 2025, 2026)}, {o: {int(yr): round(float(byop[o]['none'][dayitm & (J.index.year.values == yr)].mean()), 2) for yr in (2024, 2025, 2026)} for o in ('LEAG', 'RWE')})
nomsg = cat["none"]; tight = J.D.values; eD = np.r_[0, 20, 25, 30, 35, 40, 50, 70]; b_ = np.digitize(tight, eD) - 1; print("no-message idle GW in the money by demand band (tighter to the right):", [round(float(nomsg[itm & (b_ == i)].mean()), 2) if (itm & (b_ == i)).sum() > 40 else None for i in range(len(eD) - 1)])
ax = axs[2]; hod = J.index.hour.values
for o in ("LEAG", "RWE"):
    us = [u for u in live if OP(u) == o]; run = ~off[us].values; pl = np.where(run, lf[us].fillna(0).values, np.nan); v = np.array([np.nanmean(pl[itm & (hod == h)]) for h in range(24)]); ax.plot(range(24), v, "-o", color=COL[o], linewidth=1.8, markersize=3.5, label=o)
ax.set_ylim(0, 1); ax.set_xlabel("hour of day"); ax.set_ylabel("load factor of RUNNING units, in-the-money hours"); ax.legend(frameon=False, fontsize=8, loc="lower left"); fin(ax, "Units that run, run at about 0.9 of capacity all\nday: the idle capacity is units that are off")
ax = axs[3]; Dd = pd.DataFrame({"d_l": J.d_l.values, "off": offgw, "part": partgw}, index=J.index).groupby(J.index.normalize()).mean(); X = np.column_stack([Dd.off, Dd.part, np.ones(len(Dd))]); be = np.linalg.lstsq(X, Dd.d_l.values, rcond=None)[0]; fit = X @ be; r2 = 1 - np.var(Dd.d_l.values - fit) / np.var(Dd.d_l.values); r2o = np.corrcoef(Dd.off, Dd.d_l)[0, 1] ** 2; r2p = np.corrcoef(Dd.part, Dd.d_l)[0, 1] ** 2
ax.scatter(fit, Dd.d_l, s=10, color=BLUE, alpha=0.5, linewidths=0); lim = [min(fit.min(), Dd.d_l.min()), max(fit.max(), Dd.d_l.max())]; ax.plot(lim, lim, color=INK, linewidth=0.8); ax.set_xlabel("explained by GW off and GW held back in running units"); ax.set_ylabel("merit-order model's daily lignite shock, GW"); fin(ax, f"Units' states explain {100 * r2:.0f}% of the daily lignite shock\n(off alone {100 * r2o:.0f}%, held back alone {100 * r2p:.0f}%)")
print(f"daily lignite shock: {be[0]:+.2f} GW per GW off, {be[1]:+.2f} GW per GW held back; R2 {r2:.2f} (off {r2o:.2f}, part {r2p:.2f}); days {len(Dd)}")
print("by unit, in-the-money hours: load factor and share of hours off / part-load:"); tb = pd.DataFrame({"op": [OP(u) for u in live], "cap": cap.values.round(2), "lf": np.nanmean(np.where(itm[:, None], lf.fillna(0).values, np.nan), 0).round(2), "off": off.values[itm].mean(0).round(2), "part": part.values[itm].mean(0).round(2)}, index=live); print(tb.sort_values(["op", "lf"]).to_string())
fig.suptitle(f"Lignite unit by unit ({months[0]} to {months[-1]}, every second day): what is not producing while in the money, and what it has to do with the model's lignite shock", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04); fig.savefig(f"{SP}/out/units_lignite.png", dpi=100, bbox_inches="tight"); print("figure ok")
