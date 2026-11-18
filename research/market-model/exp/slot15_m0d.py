"""Quarter-hour prices from the market model (notes 21.13). M0d with the last week carried gives an hourly forecast; within the hour the market is cleared
on quarter-hour demand with the hour's state, and the within-hour shape is scaled by k, estimated ONLY from quarter-hours before each origin
(quarter-hour prices exist from October 2025). Also: which blocks follow a swing within the hour, against from hour to hour. True load given."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import market0d as M
SP = M.SP; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="y"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
R = pd.read_parquet(f"{SP}/out/market0d_rows.parquet"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market0d_post.npz").items() if np.ndim(v) == 1}
full = pd.date_range(R.index.min().tz_convert("UTC").floor("D"), R.index.max().tz_convert("UTC").ceil("D"), freq="1h", tz="UTC"); G = R.copy(); G.index = G.index.tz_convert("UTC"); G = G.reindex(full); hod = np.asarray(G.index.tz_convert("Europe/Berlin").hour); n = len(G)
def state(v, h=2, N=7, fresh=True):
    lag0 = np.where((hod <= 12) & fresh, 24 * h, 24 * (h + 1)); acc = np.zeros(n); cnt = np.zeros(n); idx = np.arange(n)
    for j in range(N):
        src = idx - lag0 - 24 * j; x = np.where(src >= 0, v[np.clip(src, 0, n - 1)], np.nan); m = ~np.isnan(x); acc[m] += x[m]; cnt[m] += 1
    return np.where(cnt >= 4, acc / np.maximum(cnt, 1), np.nan)
KW = dict(eps="eps", eT="eT", eX="eX", eS="eS"); p = G.p.values; tr = (G.tr == True).values; ho = (G.tr == False).values; night = hod < 8; ST = {}
for fresh in (True, False):
    st = {a: state(G[c].values, fresh=fresh) for a, c in KW.items()}; m = ~np.isnan(sum(st.values()) + p); pq = np.full(n, np.nan); pq[m] = M.clear(th, G.D.values[m], G.L.values[m], **{a: v[m] for a, v in st.items()})[0]; ST[fresh] = (st, m, pq)
    print(f"M0d + last 7 days carried, +2 days, hourly MAE ct ({'origin morning used' if fresh else 'data to the day before the origin only: like for like with the own-memory filter'}): fitting {np.mean(np.abs(pq[m & tr] - p[m & tr])) / 10:.2f} (night {np.mean(np.abs(pq[m & tr & night] - p[m & tr & night])) / 10:.2f}), held-out {np.mean(np.abs(pq[m & ho] - p[m & ho])) / 10:.2f} (night {np.mean(np.abs(pq[m & ho & night] - p[m & ho & night])) / 10:.2f})")
st, m, pq = ST[True]
# ---- quarter-hours
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["p_de", "rlreal_de", "g_de_lignite", "g_de_coal", "g_de_gas", "g_de_pump", "g_de_psgen", "g_de_trade", "g_de_other"]]; S.index = S.index.tz_convert("UTC"); hr = S.index.floor("1h"); gi = G.index.get_indexer(hr); okq = gi >= 0; q = lambda v: np.where(okq, np.asarray(v, float)[np.clip(gi, 0, n - 1)], np.nan)
S["D"] = S.rlreal_de - S.g_de_other; S["T"] = S.g_de_lignite + S.g_de_coal + S.g_de_gas; S["St"] = S.g_de_psgen + S.g_de_pump; S["X"] = S.g_de_trade; S["gap"] = S.D - S["T"] - S.X - S.St; L15 = q(G.L.values); m15 = (q(m.astype(float)) > 0.5) & S[["p_de", "D"]].notna().all(1).values
pq15 = np.full(len(S), np.nan); pq15[m15] = M.clear(th, S.D.values[m15], L15[m15], **{a: q(v)[m15] for a, v in st.items()})[0]; S["pq15"] = pq15; S["pqh"] = q(pq); hm = lambda c: S[c].groupby(hr).transform("mean"); S["dm"] = S.pq15 - hm("pq15"); S["dev"] = S.p_de - hm("p_de"); q15 = (S.index >= pd.Timestamp("2025-10-01", tz="Europe/Berlin")) & m15 & S.dev.notna().values & S.dm.notna().values
# which blocks follow a swing: within the hour (deviation from the hour's mean) against from hour to hour (change of the hour's mean)
blk = (("thermal", "T"), ("net imports", "X"), ("storage", "St"), ("unmeasured", "gap")); W = {}; Bt = {}; dD = (S.D - hm("D")).values
Hh = S[["D", "T", "X", "St", "gap"]].groupby(hr).mean(); Hh = Hh[Hh.index >= pd.Timestamp("2025-10-01", tz="UTC")].diff()
for nm, c in blk:
    y = (S[c] - hm(c)).values; k_ = q15 & ~np.isnan(y + dD); W[nm] = float(np.sum(y[k_] * dD[k_]) / np.sum(dD[k_] ** 2)); g = Hh[[c, "D"]].dropna(); Bt[nm] = float(np.sum(g[c] * g.D) / np.sum(g.D ** 2))
print("share of a swing in demand taken by each block, within the hour:", {k: round(v, 2) for k, v in W.items()}, "| from hour to hour:", {k: round(v, 2) for k, v in Bt.items()})
loc = S.index.tz_convert("Europe/Berlin"); day = np.asarray(loc.normalize().tz_localize(None)); slot = np.asarray(loc.hour * 4 + loc.minute // 15); p15 = S.p_de.values; tloc = np.asarray(loc.tz_localize(None))
PER = {"winter": pd.date_range("2025-12-01", periods=30, freq="3D"), "summer": pd.date_range("2026-03-31", periods=50, freq="3D"), "autumn": pd.date_range("2026-09-02", periods=66, freq="1D")}; CAND = {"winter": (0.67, 0.88), "summer": (0.79, 1.28), "autumn": (1.31, 1.63)}; STRUCT = {"winter": (0.86, 0.93), "summer": (0.89, 1.26), "autumn": (1.59, 1.85)}
dm = S.dm.values; dev = S.dev.values; fc = np.full(len(S), np.nan); kser = []
for nm, org in PER.items():
    for o in org:
        past = q15 & (tloc < np.datetime64(o + pd.Timedelta(hours=12))); k_ = float(np.sum(dev[past] * dm[past]) / np.sum(dm[past] ** 2)) if past.sum() > 2000 else 1.0; tg = day == np.datetime64(o + pd.Timedelta(days=2)); fc[tg] = S.pqh.values[tg] + k_ * dm[tg]; kser.append((o, k_, int(past.sum())))
K = pd.DataFrame(kser, columns=["origin", "k", "n"]); print("k (observed within-hour swing per unit of the curves' swing), estimated from the past at each origin: winter", K.k[:30].round(2).describe()[["min", "50%", "max"]].round(2).tolist(), "summer", K.k[30:80].describe()[["min", "50%", "max"]].round(2).tolist(), "autumn", K.k[80:].describe()[["min", "50%", "max"]].round(2).tolist())
res = {}
print("D+2 on quarter-hour slots, ct: hourly repeated | + within-hour shape of the curves (k = 1) | + shape scaled by k from the past | true shape known | structural curve + layer [forecast load] | candidate [forecast load]")
for nm, org in PER.items():
    tg = np.isin(day, (org + pd.Timedelta(days=2)).values)
    for j, (lab, ns) in enumerate((("night", 32), ("first 14 h", 56))):
        k_ = m15 & tg & (slot < ns) & ~np.isnan(fc); e = lambda f: float(np.mean(np.abs(f[k_] - p15[k_])) / 10); res[(nm, lab)] = (e(S.pqh.values), e(S.pqh.values + dm), e(fc), e(S.pqh.values + np.nan_to_num(dev)), STRUCT[nm][j], CAND[nm][j]); print(f"  {nm:7s} {lab:10s} " + " | ".join(f"{v:.3f}" for v in res[(nm, lab)]) + f"   ({int(k_.sum())} slots)")
# ---- figure
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 4, wspace=0.30)
ax = fig.add_subplot(gs[0, 0]); x = np.arange(4); w = 0.36
for j, (d_, col, lab) in enumerate(((Bt, NEUT, "from hour to hour"), (W, BLUE, "within the hour"))):
    v = [d_[b[0]] for b in blk]; ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=col, linewidth=0, label=lab)
    for xi, a in zip(x, v): ax.text(xi + (j - 0.5) * (w + 0.02), a + 0.012 * np.sign(a), f"{a:.2f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=8.5, color=INK)
ax.axhline(0, color=INK, linewidth=0.7); ax.set_xticks(x); ax.set_xticklabels([b[0] for b in blk]); ax.set_ylim(0, 0.52); ax.set_ylabel("share of a swing in demand taken by the block"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ttl(ax, f"Within the hour storage takes more of a demand\nswing ({W['storage']:.0%} vs {Bt['storage']:.0%}), imports less ({W['net imports']:.0%} vs {Bt['net imports']:.0%})")
ax = fig.add_subplot(gs[0, 1]); e_ = np.r_[-40, -20, -12, -7, -3, -1, 1, 3, 7, 12, 20, 40]; cen = 0.5 * (e_[:-1] + e_[1:]); b = np.digitize(dm, e_) - 1; md = np.array([np.median(dev[q15 & (b == i)]) if (q15 & (b == i)).sum() > 150 else np.nan for i in range(len(cen))]); kall = float(np.sum(dev[q15] * dm[q15]) / np.sum(dm[q15] ** 2))
ax.plot(cen / 10, cen / 10, color=NEUT, linewidth=1.6, label="the hourly curves as they are"); ax.plot(cen / 10, kall * cen / 10, color=BLUE, linewidth=1.6, linestyle=(0, (4, 2)), label=f"scaled by {kall:.1f}"); ax.plot(cen / 10, md / 10, "o", color=INK, markersize=4.5, label="observed (median per bin)"); ax.axhline(0, color=INK, linewidth=0.5); ax.set_xlabel("within-hour price swing the curves give, ct/kWh"); ax.set_ylabel("observed within-hour price swing, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax, "both")
ttl(ax, "Small within-hour price swings are twice\nwhat the hourly curves give; large ones less")
ax = fig.add_subplot(gs[0, 2]); ax.plot(K.origin, K.k, "-", color=BLUE, linewidth=1.8); ax.set_ylim(0, max(3.5, K.k.max() * 1.1)); ax.set_ylabel("scale k, from quarter-hours before each origin"); clean(ax, "both"); ax.tick_params(axis="x", labelrotation=30)
ttl(ax, f"One scale is not stable: {K.k.max():.1f} on winter data,\n{K.k.min():.1f} once the summer's larger swings are in")
ax = fig.add_subplot(gs[0, 3]); per = list(PER); x = np.arange(3); w = 0.19; ser = (("hourly forecast repeated", NEUT, 0), ("+ within-hour shape, k from the past", BLUE, 2), ("structural curve + layer (forecast load)", ORANGE, 4), ("candidate (forecast load)", AQUA, 5))
for j, (lab, col, i) in enumerate(ser):
    v = [res[(nm, "night")][i] for nm in per]; ax.bar(x + (j - 1.5) * (w + 0.015), v, w, color=col, linewidth=0, label=lab)
    for xi, a in zip(x, v): ax.text(xi + (j - 1.5) * (w + 0.015), a + 0.03, f"{a:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylim(0, 3.6); ax.set_ylabel("night slots of D+2, price error, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax)
g_ = [res[(nm, "night")][0] - res[(nm, "night")][2] for nm in per]; ttl(ax, f"The quarter-hour shape takes {min(g_):.2f}-{max(g_):.2f} ct off the night\nslots; the gap to the candidate's curve is 0.4-1.1 ct")
fig.suptitle("Quarter-hour prices from the market model: the hour's state, cleared per quarter, within-hour swing scaled from past data only (true load given)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market0d_slots.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0d_slots.png")
