"""Three checks. (1) price-free: each fuel's output above its floor against total thermal output above floors, same season in three years.
(2) where is the German price set in the late-2026 hours with idle lignite: equal to which neighbours' prices?
(3) the September fit with the latent weekly cost level set to zero: what quantities does that state absorb?"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet"); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
H = D.resample("1h").mean(); loc = H.index.tz_convert("Europe/Berlin"); day = loc.normalize()
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); H["eua"] = ed.reindex(day.tz_localize(None)).values; H["pbar"] = H["p_de"].groupby(day).transform("mean").values
FU = [("lignite", "g_de_lignite"), ("hard coal", "g_de_coal"), ("gas", "g_de_gas")]
# ---------- (1)
YRS = [("2024", "Aug-Nov 2024"), ("2025", "Aug-Nov 2025"), ("2026", "Aug-Nov 2026")]; md = loc.strftime("%m-%d"); seas = (md >= "08-01") & (md <= "11-09")
SEQ = ["#9cc0ee", "#2a78d6", "#0e3568"]; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3)); print("(1) price-free: GW above floor at a given total thermal output above floors [median]; floors = 2% quantile of the season")
edges = np.arange(0, 30.1, 2.5); tab = {}
for i, (y, lab) in enumerate(YRS):
    k = seas & (loc.year == int(y)); fl = {c: np.nanquantile(H[c][k], 0.02) for _, c in FU}; tot = sum(H[c][k] - fl[c] for _, c in FU).values
    print(f"  {lab}: floors lignite {fl['g_de_lignite']:.1f}, hard coal {fl['g_de_coal']:.1f}, gas {fl['g_de_gas']:.1f}; season maxima {H['g_de_lignite'][k].max():.1f} / {H['g_de_coal'][k].max():.1f} / {H['g_de_gas'][k].max():.1f}; mean price {H['p_de'][k].mean():.0f}, gas {H['gas'][k].mean():.0f}, carbon {H['eua'][k].mean():.0f}")
    for j, (name, c) in enumerate(FU):
        y_ = (H[c][k] - fl[c]).values; mid, med = [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            kk = (tot >= a_) & (tot < b_)
            if kk.sum() >= 20: mid.append(0.5 * (a_ + b_)); med.append(np.median(y_[kk]))
        tab[(name, lab)] = dict(zip(mid, med)); ax[j].plot(mid, med, "-o", color=SEQ[i], linewidth=2, markersize=4, label=lab)
    # price at a given total thermal output
    pr = H["p_de"][k].values; mid, med = [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        kk = (tot >= a_) & (tot < b_)
        if kk.sum() >= 20: mid.append(0.5 * (a_ + b_)); med.append(np.median(pr[kk]))
    tab[("price", lab)] = dict(zip(mid, med)); ax[3].plot(mid, med, "-o", color=SEQ[i], linewidth=2, markersize=4, label=lab)
for name in ("lignite", "hard coal", "gas", "price"):
    print(f"  {name:9s} at total thermal above floors (GW): " + "  ".join(f"{m:4.1f}:" + "/".join(f"{tab[(name, lab)].get(m, np.nan):.1f}" if name != "price" else f"{tab[(name, lab)].get(m, np.nan):.0f}" for _, lab in YRS) for m in (1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75, 21.25, 23.75)))
for j, t in enumerate(("Lignite", "Hard coal", "Gas", "Price")):
    a = ax[j]; a.set_title(t + (" output above its floor" if j < 3 else " at that thermal output"), loc="left", fontsize=10, fontweight="bold", color=INK); a.set_xlabel("lignite + hard coal + gas above their floors, GW"); a.set_ylabel("GW" if j < 3 else "EUR/MWh"); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
ax[0].legend(frameon=False, loc="upper left")
fig.suptitle("No prices involved in the first three panels: which plants deliver a given amount of thermal output, same season in three years", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{SP}/out/drift_look4.png", dpi=110); print("figure:", f"{SP}/out/drift_look4.png")
# ---------- (2)
ym = loc.strftime("%Y-%m"); A = (ym >= "2024-07") & (ym <= "2026-03"); B = (ym >= "2026-08")
mh = (H["p_de"] - H["eua"]).values; mdy = (H["pbar"] - H["eua"]).values; NB = ["fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]
print("\n(2) share of hours in which the German price equals the neighbour's (within 0.5 EUR/MWh), and the median neighbour-minus-German price")
for lab, k in (("early, all hours", A), ("late, all hours", B), ("early, hour margin 30-60 & day margin >= 40", A & (mh >= 30) & (mh < 60) & (mdy >= 40)), ("late,  hour margin 30-60 & day margin >= 40", B & (mh >= 30) & (mh < 60) & (mdy >= 40)), ("late, 9-17h", B & (loc.hour >= 9) & (loc.hour < 17)), ("late, 18-22h", B & (loc.hour >= 18) & (loc.hour < 22))):
    k = k & H["p_de"].notna().values
    print(f"  {lab:45s} n {k.sum():5d}: " + "  ".join(f"{z} {100 * np.nanmean(np.abs(H['p_' + z][k] - H['p_de'][k]) < 0.5):3.0f}% ({np.nanmedian(H['p_' + z][k] - H['p_de'][k]):+4.0f})" for z in NB) + f" | equal to none: {100 * np.mean(np.all([np.abs(H['p_' + z][k] - H['p_de'][k]).fillna(99) >= 0.5 for z in NB], 0)):3.0f}% | net imports {np.nanmedian(H['x_sum'][k]):+.1f} GW, RL {np.nanmedian(H['rl_de'][k]):.1f}")
# ---------- (3)
import star as S
tag, period = "2026-09-18", "autumn"; z = np.load(f"{SP}/out/star_v10/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")})
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96; lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4
Hh = S.window(lo, d1); week = (Hh["day"] - Hh["day"].min()) // 7; d = S.data_of(Hh, week, int(week.max()) + 1); P = Hh["P"]; ok = d["ok"][:, 0]; sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
q0 = dict(q); q0["c_e"] = np.zeros_like(q["c_e"]); mo = (Hh["day"] - Hh["day"].min()) // 60
print("\n(3) September fit, German output by fuel at the observed prices: mean (realised - model) GW per two-month block; first with the fitted weekly cost level, then with it set to zero")
t0 = pd.Timestamp(S.idx[lo]).tz_convert("Europe/Berlin"); print("   blocks start: " + "  ".join((t0 + pd.Timedelta(days=60 * k)).strftime("%y-%m-%d") for k in range(int(mo.max()) + 1)))
for lab_, qq in (("fitted cost level", q), ("cost level = 0   ", q0)):
    cap, mc, wd, sl, shift, ct = S.build(qq, d, np)
    print(f"   {lab_}: level per block " + "  ".join(f"{ct[mo == k].mean():+5.0f}" for k in range(int(mo.max()) + 1)))
    for name, ks, gi in (("lignite", [1], 0), ("hard coal", [2], 1), ("gas", [3, 4], 2)):
        mod = q["floor"][gi] + sum(cap[:, 0, k] * sg((P[:, 0] + q["lam"][0, k] * (d["pdmz"][:, 0] - P[:, 0]) - mc[:, 0, k]) / wd[0, k]) for k in ks); obs = Hh["g"][:, gi]; mm = np.isfinite(obs) & ok
        print(f"      {name:9s} " + "  ".join(f"{(obs - mod)[mm & (mo == k)].mean():+5.1f}" for k in range(int(mo.max()) + 1)))
print("   German blocks (fitted): alpha", np.round(q["alpha"][0], 0), "beta", np.round(q["beta"][0], 2), "capf", np.round(q["capf"][0], 1), "kmeas", np.round(q["kmeas"][0], 2), "wd", np.round(q["wd"][0], 0), "floor", np.round(q["floor"], 1))
print("   gas and carbon per block: " + "  ".join(f"{Hh['gas'][mo == k].mean():.0f}/{Hh['eua'][mo == k].mean() + 70:.0f}" for k in range(int(mo.max()) + 1)))
