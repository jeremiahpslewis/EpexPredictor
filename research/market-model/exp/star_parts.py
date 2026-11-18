"""The two new parts of version 11 against the data. Top: output of each German fuel in the hours when running does not pay, against the day's temperature
(dots: realised, line: the model's must-run plus what its block still delivers at those prices). Bottom: scheduled exchange against the price difference on
calm and on windy days in Germany (dots) with the model's curve at wind -1 sd and +1 sd.  usage: star_parts.py RUN PERIOD TAG"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import star as S
run, period, tag = sys.argv[1], sys.argv[2], sys.argv[3]; SP = S.lc.SP
z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")})
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96; lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4
H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1); P = H["P"]; n = len(P); ok = d["ok"][:, 0]
cap, mc, wd, sl, shift, ct = S.build(q, d, np); wd = S.w3(wd, n); FL = S.floors(q, d, np) * np.ones((n, 1)); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
temp = 15.0 - d["hdd"][:, 0]                                                    # effective temperature below 15 C (heating degrees turned back); above 15 it is flat
F = S.F["DE"]; tq = F[[c for c in F.columns if c.startswith("temp_")]].mean(1).to_numpy(float); tday = pd.Series(tq).groupby(S.day_all).transform("mean").to_numpy(float); temp = tday[lo:lo + 4 * n:4]
BLUE, ORANGE, INK, MUTED, SURF, LO, HI = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#8fb9ea", "#0e3568"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(2, 3, figsize=(14, 8)); eT = np.arange(-4, 25, 2.0)
for j, (lab, ks, gi) in enumerate((("lignite", [1], 0), ("hard coal", [2], 1), ("gas", [3, 4], 2))):
    pef = lambda k: P[:, 0] + q["lam"][0, k] * (d["pdmz"][:, 0] - P[:, 0]); blk = sum(cap[:, 0, k] * sg((pef(k) - mc[:, 0, k]) / wd[:, 0, k]) for k in ks); mod = FL[:, gi] + blk; obs = H["g"][:, gi]
    cold = ok & np.isfinite(obs) & ((pef(ks[0]) - mc[:, 0, ks[0]]) < -30); a = ax[0, j]; mid, mo_, mm_, mf_ = [], [], [], []
    for t0, t1 in zip(eT[:-1], eT[1:]):
        kk = cold & (temp >= t0) & (temp < t1)
        if kk.sum() >= 25: mid.append(0.5 * (t0 + t1)); mo_.append(np.median(obs[kk])); mm_.append(np.median(mod[kk])); mf_.append(np.median(FL[kk, gi]))
    a.plot(mid, mm_, color=BLUE, linewidth=2, label="model: must-run plus block at those prices"); a.plot(mid, mf_, color=BLUE, linewidth=1.2, linestyle="--", label="model: must-run alone"); a.plot(mid, mo_, "o", color=ORANGE, markersize=5, label="realised")
    a.set_title(f"German {lab}: output when running does not pay", loc="left", fontsize=10, fontweight="bold", color=INK); a.set_xlabel("daily mean temperature, deg C"); a.set_ylim(0, None); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    if j == 0: a.set_ylabel("median output, GW"); a.legend(frameon=False, loc="lower left", fontsize=8)
    print(f"{lab}: hours where it does not pay {cold.sum()}; realised - model by temperature bin: " + "  ".join(f"{m:.0f}C {o - mm:+.1f}" for m, o, mm in zip(mid, mo_, mm_)))
wind = d["wx"][:, 0]; edges = np.r_[-200, -60, -30, -15, -7, -3, -1, 1, 3, 7, 15, 30, 60, 200]; g = np.linspace(-90, 90, 200)
for j, zn in enumerate(("dk", "cz", "ch")):
    k = S.ZN.index(zn) - 1; a = ax[1, j]; dp = P[:, k + 1] - P[:, 0]; x = np.where(d["xok"][:, k], d["x"][:, k], np.nan)
    for lab, sel, col, wv in (("calm days (wind below -0.5 sd)", wind < -0.5, LO, -1.0), ("windy days (wind above +0.5 sd)", wind > 0.5, HI, 1.0)):
        c_, v_ = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            kk = sel & np.isfinite(x) & np.isfinite(dp) & (dp >= e0) & (dp < e1)
            if kk.sum() >= 25: c_.append(np.median(dp[kk])); v_.append(np.median(x[kk]))
        a.plot(np.array(c_) / 10, v_, "o", color=col, markersize=5, label=f"scheduled, {lab}"); wx = np.tile(np.array([[wv, 0.0]]), (len(g), 1)); a.plot(g / 10, S.exch(q, np.tile(g[:, None], (1, S.Z - 1)), np, wx)[:, k], color=col, linewidth=2, label=f"model, wind {wv:+.0f} sd")
    a.set_title(f"Exports to {zn.upper()} against the price difference", loc="left", fontsize=10, fontweight="bold", color=INK); a.set_xlabel(f"{zn.upper()} price minus German price, ct/kWh"); a.set_xlim(-9, 9); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    if j == 0: a.set_ylabel("median German exports, GW"); a.legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle(f"The two new parts against the data (fit to the 540 days before {tag})", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); out = f"{SP}/out/star_{run}_{tag}_parts.png"; fig.savefig(out, dpi=110); print("figure:", out)
