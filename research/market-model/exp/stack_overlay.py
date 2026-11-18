"""A fitted offer stack against its model-free target: German price against thermal output by period. Dots: price against measured lignite + hard coal + gas.
Lines: price against what the stack supplies at that price (must-run + stack blocks 1-4).  usage: stack_overlay.py RUN [RUN2 ...]  (September fit; last period is after the fit)"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import star as S
SP = S.lc.SP; tag = "2026-09-18"; runs = sys.argv[1:]; ix = lambda s: int(np.searchsorted(S.idx.values, np.datetime64(pd.Timestamp(s, tz="Europe/Berlin").tz_convert("UTC").tz_localize(None))))
PER = [("2025-07-01", "2025-12-31", "Jul-Dec 2025"), ("2026-04-01", "2026-07-31", "Apr-Jul 2026"), ("2026-08-01", "2026-09-17", "Aug-17 Sep 2026"), ("2026-09-18", "2026-11-07", "18 Sep-6 Nov 2026 (after the fit)")]
BLUE, ORANGE, INK, MUTED, SURF, AQ = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#8a8983"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True); e = np.arange(2, 34.1, 2.0); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
for r_i, run in enumerate(runs):
    z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")}); nw = z["raw_c_e"].shape[0] + 1; print(run)
    for j, (a_, b_, lab) in enumerate(PER):
        H = S.window(ix(a_), ix(b_)); n = len(H["P"]); d = S.data_of(H, np.full(n, nw - 1), nw); cap, mc, wd, sl, shift, ct = S.build(q, d, np); wd = S.w3(wd, n); P = H["P"][:, 0]; FL = S.floors(q, d, np) * np.ones((n, 1))
        mod = FL.sum(1) + sum(cap[:, 0, k] * sg((P + q["lam"][0, k] * (d["pdmz"][:, 0] - P) - mc[:, 0, k]) / wd[:, 0, k]) for k in range(1, 5)); obs = H["g"].sum(1); ok = np.isfinite(obs) & np.isfinite(P)
        def curve(x):
            m_, p_ = [], []
            for lo_, hi_ in zip(e[:-1], e[1:]):
                kk = ok & (x >= lo_) & (x < hi_)
                if kk.sum() >= 25: m_.append(0.5 * (lo_ + hi_)); p_.append(np.median(P[kk]))
            return np.array(m_), np.array(p_)
        mo, po = curve(obs); mm, pm = curve(mod)
        if r_i == 0: ax[j].plot(mo, po, "o", color=ORANGE, markersize=5, label="observed thermal output")
        ax[j].plot(mm, pm, "-", color=(BLUE if r_i == len(runs) - 1 else AQ), linewidth=2, label=("stack, last version" if r_i == len(runs) - 1 else "stack, plant-block version"))
        common = [m for m in mo if m in set(mm)]; dv = [pm[list(mm).index(m)] - po[list(mo).index(m)] for m in common]
        print(f"   {lab}: gas-fired cost {np.nanmean(1.85 * d['gas'] + 0.37 * (d['eua'] + 70)):.0f}; stack minus observed price at thermal output " + "  ".join(f"{m:.0f}GW {v:+.0f}" for m, v in zip(common, dv)) + f" | thermal total: model mean {mod[ok].mean():.1f}, observed {obs[ok].mean():.1f}")
        ax[j].set_title(lab, loc="left", fontsize=10, fontweight="bold", color=INK); ax[j].set_xlabel("lignite + hard coal + gas output, GW"); ax[j].grid(True, color="#e9e8e3", linewidth=0.6); ax[j].set_axisbelow(True)
        for s_ in ("top", "right"): ax[j].spines[s_].set_visible(False)
ax[0].set_ylabel("median price, EUR/MWh"); ax[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("The offer stack against its target: price at a given thermal output, by period", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); out = f"{SP}/out/stack_overlay_{'_'.join(runs)}.png"; fig.savefig(out, dpi=110); print("figure:", out)
