"""The last gas spike, 2021-2023: did German lignite and hard coal throttle at prices above their own cost when gas was far more expensive?"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
H = pd.read_parquet(f"{SP}/out/hist2123.parquet").loc["2021-01-01":"2023-12-31"]; loc = H.index.tz_convert("Europe/Berlin"); day = loc.normalize(); H["pbar"] = H["p"].groupby(day).transform("mean").values
PER = [("2021-01", "2021-06"), ("2021-07", "2021-12"), ("2022-01", "2022-06"), ("2022-07", "2022-12"), ("2023-01", "2023-06"), ("2023-07", "2023-12")]; LAB = ["Jan-Jun 2021", "Jul-Dec 2021", "Jan-Jun 2022", "Jul-Dec 2022", "Jan-Jun 2023", "Jul-Dec 2023"]
ym = loc.strftime("%Y-%m"); print("per half-year: mean price, gas, coal (EUR/MWh thermal), carbon; cost per MWh of electricity: lignite (carbon + 12), hard coal (0.85 carbon + coal / 0.40), combined cycle (1.85 gas + 0.37 carbon)")
for (lo, hi), lab in zip(PER, LAB):
    k = (ym >= lo) & (ym <= hi); print(f"  {lab}: price {H['p'][k].mean():4.0f}  gas {H['gas'][k].mean():4.0f}  coal {H['coal'][k].mean():3.0f}  carbon {H['eua'][k].mean():3.0f} | lignite {H['eua'][k].mean() + 12:4.0f}  hard coal {0.85 * H['eua'][k].mean() + H['coal'][k].mean() / 0.4:4.0f}  combined cycle {1.85 * H['gas'][k].mean() + 0.37 * H['eua'][k].mean():4.0f} | output lignite {H['lignite'][k].mean():.1f} hard coal {H['hcoal'][k].mean():.1f} gas {H['gas_out'][k].mean() if 'gas_out' in H else np.nan:.1f}")
FU = {"lignite": ("lignite", lambda h: h["eua"] * 1.0 + 12.0), "hard coal": ("hcoal", lambda h: 0.85 * h["eua"] + h["coal"] / 0.40), "hard coal, coal price ignored": ("hcoal", lambda h: 0.85 * h["eua"] + 26.0)}
SEQ = ["#b9d3f2", "#8fb9ea", "#5f9be0", "#2a78d6", "#1b56a3", "#0e3568"]; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))
for j, (name, (col, costf)) in enumerate(FU.items()):
    g = H[col]; avail = g.rolling(14 * 24, min_periods=7 * 24).max().shift(24); floor = g.rolling(60 * 24, min_periods=20 * 24).quantile(0.02).shift(24); u = ((g - floor) / (avail - floor)).clip(-0.2, 1.3)
    m = H["p"] - costf(H); md = H["pbar"] - costf(H); pm1 = H["p"].shift(1) - costf(H); pp1 = H["p"].shift(-1) - costf(H); flat = ((pm1 - m).abs() < 15) & ((pp1 - m).abs() < 15)
    print(f"\n{name}: median utilisation by margin over cost (hour price), flat stretches only (neighbouring hours within 15 EUR/MWh); count in brackets")
    bands = [(-1e9, -30), (-30, -10), (-10, 10), (10, 30), (30, 60), (60, 120), (120, 1e9)]; print("                  " + "".join(f"{('<-30' if a < -1e8 else ('>120' if b > 1e8 else f'{a}..{b}')):>14s}" for a, b in bands))
    for i, ((lo, hi), lab) in enumerate(zip(PER, LAB)):
        k = (ym >= lo) & (ym <= hi) & u.notna().values & m.notna().values; row = ""
        for a, b in bands:
            kk = k & flat.values & (m.values >= a) & (m.values < b); row += f"{np.median(u.values[kk]):7.2f} ({kk.sum():4d})" if kk.sum() >= 20 else "       .      "
        print(f"   {lab:14s} {row}")
        xs = m.values[k]; us = u.values[k]; e = np.quantile(xs, np.linspace(0.02, 0.98, 15)); mid, med = [], []
        for a_, b_ in zip(e[:-1], e[1:]):
            kk = (xs >= a_) & (xs < b_)
            if kk.sum() > 20: mid.append(np.median(xs[kk])); med.append(np.median(us[kk]))
        ax[j].plot(mid, med, "-o", color=SEQ[i], linewidth=2, markersize=3.5, label=lab)
    a = ax[j]; a.set_ylim(-0.1, 1.15); a.set_xlim(-150, 300); a.set_xlabel("hour price minus the plant type's own cost, EUR/MWh"); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True); a.set_title(f"German {name}", loc="left", fontsize=10, fontweight="bold", color=INK)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
ax[0].set_ylabel("median utilisation of available capacity"); ax[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.suptitle("The 2021-2023 gas spike: utilisation of lignite and hard coal against the margin over their own cost", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{SP}/out/hist_look.png", dpi=110); print("figure:", f"{SP}/out/hist_look.png")
