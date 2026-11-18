"""The offer stack's target, free of any model: German price against measured thermal output (lignite + hard coal + gas), by period, in EUR/MWh and
relative to gas-fired cost; and with solid-fuel cost shown. Optionally overlays a fitted stack: stack_target.py [RUN PERIOD TAG]"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); D["eua"] = ed.reindex(loc.normalize().tz_localize(None)).values
D["L"] = 1.85 * D["gas"] + 0.37 * D["eua"]; D["th"] = D["g_de_lignite"] + D["g_de_coal"] + D["g_de_gas"]; ym = loc.strftime("%Y-%m")
PER = [("2024-07", "2024-12"), ("2025-01", "2025-06"), ("2025-07", "2025-12"), ("2026-01", "2026-03"), ("2026-04", "2026-07"), ("2026-08", "2026-11")]; LAB = ["Jul-Dec 2024", "Jan-Jun 2025", "Jul-Dec 2025", "Jan-Mar 2026", "Apr-Jul 2026", "Aug-Nov 2026"]
SEQ = ["#b9d3f2", "#8fb9ea", "#5f9be0", "#2a78d6", "#1b56a3", "#0e3568"]; INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8)); e = np.arange(2, 34.1, 2.0); T = {}
print("median price (EUR/MWh) and price / gas-fired cost by thermal output bin; per period: gas-fired cost L, lignite cost (12 + carbon), hard coal cost")
for i, ((lo, hi), lab) in enumerate(zip(PER, LAB)):
    k = (ym >= lo) & (ym <= hi) & D["th"].notna().values & D["p_de"].notna().values; x = D[k]; mid, p_, r_ = [], [], []
    for a, b in zip(e[:-1], e[1:]):
        kk = (x["th"] >= a) & (x["th"] < b)
        if kk.sum() >= 30: mid.append(0.5 * (a + b)); p_.append(np.median(x["p_de"][kk])); r_.append(np.median((x["p_de"] / x["L"])[kk]))
    T[lab] = dict(zip(mid, zip(p_, r_))); ax[0].plot(mid, p_, "-o", color=SEQ[i], linewidth=2, markersize=3, label=f"{lab} (gas-fired cost {x['L'].mean():.0f})"); ax[1].plot(mid, r_, "-o", color=SEQ[i], linewidth=2, markersize=3)
    print(f"  {lab}: L {x['L'].mean():4.0f}, lignite {12 + x['eua'].mean():4.0f} | " + "  ".join(f"{m:.0f}GW {T[lab][m][0]:4.0f}/{T[lab][m][1]:.2f}" for m in (5, 9, 13, 17, 21, 25, 29) if m in T[lab]))
for a, t, yl in ((ax[0], "German price against thermal output", "median price, EUR/MWh"), (ax[1], "The same, divided by the gas-fired cost", "median price / (1.85 gas + 0.37 carbon)")):
    a.set_title(t, loc="left", fontsize=10, fontweight="bold", color=INK); a.set_xlabel("lignite + hard coal + gas output, GW"); a.set_ylabel(yl); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
ax[0].legend(frameon=False, fontsize=8, loc="upper left"); ax[1].set_ylim(0, 2.2)
fig.tight_layout(); fig.savefig(f"{SP}/out/stack_target.png", dpi=110); print("figure:", f"{SP}/out/stack_target.png")
