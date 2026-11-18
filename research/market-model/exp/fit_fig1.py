"""Fit assessment of the gas term of the candidate's curve, panels P1-P4 and P8 of notes 19.1 (no sampling needed)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
GREY = LinearSegmentedColormap.from_list("g", ["#dcdbd5", "#8f8e87", "#2b2a27"]); T = "2026-09-18"; out = {}
fig = plt.figure(figsize=(16.0, 9.8)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
dh, dg = np.load(f"{SP}/out/struct_pch/debug_{T}.npz"), np.load(f"{SP}/out/struct_pcg/debug_{T}.npz"); ph, pg = np.load(f"{SP}/out/struct_pch/par_{T}.npz"), np.load(f"{SP}/out/struct_pcg/par_{T}.npz"); zh, zg = np.load(f"{SP}/out/struct_pch/{T}.npz"), np.load(f"{SP}/out/struct_pcg/{T}.npz")
sdr = lambda p: np.concatenate([-np.cumsum((0.02 * p["v_e"])[::-1])[::-1], np.zeros(1)])
# P1 in sample: last 8 weeks, data and both fitted curves
ax = fig.add_subplot(gs[0, 0]); wk = dh["week"]; k = wk >= wk.max() - 7; gm = float(dh["gas"][k].mean()); xx = np.linspace(-5, 62, 200)
sc = ax.scatter(dh["x"][k], dh["y"][k] / 10, c=dh["gas"][k], cmap=GREY, s=7, linewidths=0, alpha=0.8, vmin=55, vmax=90); cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045); cb.set_label("gas price, EUR/MWh", fontsize=8); cb.outline.set_visible(False)
for z, p, col, nm in ((zh, ph, BLUE, "as the candidate"), (zg, pg, ORANGE, "with the prior")):
    scl = float(np.exp(sdr(p)[-8:]).mean()); ax.plot(xx, (np.interp(xx, z["grid"], z["b_curve"]) + gm * np.interp(xx, z["grid"], z["h_curve"])) * scl / 10, color=col, linewidth=2, label=nm)
mae = [float(np.abs(d["r"][k]).mean() / 10) for d in (dh, dg)]; out["P1 in-sample MAE last 8 weeks, as candidate / with prior"] = mae
ax.set_xlim(-5, 62); ax.set_ylim(-3, 32); ax.set_xlabel("effective residual load, GW"); ax.set_ylabel("price, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ax.set_title(f"P1  In sample the two fits are the same curve\n(last 8 weeks; error {mae[0]:.2f} and {mae[1]:.2f} ct)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
# P2 weekly price-scale path against weekly gas
ax = fig.add_subplot(gs[0, 1]); gw = pd.Series(dh["gas"]).groupby(wk).mean().reindex(range(int(wk.max()) + 1)).to_numpy()
for p, col, nm in ((ph, BLUE, "as the candidate"), (pg, ORANGE, "with the prior")):
    s_ = np.exp(sdr(p)); ok = np.isfinite(gw); b_ = np.polyfit(gw[ok], s_[ok], 1); ax.plot(gw[ok], s_[ok], "-", color=col, linewidth=1, alpha=0.45); ax.scatter(gw[ok], s_[ok], s=16, color=col, edgecolor=SURF, linewidth=0.5, zorder=3, label=f"{nm}: {100 * b_[0] * 10:+.1f}% per 10 EUR")
    ax.plot([gw[ok].min(), gw[ok].max()], np.polyval(b_, [gw[ok].min(), gw[ok].max()]), color=col, linewidth=2); out[f"P2 slope of scale on gas ({nm}), per 10 EUR"] = float(b_[0] * 10); out[f"P2 corr ({nm})"] = float(np.corrcoef(gw[ok], s_[ok])[0, 1]); out[f"P2 scale range ({nm})"] = (float(s_.min()), float(s_.max()))
ax.axhline(1.0, color=NEUT, linewidth=0.8); ax.set_xlabel("weekly mean gas price, EUR/MWh"); ax.set_ylabel("weekly price scale (last week = 1)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8); clean(ax)
ax.set_title("P2  Without the prior the weekly drift tracks gas\n(78 weeks to 18 Sep 2026, joined in time order)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
# P3 frozen March curve in raw space: night price against gas within load bands, published loads, coloured by month
R = []
for period, sfx in (("summer", "_s"), ("autumn", "_a")):
    I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float)
    for j, c in enumerate(I["cutoffs"]):
        tag = str(c)[:10]; z = np.load(f"{SP}/out/struct_zh4m/{tag}.npz"); R.append(pd.DataFrame({"tag": tag, "x": z["x_known"][:32], "a": a[j, :32], "gas": float(I["gas"][j]), "ph": np.load(f"{SP}/out/struct_zh4m/{tag}.npz")["known_struct"][:32], "pg": np.load(f"{SP}/out/struct_zg4m/{tag}.npz")["known_struct"][:32]}))
F = pd.concat(R, ignore_index=True); F["mo"] = pd.to_datetime(F["tag"]).dt.month + (pd.to_datetime(F["tag"]).dt.day - 1) / 31.0; zfh, zfg = np.load(f"{SP}/out/struct_fh4/2026-03-31.npz"), np.load(f"{SP}/out/struct_fg4/2026-03-31.npz"); gl = np.linspace(40, 108, 50)
for c_, (lo, hi) in enumerate(((25, 35), (35, 50))):
    ax = fig.add_subplot(gs[1, c_]); d = F[(F.x >= lo) & (F.x < hi)]; g = d.groupby("tag").agg(a=("a", "median"), gas=("gas", "first"), mo=("mo", "first"), x=("x", "median"), n=("a", "size")); g = g[g.n >= 6]; xb = float(g.x.median())
    sc = ax.scatter(g.gas, g.a, c=g.mo, cmap=GREY, s=26, edgecolor=SURF, linewidth=0.5, vmin=4, vmax=11.3, zorder=3); cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, ticks=[4, 6, 8, 10]); cb.ax.set_yticklabels(["Apr", "Jun", "Aug", "Oct"]); cb.outline.set_visible(False)
    for z, col, nm in ((zfh, BLUE, "as the candidate"), (zfg, ORANGE, "with the prior")):
        b0 = float(np.interp(xb, z["grid"], z["b_curve"])); h0 = float(np.interp(xb, z["grid"], z["h_curve"])); ax.plot(gl, (b0 + gl * h0) / 10, color=col, linewidth=2, label=f"{nm}: {h0 / 10:.3f} ct per EUR"); out[f"P3 {lo}-{hi} GW slope {nm}"] = h0 / 10
    sl = np.polyfit(g.gas, g.a, 1); out[f"P3 {lo}-{hi} GW data slope, all months (confounded with time)"] = float(sl[0]); wi = [np.polyfit(v.gas, v.a, 1)[0] for _, v in g.groupby(g.mo.astype(int)) if len(v) >= 6 and v.gas.max() - v.gas.min() >= 6]; out[f"P3 {lo}-{hi} GW data slopes within single months"] = [round(float(v), 3) for v in wi]; out[f"P3 {lo}-{hi} GW nights"] = int(len(g))
    ax.set_xlabel("gas price of the day, EUR/MWh"); ax.set_ylabel("median night price in the band, ct/kWh"); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
    ax.set_title(f"P3  Curve frozen 31 Mar against later nights, {lo}-{hi} GW:\nthe level moves at flat gas (shade = month)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
# P4 the prior's source: 2021-2023, 25-35 GW net of nuclear
ax = fig.add_subplot(gs[1, 2]); H = pd.read_parquet(f"{SP}/out/hist2123.parquet").loc["2021-01-01":"2023-12-31"]; H["x"] = H["resload"] - H["nuclear"]; H["L"] = 1.85 * H["gas"] + 0.37 * H["eua"]; d = H[(H.x >= 25) & (H.x < 35)].dropna(subset=["p", "L"]); day = d.index.tz_convert("Europe/Berlin").normalize()
g = d.groupby(day).agg(p=("p", "median"), L=("L", "median"), n=("p", "size")); g = g[g.n >= 3]; g["yr"] = g.index.year
def lad(x, y):
    b = np.polyfit(x, y, 1)
    for _ in range(25): w = 1 / np.maximum(np.abs(y - np.polyval(b, x)), 2.0); b = np.polyfit(x, y, 1, w=np.sqrt(w))
    return b
for yr, col in ((2021, BLUE), (2022, ORANGE), (2023, AQUA)):
    v = g[g.yr == yr]; b = lad(v.L.values, v.p.values); ax.scatter(v.L / 10, v.p / 10, s=9, color=col, alpha=0.55, linewidths=0); xs = np.array([v.L.quantile(0.03), v.L.quantile(0.97)]); ax.plot(xs / 10, np.polyval(b, xs) / 10, color=col, linewidth=2.2, label=f"{yr}: slope {b[0]:.2f}"); out[f"P4 slope {yr}"] = float(b[0])
b = lad(g.L.values, g.p.values); xs = np.array([g.L.quantile(0.01), g.L.quantile(0.99)]); ax.plot(xs / 10, np.polyval(b, xs) / 10, color=INK, linewidth=1.6, linestyle=(0, (5, 2)), label=f"all three years: slope {b[0]:.2f}"); out["P4 slope pooled"] = float(b[0])
ax.set_xlabel("gas-fired cost (1.85 gas + 0.37 carbon), ct/kWh"); ax.set_ylabel("median day price in the band, ct/kWh"); ax.set_xlim(0, 47); ax.set_ylim(0, 60); ax.legend(frameon=False, loc="upper left", fontsize=8); clean(ax)
ax.set_title("P4  The prior's source, 2021-2023 at 25-35 GW:\nthe three years do not share one slope", loc="left", fontsize=9.5, fontweight="bold", color=INK)
# P8 day-level replicated check (baseline arm; the noise model is the same in both)
ax = fig.add_subplot(gs[0, 2]); rng = np.random.default_rng(0); s = ph["s"].astype(float); nu = float(ph["nu"]); gref = float(ph["gref"]); zx = (dh["x"] - 25.0) / 15.0; scale = 10.0 * np.exp(s[0] + s[1] * np.log(dh["gas"] / gref) + s[2] * zx + s[3] * zx ** 2)
bands = (("below 15 GW", dh["x"] < 15), ("15-35 GW", (dh["x"] >= 15) & (dh["x"] < 35)), ("above 35 GW", dh["x"] >= 35)); NR = 300; rep = scale[None] * rng.standard_t(nu, size=(NR, len(scale)))
def T_(res, m):
    g_ = pd.Series(np.where(m, res, np.nan)).groupby(dh["day"]); v = g_.mean()[g_.count() >= 4]; return float(np.sqrt(np.mean(v ** 2)) / 10)
for i, (nm, m) in enumerate(bands):
    obs = T_(dh["r"], m); tr = np.array([T_(rep[k_], m) for k_ in range(NR)]); lo, md, hi = np.percentile(tr, [5, 50, 95]); out[f"P8 {nm}: observed / replicated median (5-95%)"] = (obs, md, lo, hi)
    hr_o = float(np.median(np.abs(dh["r"][m])) / 10); hr_r = float(np.median(np.abs(rep[:, m])) / 10); out[f"P8 {nm}: hourly median |residual| observed / replicated"] = (hr_o, hr_r)
    ax.plot([lo, hi], [i, i], color=NEUT, linewidth=5, solid_capstyle="round", alpha=0.7, label="replicated from the fitted noise model (5-95%)" if i == 0 else None); ax.scatter([md], [i], s=40, color=NEUT, zorder=3, edgecolor=SURF, linewidth=0.8); ax.scatter([obs], [i], s=64, color=INK, zorder=4, edgecolor=SURF, linewidth=0.8, label="observed" if i == 0 else None); ax.annotate(f"{obs / md:.1f} x", (obs, i), xytext=(8, -3), textcoords="offset points", fontsize=8, color=MUTED)
ax.set_yticks(range(3)); ax.set_yticklabels([b_[0] for b_ in bands]); ax.set_ylim(-0.6, 3.1); ax.set_xlim(0, None); ax.set_xlabel("size of a typical day-level error in the band, ct/kWh\n(root mean square of the daily mean residual)"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax)
ax.set_title("P8  Errors come as whole days; the noise model\nassumes independent hours", loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig.savefig(f"{SP}/out/fit_gas_1.png", dpi=110, bbox_inches="tight"); print("figure:", f"{SP}/out/fit_gas_1.png")
for k_, v in out.items(): print(f"  {k_}: {v if not isinstance(v, (tuple, list)) else [round(float(x), 3) for x in v]}" if not isinstance(v, float) else f"  {k_}: {v:.3f}")
