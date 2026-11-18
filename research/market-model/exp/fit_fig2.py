"""Fit assessment of the gas term, panels P5-P7 and P9 of notes 19.1: sampling of the tempered posterior (chains drawn separately), prior predictive
(sampled with the likelihood weight at zero, so the flat-top and gas-share terms are included), prior sensitivity."""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
T = "2026-09-18"; NCH = 4; CH = [BLUE, ORANGE, AQUA, YELLOW]; out = {}
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def rhat(v):
    c = v.reshape(NCH, -1); n = c.shape[1]; h = n // 2; s = np.concatenate([c[:, :h], c[:, h:2 * h]]); W = s.var(1, ddof=1).mean(); B = h * s.mean(1).var(ddof=1); return float(np.sqrt(((h - 1) / h * W + B / h) / W))
def load(name):
    f = f"{SP}/out/struct_{name}/post_{T}.npz"
    if not os.path.exists(f): return None
    z = np.load(f); g = z["grid"]; i30 = int(np.argmin(np.abs(g - 30))); return z, g, i30
fig = plt.figure(figsize=(16.0, 9.8)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3); gref = float(np.load(f"{SP}/out/struct_pch/{T}.npz")["gref"])
# P5, P6: chains in the plane base curve against gas term at 30 GW
lims = None
for c_, (name, nm) in enumerate((("sph", "as the candidate"), ("spg", "with the prior"))):
    ax = fig.add_subplot(gs[0, c_]); L = load(name)
    if L is None: ax.axis("off"); continue
    z, g, i30 = L; b = z["b"][:, i30].astype(float); h = z["h"][:, i30].astype(float); n = len(b) // NCH; sh = gref * h / (b + gref * h)
    for k in range(NCH): ax.scatter(b[k * n:(k + 1) * n] / 10, gref * h[k * n:(k + 1) * n] / 10, s=9, color=CH[k], alpha=0.6, linewidths=0, label=f"chain {k + 1}")
    if name == "sph" and load("sphb") is not None:   # the same model, chains started at the fit WITH the prior
        z2, g2, _ = load("sphb"); b2 = z2["b"][:, i30].astype(float); h2 = z2["h"][:, i30].astype(float); ax.scatter(b2 / 10, gref * h2 / 10, s=14, facecolor="none", edgecolor=INK, linewidth=0.5, alpha=0.55, label="started at the other fit")
        zp = np.load(f"{SP}/out/struct_pcg/{T}.npz"); ax.scatter([np.interp(30.0, zp["grid"], zp["b_curve"]) / 10], [gref * np.interp(30.0, zp["grid"], zp["h_curve"]) / 10], marker="x", s=70, color=INK, linewidth=1.8, zorder=5, label="where those chains started"); out["sphb: share at 30 GW by chain"] = tuple(float((gref * h2 / (b2 + gref * h2))[k * n:(k + 1) * n].mean()) for k in range(NCH))
    out[f"{name}: r-hat base / gas term / share at 30 GW"] = (rhat(b), rhat(h), rhat(sh)); out[f"{name}: sampler's own worst r-hat, smallest ESS, divergences, seconds"] = tuple(float(v) for v in z["diag"]); out[f"{name}: share at 30 GW by chain (mean)"] = tuple(float(sh[k * n:(k + 1) * n].mean()) for k in range(NCH)); out[f"{name}: corr base vs gas term"] = float(np.corrcoef(b, h)[0, 1])
    ax.set_xlabel("base curve at 30 GW, ct/kWh"); ax.set_ylabel("gas term at 30 GW at the window's mean gas, ct/kWh"); ax.legend(frameon=False, loc="upper right", fontsize=7.5, ncol=2); clean(ax)
    ax.text(0.03, 0.05, f"r-hat of the gas share {rhat(sh):.2f}; worst r-hat in the model {z['diag'][0]:.2f},\nsmallest ESS {z['diag'][1]:.0f}; warm-up draws are not kept", transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax.set_title((f"P5  As the candidate: chains started at the other split\nwalk back, so the window does identify a low gas share" if c_ == 0 else f"P6  With the tight prior the chains sit where the prior\nputs them (tempered posterior, 4 chains)"), loc="left", fontsize=9.5, fontweight="bold", color=INK)
# P9 prior sensitivity
ax = fig.add_subplot(gs[0, 2]); d = np.load(f"{SP}/out/struct_pch/debug_{T}.npz"); wk = d["week"]; gw = pd.Series(d["gas"]).groupby(wk).mean().reindex(range(int(wk.max()) + 1)).to_numpy(); labs, cpl, shr = [], [], []
for nm, lab in (("pch", "none"), ("gsx30", "30 x wider"), ("gsx10", "10 x wider"), ("gsx3", "3 x wider"), ("pcg", "as adopted")):
    p = np.load(f"{SP}/out/struct_{nm}/par_{T}.npz"); z = np.load(f"{SP}/out/struct_{nm}/{T}.npz"); s = np.exp(np.concatenate([-np.cumsum((0.02 * p["v_e"])[::-1])[::-1], np.zeros(1)])); ok = np.isfinite(gw); b_ = np.polyfit(gw[ok], s[ok], 1); g_ = float(z["gref"])
    labs.append(lab); cpl.append(100 * b_[0] * 10); shr.append(g_ * np.interp(30.0, z["grid"], z["h_curve"]) / (np.interp(30.0, z["grid"], z["b_curve"]) + g_ * np.interp(30.0, z["grid"], z["h_curve"])))
ax.plot(range(5), cpl, "-o", color=BLUE, linewidth=2, markersize=7, markeredgecolor=SURF)
for i, (v, s_) in enumerate(zip(cpl, shr)): ax.annotate(f"{v:+.1f}%\nshare {s_:.2f}", (i, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 6.6); ax.set_xlabel("width of the gas-share prior"); ax.set_ylabel("how much the weekly price scale follows gas,\n% per 10 EUR of gas"); clean(ax)
ax.set_title("P9  Only the tight prior takes gas out of the drift;\none as wide as its source leaves the fit unchanged", loc="left", fontsize=9.5, fontweight="bold", color=INK); out["P9 coupling % per 10 EUR"] = cpl; out["P9 share at 30 GW"] = shr
# P7 prior predictive
xx = np.linspace(-5, 62, 120); rng = np.random.default_rng(0)
for c_, (name, nm) in enumerate((("sp0h", "priors as in the candidate"), ("sp0g", "with the gas-share prior"))):
    L = load(name)
    if L is None: continue
    z, g, i30 = L; S_ = len(z["b"]); pick = rng.choice(S_, min(80, S_), replace=False)
    ax = fig.add_subplot(gs[1, 0]) if c_ == 0 else ax7
    if c_ == 0: ax7 = ax
    for i in pick: ax.plot(xx, (np.interp(xx, g, z["b"][i]) + gref * np.interp(xx, g, z["h"][i])) / 10, color=(BLUE, ORANGE)[c_], linewidth=0.6, alpha=0.3)
    ax.plot([], [], color=(BLUE, ORANGE)[c_], linewidth=2, label=nm)
    bx = fig.add_subplot(gs[1, 1]) if c_ == 0 else bx7
    if c_ == 0: bx7 = bx
    shx = np.array([gref * np.interp(xx, g, z["h"][i]) / np.maximum(np.interp(xx, g, z["b"][i]) + gref * np.interp(xx, g, z["h"][i]), 1e-6) for i in range(S_)]); lo, md, hi = np.nanpercentile(shx, [5, 50, 95], axis=0)
    bx.fill_between(xx, lo, hi, color=(BLUE, ORANGE)[c_], alpha=0.18, linewidth=0); bx.plot(xx, md, color=(BLUE, ORANGE)[c_], linewidth=2, label=nm); out[f"{name}: prior price at 30 GW, 5 / 50 / 95% (ct)"] = tuple(float(v) for v in np.percentile((z["b"][:, i30] + gref * z["h"][:, i30]) / 10, [5, 50, 95])); out[f"{name}: prior share at 30 GW 5 / 50 / 95%"] = tuple(float(v) for v in np.nanpercentile(shx[:, np.argmin(np.abs(xx - 30))], [5, 50, 95])); out[f"{name}: sampler worst r-hat, ESS"] = (float(z["diag"][0]), float(z["diag"][1]))
if "ax7" in dir():
    ax7.set_ylim(-20, 80); ax7.set_xlabel("effective residual load, GW"); ax7.set_ylabel(f"price at gas {gref:.0f}, ct/kWh"); ax7.legend(frameon=False, loc="upper left", fontsize=8); clean(ax7); ax7.set_title("P7  Before any data: the candidate's priors allow\nprices from below zero to above 1 EUR/kWh (80 draws)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
    bx7.set_ylim(-0.05, 1.05); bx7.set_xlabel("effective residual load, GW"); bx7.set_ylabel("gas-priced share of the price"); bx7.legend(frameon=False, loc="lower right", fontsize=8); clean(bx7); bx7.set_title("P7  Before any data: the candidate's priors say nothing\nabout the gas share (median above 1: negative base)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
# posterior gas share by load, per chain (no pooled band: chains may disagree)
cx = fig.add_subplot(gs[1, 2])
for name, col, nm in (("sph", BLUE, "as the candidate"), ("spg", ORANGE, "with the prior")):
    L = load(name)
    if L is None: continue
    z, g, i30 = L; n = len(z["b"]) // NCH
    for k in range(NCH):
        b = z["b"][k * n:(k + 1) * n].astype(float); h = z["h"][k * n:(k + 1) * n].astype(float); sh = np.median(gref * h / np.maximum(b + gref * h, 1e-6), 0); m = g >= 10; cx.plot(g[m], sh[m], color=col, linewidth=1.6, alpha=0.9, label=nm if k == 0 else None)
cx.plot([20, 30, 42.5, 60], np.array([0.78, 0.91, 0.95, 0.97]) * 0.733, "o", color=INK, markersize=5, label="prior's centre"); cx.set_ylim(0, 1); cx.set_xlabel("effective residual load, GW"); cx.set_ylabel("gas-priced share of the price"); cx.legend(frameon=False, loc="lower right", fontsize=8); clean(cx)
cx.set_title("P5/P6  Gas share by load, one line per chain: the data's\nsplit and the prior's are far apart", loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig.savefig(f"{SP}/out/fit_gas_2.png", dpi=110, bbox_inches="tight"); print("figure:", f"{SP}/out/fit_gas_2.png")
for k_, v in out.items(): print(f"  {k_}: " + (", ".join(f"{float(x):.3f}" for x in v) if isinstance(v, (tuple, list)) else f"{v:.3f}"))
