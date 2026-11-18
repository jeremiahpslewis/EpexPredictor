"""Look at a fitted coupled-market model against the data, including data it was not fitted to.
Prints what the fit implies (blocks, exchanges) next to what is known or measured, and draws one figure of small
multiples: the price against residual load, each German technology block against realised output by fuel, and the
exchange with four neighbours against the scheduled exchange. usage: star_check.py RUN PERIOD TAG [SEED]"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib; S = importlib.import_module(os.environ.get("STARMOD", "star"))
run, period, tag = sys.argv[1], sys.argv[2], sys.argv[3]; seed = sys.argv[4] if len(sys.argv) > 4 else "0"
SP = S.lc.SP; z = np.load(f"{SP}/out/star_{run}/{tag}_s{seed}.npz"); raw = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}; q = S.unpack(raw)
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96
lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4; H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1)
FWD = os.environ.get("FWD", "")            # "period:tag": clear the market from this fit's last day up to that date, nothing refitted, cost level frozen
if FWD:
    fp, ft = FWD.split(":"); I2 = np.load(f"{SP}/out/inputs_{fp}.npz"); hi = int(I2["cuts"][[str(c)[:10] for c in I2["cutoffs"]].index(ft)]) - 96; nw = int(week.max()) + 1
    lo = d1; H = S.window(lo, hi); d = S.data_of(H, np.full(len(H["P"]), nw - 1), nw); pe, e_eq = S.equilibrium(q, d, blend=False) if "theta" in q else S.equilibrium(q, d)   # forward: published residual load only
cap, mc, wd, sl, shift, ct = S.build(q, d, np); P = H["P"]; ok = d["ok"][:, 0]; n = len(P); wd = S.w3(wd, n) if hasattr(S, "w3") else np.broadcast_to(wd[None], (n,) + wd.shape); wdm = np.median(wd, 0)
FL = S.floors(q, d, np) * np.ones((n, 1)) if hasattr(S, "floors") else np.tile(q["floor"], (n, 1))        # must-run per row (n, 3)
if not FWD: pe = z["peq"].astype(float); e_eq = z["eeq"].astype(float)
hq = lambda col: np.nanmean(S.D[col].to_numpy(float)[lo:lo + 4 * n].reshape(-1, 4), 1)
sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
np.set_printoptions(precision=1, suppress=True, linewidth=220)
print(f"fit {tag} (seed {seed}): {n} hours; gas mean {H['gas'].mean():.0f} EUR/MWh; " + ((f"offer share of the gap to gas-fired cost: lignite {ct[:, 0].min():.2f}..{ct[:, 0].max():.2f} (last week {ct[-1, 0]:.2f}), hard coal {ct[:, 1].min():.2f}..{ct[:, 1].max():.2f} (last week {ct[-1, 1]:.2f})" if "phi0" in q else f"offer premium over the window (zero in the last week): lignite {ct[:, 0].min():+.0f}..{ct[:, 0].max():+.0f}, hard coal {ct[:, 1].min():+.0f}..{ct[:, 1].max():+.0f} EUR/MWh") if ct.ndim == 2 else f"common cost level over the window {ct.min():+.0f}..{ct.max():+.0f} (zero in the last week)"))
if ct.ndim == 2 and not FWD:
    mo_ = (H["day"] - H["day"].min()) // 60; print("   " + ("share" if "phi0" in q else "premium") + " by two-month block, lignite: " + " ".join((f"{ct[mo_ == k, 0].mean():.2f}" if "phi0" in q else f"{ct[mo_ == k, 0].mean():+.0f}") for k in range(int(mo_.max()) + 1)) + " | hard coal: " + " ".join((f"{ct[mo_ == k, 1].mean():.2f}" if "phi0" in q else f"{ct[mo_ == k, 1].mean():+.0f}") for k in range(int(mo_.max()) + 1)))
if "theta" in q: print(f"   market-expected residual load = published + {float(q['theta']):.2f} x (realised - published); response of realised output to the rest of the difference, GW per GW: lignite {q['rho'][0]:.2f}, hard coal {q['rho'][1]:.2f}, gas {q['rho'][2]:.2f}" + (("; heating threshold " + (" ".join(f"{zn} {t:.0f}" for zn, t in zip(S.ZN, q["tstar"])) if np.ndim(q["tstar"]) == 1 else f"{float(q['tstar']):.1f}") + " C") if "tstar" in q else ""))
if "gx" in q: print("   exchange shift per sd of German wind / sun, GW: " + "  ".join(f"{zn} {q['gx'][0, k]:+.2f}/{q['gx'][1, k]:+.2f}" for k, zn in enumerate(S.ZN[1:])))
print("blocks: capacity GW (mean) / marginal cost EUR/MWh at mean gas / width.   Installed 2025 for comparison where it maps directly")
INST = {"de": "lignite 14.8, hard coal 15.2, gas 35.3 (+oil 3.6)", "fr": "nuclear 63, run-of-river 11.8, reservoir+pumped 13.9, gas 12.8", "nl": "coal 4.0, gas 18.4", "pl": "lignite 7.0, hard coal 18.5, gas 5.4"}
for zn in ("de", "fr", "nl", "pl"):
    i = S.ZN.index(zn); print(f"  {zn}: " + "  ".join(f"[{['low', 'A', 'B', 'ccgt', 'peak'][k]} {cap[:, i, k].mean():.1f} / {mc[:, i, k].mean():.0f} / {wdm[i, k]:.0f}]" for k in range(5)) + f"   installed: {INST[zn]}")
print("exchange: fitted capacity C [GW] and price-difference scale kappa [EUR/MWh] per border, prior C, and the scheduled exchange (5%..95%)")
for k, zn in enumerate(S.ZN[1:]):
    x = -hq(f"x_{zn}"); print(f"  {zn}: export cap {q['Ce'][k]:.1f}, import cap {q['Ci'][k]:.1f} (prior {S.NTC[k]:.1f} each), offset {q['m'][k]:+.0f}, kappa {q['kappa'][k]:.0f}; scheduled exports {np.nanpercentile(x, 5):+.1f}..{np.nanpercentile(x, 95):+.1f}; model at equilibrium {np.percentile(e_eq[:, k], 5):+.1f}..{np.percentile(e_eq[:, k], 95):+.1f}; correlation model vs scheduled {np.corrcoef(e_eq[:, k][np.isfinite(x)], x[np.isfinite(x)])[0, 1]:+.2f}")
xs = -hq("x_sum"); m = np.isfinite(xs); print(f"  net exports: scheduled 5%..95% {np.nanpercentile(xs, 5):+.1f}..{np.nanpercentile(xs, 95):+.1f} GW; model {np.percentile(e_eq.sum(1), 5):+.1f}..{np.percentile(e_eq.sum(1), 95):+.1f}; correlation {np.corrcoef(e_eq.sum(1)[m], xs[m])[0, 1]:+.2f}")
print(f"German output by fuel: model (must-run floors {np.round(q['floor'], 1)} GW + block at the observed price) against realised output")
G = {}
for lab, ks, col in (("lignite", [1], "g_de_lignite"), ("hard coal", [2], "g_de_coal"), ("gas", [3, 4], "g_de_gas")):
    pef = lambda k: P[:, 0] + (q["lam"][0, k] * (d["pdmz"][:, 0] - P[:, 0]) if "lam" in q and "pdmz" in d else 0.0)      # the price the block responds to (inertia)
    if "da" in q: mod = S.dispatch(q, d, np.nan_to_num(P[:, 0]), FL, np)[:, ["lignite", "hard coal", "gas"].index(lab)] + q["rho"][["lignite", "hard coal", "gas"].index(lab)] * (1.0 - float(q["theta"])) * d["drl"]; obs = hq(col); mm = np.isfinite(obs) & ok; G[lab] = (mod, obs); print(f"  {lab:<9} dispatch layer: mean {mod[mm].mean():.1f} GW, realised {obs[mm].mean():.1f}; correlation {np.corrcoef(mod[mm], obs[mm])[0, 1]:+.2f}; realised 5%..95% {np.percentile(obs[mm], 5):.1f}..{np.percentile(obs[mm], 95):.1f}, model {np.percentile(mod[mm], 5):.1f}..{np.percentile(mod[mm], 95):.1f}"); continue
    mod = (q["rho"][["lignite", "hard coal", "gas"].index(lab)] * (1.0 - float(q["theta"])) * d["drl"] if "rho" in q else 0.0) + FL[:, ["lignite", "hard coal", "gas"].index(lab)] + sum(cap[:, 0, k] * sg((pef(k) - mc[:, 0, k]) / wd[:, 0, k]) for k in ks); obs = hq(col); mm = np.isfinite(obs) & ok; G[lab] = (mod, obs)
    print(f"  {lab:<9} model mean {mod[mm].mean():.1f} GW, realised {obs[mm].mean():.1f}; correlation {np.corrcoef(mod[mm], obs[mm])[0, 1]:+.2f}; realised 5%..95% {np.percentile(obs[mm], 5):.1f}..{np.percentile(obs[mm], 95):.1f}, model {np.percentile(mod[mm], 5):.1f}..{np.percentile(mod[mm], 95):.1f}")
if "da" in q: print("  dispatch layer: half loading at price / gas-fired cost = %.2f lignite, %.2f hard coal, %.2f gas (relative width %.2f / %.2f / %.2f); flexible share of availability %.2f / %.2f / %.2f; weight on the day's mean price %.2f / %.2f / %.2f; own-cost factor %.2f / %.2f" % (*q["da"], *q["dw"], *q["dk"], *q["dlam"], q["db"][0], q["db"][1]))
HASPS = "p_ps" in q
if HASPS:
    psm = q["p_ps"] * np.tanh((P[:, 0] - d["pdm"]) / q["k_ps"]); mm = d["psok"]; print(f"pumped storage: capacity {float(q['p_ps']):.1f} GW, price scale {float(q['k_ps']):.0f} EUR/MWh; model at observed prices against realised: correlation {np.corrcoef(psm[mm], H['ps'][mm])[0, 1]:+.2f}; realised 5%..95% {np.percentile(H['ps'][mm], 5):+.1f}..{np.percentile(H['ps'][mm], 95):+.1f}, model {np.percentile(psm[mm], 5):+.1f}..{np.percentile(psm[mm], 95):+.1f}")
if "f1" in q and "ae" not in q:
    print("heat-led must-run: German lignite %.1f + %.2f per degree, hard coal %.1f + %.2f, gas %.1f + %.2f GW" % (q["floor"][0], q["f1"][0], q["floor"][1], q["f1"][1], q["floor"][2], q["f1"][2]) + "; other heat-led output per degree, GW: " + "  ".join(f"{zn} {q['hz'][i]:.2f}" for i, zn in enumerate(S.ZN)))
if "f1" in q and "ae" in q:
    print("heat-led must-run: German lignite %.1f + %.2f per degree, hard coal %.1f + %.2f, gas %.1f + %.2f GW; on a day at 0 C that is %.1f / %.1f / %.1f GW" % (q["floor"][0], q["f1"][0], q["floor"][1], q["f1"][1], q["floor"][2], q["f1"][2], *(q["floor"] + 15 * q["f1"])))
    print("   other heat-led output per degree, GW: " + "  ".join(f"{zn} {q['hz'][i]:.2f}" for i, zn in enumerate(S.ZN)))
    print("border capacity against German weather (export / import capacity at wind -1 sd, +1 sd | sun -1 sd, +1 sd):")
    for k, zn in enumerate(S.ZN[1:]): print(f"   {zn}: wind {q['Ce'][k] * np.exp(-q['ae'][0, k]):.1f}/{q['Ci'][k] * np.exp(-q['ai'][0, k]):.1f} -> {q['Ce'][k] * np.exp(q['ae'][0, k]):.1f}/{q['Ci'][k] * np.exp(q['ai'][0, k]):.1f} | sun {q['Ce'][k] * np.exp(-q['ae'][1, k]):.1f}/{q['Ci'][k] * np.exp(-q['ai'][1, k]):.1f} -> {q['Ce'][k] * np.exp(q['ae'][1, k]):.1f}/{q['Ci'][k] * np.exp(q['ai'][1, k]):.1f}")
if "lam" in q: print("inertia (weight on the day's mean price): German lignite %.2f, hard coal %.2f, combined cycle %.2f, peakers %.2f; neighbours' median %.2f" % (*q["lam"][0, 1:5], float(np.median(q["lam"][1:, 1:5]))))
print(f"German low-cost block: {cap[:, 0, 0].mean():.1f} GW at {mc[:, 0, 0].mean():.0f} EUR/MWh, width {wdm[0, 0]:.0f}; curtailment block {q['capf'][0, 5] if q['capf'].shape[1] > 5 else float('nan'):.1f} GW; elastic slope {sl[0]:.2f} GW per 10 ct")
err = (P[:, 0] - pe[:, 0]) / 10; rl = H["rl"][:, 0]
print(f"German price, full equilibrium, {'forward on published residual load' if FWD else 'in sample'}: MAE {np.abs(err[ok]).mean():.2f} ct" + (f" (with the published residual load alone {np.abs((P[:, 0] - z['peq_pub'][:, 0].astype(float))[ok]).mean() / 10:.2f})" if ("peq_pub" in z.files and not FWD) else "") + ". Mean error (actual - model) by ...")
print("  residual load GW: " + "  ".join(f"[{a},{b}) {err[ok & (rl >= a) & (rl < b)].mean():+.2f}" for a, b in ((-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90))))
print("  hour block:       " + "  ".join(f"{a}-{b}h {err[ok & (H['hour'] >= a) & (H['hour'] < b)].mean():+.2f}" for a, b in ((0, 6), (6, 10), (10, 16), (16, 20), (20, 24))))
mo = (H["day"] - H["day"].min()) // (30 if FWD else 60); print(f"  {'month' if FWD else 'two-month'} block:  " + "  ".join(f"{err[ok & (mo == k)].mean():+.2f}" for k in range(int(mo.max()) + 1)))
print("  gas tercile:      " + "  ".join(f"{err[ok & (H['gas'] >= a) & (H['gas'] < b)].mean():+.2f}" for a, b in zip(np.percentile(H["gas"], [0, 33, 67]), np.percentile(H["gas"], [33, 67, 100]) + 1e-6)))
print("neighbours' prices, own clearing at observed exchanges: MAE ct " + "  ".join(f"{zn} {np.nanmean(np.abs(P[:, i] - pe[:, i])[d['ok'][:, i]]) / 10:.2f}" for i, zn in enumerate(S.ZN) if i > 0))
# ---- figure
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(2, 4, figsize=(15, 7.2)); ax = ax.ravel()
def binned(x, y, edges):
    c, v = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() >= 25: c.append(np.median(x[m])); v.append(np.median(y[m]))
    return np.array(c), np.array(v)
def style(a, title, xl, yl):
    a.set_title(title); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
e = np.arange(-10, 71, 4.0); c1, v1 = binned(rl[ok], P[ok, 0] / 10, e); c2, v2 = binned(rl[ok], pe[ok, 0] / 10, e)
ax[0].plot(c2, v2, color=BLUE, linewidth=2, label="model, market cleared"); ax[0].plot(c1, v1, "o", color=ORANGE, markersize=5, label="observed"); style(ax[0], "German price against residual load", "German residual load, GW", "median price, ct/kWh"); ax[0].legend(frameon=False, loc="upper left")
pg = np.linspace(-50, 400, 200)
if "da" in q:
    ea_ = d["eua"] + 70.0; Lg_ = 1.85 * d["gas"] + 0.37 * ea_; own_ = np.stack([12.0 + ea_, 4.0 + 2.6 * d["coal"] + 0.85 * ea_, Lg_], 1) / Lg_[:, None]; pos_ = np.maximum(q["da"][None], q["db"][None] * own_)
    for i_, (a, lab) in enumerate(zip(ax[1:4], ("lignite", "hard coal", "gas"))):
        mod, obs = G[lab]; mm = np.isfinite(obs) & ok; r_ = (P[:, 0] + q["dlam"][i_] * (d["pdmz"][:, 0] - P[:, 0])) / Lg_ - pos_[:, i_]; util = (obs - FL[:, i_]) / np.maximum(q["dk"][i_] * d["avail"][:, 0, 1 + i_], 0.3)
        c, v = binned(r_[mm], util[mm], np.r_[-2, -0.8, np.arange(-0.6, 0.81, 0.1), 1.2, 3]); g = np.linspace(-0.9, 1.1, 200)
        a.plot(g, sg(g / q["dw"][i_]), color=BLUE, linewidth=2, label="dispatch layer"); a.plot(c, v, "o", color=ORANGE, markersize=5, label="realised"); style(a, f"German {lab}: loading against relative price", "price / gas-fired cost, minus the fuel's half-loading point", "median (output - must-run) / flexible capacity"); a.set_xlim(-0.9, 1.1); a.set_ylim(-0.15, 1.35)
for a, (lab, ks) in zip(ax[1:4] if "da" not in q else [], (("lignite", [1]), ("hard coal", [2]), ("gas", [3, 4]))):
    mod, obs = G[lab]; mm = np.isfinite(obs) & ok; k0 = ks[0]; fl = FL[:, ["lignite", "hard coal", "gas"].index(lab)]
    capt = sum(cap[:, 0, k] for k in ks); marg = (pef(k0) - mc[:, 0, k0]) / 10; util = (obs - fl) / np.maximum(capt, 0.3)             # utilisation of the flexible part against the margin over that hour's marginal cost
    c, v = binned(marg[mm], util[mm], np.r_[-30, -12, np.arange(-9, 12.1, 1.5), 16, 30]); g = np.linspace(-14, 16, 200)
    a.plot(g, sg(g * 10 / wdm[0, k0]), color=BLUE, linewidth=2, label="model block"); a.plot(c, v, "o", color=ORANGE, markersize=5, label="realised"); style(a, f"German {lab}: utilisation against margin", "price minus the block's marginal cost that hour, ct/kWh", "median (output - must-run) / capacity"); a.set_xlim(-14, 16); a.set_ylim(-0.15, 1.35)
ax[1].legend(frameon=False, loc="upper left")
if HASPS:
    dpm = (P[:, 0] - d["pdm"]); mm = d["psok"]; c, v = binned(dpm[mm], H["ps"][mm], np.r_[-300, -90, -60, -40, -25, -15, -8, -3, 3, 8, 15, 25, 40, 60, 90, 300]); g = np.linspace(-110, 110, 200)
    ax[4].plot(g / 10, q["p_ps"] * np.tanh(g / q["k_ps"]), color=BLUE, linewidth=2, label="model"); ax[4].plot(c / 10, v, "o", color=ORANGE, markersize=5, label="realised"); style(ax[4], "Pumped storage against the price within the day", "German price minus its mean of the day, ct/kWh", "median net output, GW"); ax[4].set_xlim(-11, 11)
for a, zn in zip(ax[5:8] if HASPS else ax[4:8], ("nl", "fr", "dk") if HASPS else ("nl", "fr", "at", "dk")):
    k = S.ZN.index(zn) - 1; dp = P[:, k + 1] - P[:, 0]; x = -hq(f"x_{zn}"); mm = np.isfinite(x) & np.isfinite(dp); c, v = binned(dp[mm], x[mm], np.r_[-200, -80, -50, -30, -20, -12, -7, -3, -1, 1, 3, 7, 12, 20, 30, 50, 80, 200]); g = np.linspace(-90, 90, 200)
    a.plot(g / 10, S.exch(q, np.tile(g[:, None], (1, S.Z - 1)), np)[:, k], color=BLUE, linewidth=2, label="model"); a.plot(c / 10, v, "o", color=ORANGE, markersize=5, label="scheduled exchange"); style(a, f"Exports to {zn.upper()} against the price difference", f"{zn.upper()} price minus German price, ct/kWh", "median German exports, GW"); a.set_xlim(-9, 9)
ax[5 if HASPS else 4].legend(frameon=False, loc="upper left")
fig.suptitle((f"Coupled-market model fitted to the 540 days before {tag}, applied unchanged to the following months up to {FWD.split(':')[1]}" if FWD else f"Coupled-market model fitted to the 540 days before {tag}: what it implies against the data"), x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.925, ("Nothing refitted: only gas price, residual loads and measured availability are updated; the latent cost level is frozen. " if FWD else "Output by fuel and scheduled exchanges are data in this version of the model. ") + "Dots are medians within bins of the horizontal variable.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92)); out = f"{SP}/out/star_{run}_{tag}_s{seed}_{'forward' if FWD else 'check'}.png"; fig.savefig(out, dpi=110); print("figure:", out)
