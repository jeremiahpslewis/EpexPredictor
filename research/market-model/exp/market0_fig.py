"""Pictures for the market model, level M0c (notes 21): (A) before data: simulated markets from the priors, simulate-and-recover; (B) against data."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm, t as tdist
import market0 as M
SP = M.SP; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
rng = np.random.default_rng(0); out = {}
# ---------------------------------------------------------------- figure A: before data
Z = np.load(f"{SP}/out/market0_prior.npz"); R = np.load(f"{SP}/out/market0_recover.npz"); fig = plt.figure(figsize=(17.5, 12.6)); gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.28)
ax = fig.add_subplot(gs[0, 0]); ax.scatter(Z["rl"], Z["p_obs"] / Z["L"], s=4, color=ORANGE, alpha=0.5, linewidths=0); ax.set_xlim(-25, 70); ax.set_ylim(-1.5, 4); ax.set_xlabel("residual load, GW"); ax.set_ylabel("price / gas-plus-CO2 cost"); clean(ax); ttl(ax, "The real market, 3000 hours\n(for comparison)")
for i, k in enumerate((3, 11, 27)):
    ax = fig.add_subplot(gs[0, i + 1]); ax.scatter(Z["rl"], Z["P"][k] / Z["L"], s=4, color=BLUE, alpha=0.5, linewidths=0); ax.set_xlim(-25, 70); ax.set_ylim(-1.5, 4); ax.set_xlabel("residual load, GW"); clean(ax)
    ttl(ax, f"A market simulated from the priors\n(thermal {Z['K_T'][k]:.0f} GW, import slope {Z['x1'][k]:.0f}, shelf {Z['K_s'][k]:.1f} GW)")
P = Z["P"]; stats = (("share of hours at or below zero", (P <= 0).mean(1), float((Z["p_obs"] <= 0).mean()), (0, 0.5)), ("share of hours exactly on the shelf", (P == 0).mean(1), float((np.abs(Z["p_obs"]) <= M.EPS).mean()), (0, 0.15)), ("median price / cost", np.median(P / Z["L"][None], 1), float(np.median(Z["p_obs"] / Z["L"])), (0, 2.2)), ("mean thermal output, GW", Z["T"].mean(1), float(Z["T_obs"].mean()), (0, 40)))
for i, (nm, v, obs, lim) in enumerate(stats):
    ax = fig.add_subplot(gs[1, i]); ax.hist(np.clip(v, *lim), bins=30, range=lim, color=BLUE, alpha=0.75, linewidth=0); ax.axvline(obs, color=ORANGE, linewidth=2.2); ax.set_xlabel(nm); ax.set_yticks([]); clean(ax); ax.grid(False); q = np.mean(v < obs); out[f"prior predictive, {nm}: observed {obs:.3f} at quantile"] = float(q)
    ttl(ax, ("200 simulated markets against the real one\n(orange line)" if i == 0 else f"real market at the {100 * q:.0f}th percentile\nof the simulated ones"))
ax = fig.add_subplot(gs[2, :3]); PN = M.PN; y = np.arange(len(PN))[::-1]
for i, k in enumerate(PN):
    tv = float(R[f"true_{k}"]); q = np.percentile(R[k], [5, 50, 95]) / tv; ax.plot([q[0], q[2]], [y[i], y[i]], color=BLUE, linewidth=4, solid_capstyle="round", alpha=0.45); ax.scatter([q[1]], [y[i]], s=34, color=BLUE, zorder=3, edgecolor=SURF, linewidth=0.8)
ax.axvline(1.0, color=INK, linewidth=1); ax.set_yticks(y); ax.set_yticklabels(["must-run floor", "thermal capacity", "thermal threshold", "thermal spread", "imports at price = cost", "import slope", "price-responsive rest: size", "... centre", "... width", "shelf at zero", "constant rest B", "demand shock sd", "thermal shock sd", "import shock sd"], fontsize=8.5); ax.set_xlim(0.8, 1.2); ax.set_xlabel("recovered value / true value (dot: median, bar: 90% band)"); clean(ax)
ttl(ax, f"Simulate from known parameters, then fit: all come back (worst r-hat {float(R['rhat']):.3f}; 6699 hours with gaps)")
ax = fig.add_subplot(gs[2, 3]); hi = P.max(1) / 10; ax.hist(np.clip(hi, 0, 400), bins=30, range=(0, 400), color=BLUE, alpha=0.75, linewidth=0); ax.axvline(float(Z["p_obs"].max()) / 10, color=ORANGE, linewidth=2.2); ax.set_xlabel("highest price in the sample, ct/kWh"); ax.set_yticks([]); clean(ax); ax.grid(False); ttl(ax, "Simulated markets always clear;\npeaks are of a plausible size")
fig.suptitle("Market model M0, before any real fit: what its priors generate, and whether fitting can recover what generated the data", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.93)
fig.savefig(f"{SP}/out/market0_before.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_before.png")
# ---------------------------------------------------------------- figure B: against data
H = pd.read_parquet(f"{SP}/out/market0_rows.parquet"); S = np.load(f"{SP}/out/market0_post.npz"); th = {k: float(np.median(S[k])) for k in M.PN}; tr = H[H.tr]; ho = H[~H.tr]; Z1 = pd.read_parquet(f"{SP}/out/zoom1_rows.parquet")[["nbs", "teff"]]; H = H.join(Z1, how="left")
fig = plt.figure(figsize=(18.5, 13.6)); gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.34); rr = np.linspace(-0.6, 3.2, 300); st = tr.iloc[rng.choice(len(tr), 4000, replace=False)]; sh_ = ho.iloc[rng.choice(len(ho), 2500, replace=False)]; t90 = tdist.ppf(0.95, M.NU)
ax = fig.add_subplot(gs[0, 0]); ax.scatter(st.p / st.L, st["T"], s=4, color=NEUT, alpha=0.5, linewidths=0, label="fitting period"); ax.scatter(sh_.p / sh_.L, sh_["T"], s=4, color=ORANGE, alpha=0.35, linewidths=0, label="held-out year"); g = th["F_T"] + th["K_T"] * norm.cdf((rr - th["m_T"]) / th["s_T"])
ax.fill_between(rr, g - t90 * th["sd_T"], g + t90 * th["sd_T"], color=BLUE, alpha=0.12, linewidth=0); ax.plot(rr, g, color=BLUE, linewidth=2.2, label="supply curve (90% of shocks)"); ax.set_xlim(-0.6, 3.2); ax.set_ylim(0, 45); ax.set_xlabel("price / gas-plus-CO2 cost"); ax.set_ylabel("thermal output, GW"); ax.legend(frameon=False, loc="lower right", fontsize=8, markerscale=2.5); clean(ax)
ttl(ax, f"Domestic supply: {th['F_T']:.1f} GW floor, then {th['K_T']:.0f} GW\ncoming on around price = {th['m_T']:.2f} x cost")
ax = fig.add_subplot(gs[0, 1]); ax.scatter(st.p / st.L, st.X, s=4, color=NEUT, alpha=0.5, linewidths=0); ax.scatter(sh_.p / sh_.L, sh_.X, s=4, color=ORANGE, alpha=0.35, linewidths=0); g = th["x0"] + th["x1"] * (rr - 1); ax.fill_between(rr, g - t90 * th["sd_X"], g + t90 * th["sd_X"], color=BLUE, alpha=0.12, linewidth=0); ax.plot(rr, g, color=BLUE, linewidth=2.2)
ax.set_xlim(-0.6, 3.2); ax.set_ylim(-22, 22); ax.axhline(0, color=INK, linewidth=0.6); ax.set_xlabel("price / gas-plus-CO2 cost"); ax.set_ylabel("net imports, GW"); clean(ax); ttl(ax, f"Foreign supply: {th['x0']:+.1f} GW at price = cost,\n{th['x1']:.0f} GW more per unit of price / cost")
ax = fig.add_subplot(gs[0, 2]); pp = np.linspace(-150, 400, 400); rest = -(th["B"] - th["K_0"] * norm.cdf((th["c_0"] - pp) / th["s_0"]) - th["K_s"] * (pp < 0)); ax.scatter(st.p, st["T"] + st.X - st.rlreal, s=4, color=NEUT, alpha=0.5, linewidths=0); ax.scatter(sh_.p, sh_["T"] + sh_.X - sh_.rlreal, s=4, color=ORANGE, alpha=0.35, linewidths=0); ax.plot(pp, rest, color=BLUE, linewidth=2.2)
ax.set_xlim(-150, 400); ax.set_ylim(-30, 8); ax.set_xlabel("price, EUR/MWh"); ax.set_ylabel("thermal + imports - residual load, GW"); clean(ax); ttl(ax, "Rest of the system responds to the EUR price;\nthe held-out year sits off the curve")
# implied price-against-load curve over level 1's
ax = fig.add_subplot(gs[0, 3]); rl = np.linspace(-15, 68, 160); Zl = np.load(f"{SP}/out/zoom1_post.npz"); ax.scatter(st.rlreal, st.p / st.L, s=4, color=NEUT, alpha=0.4, linewidths=0)
for Lv, col, lab in ((float(tr.L.median()), BLUE, f"market model, cost {tr.L.median():.0f} (fitting period)"), (float(ho.L.median()), ORANGE, f"market model, cost {ho.L.median():.0f} (held-out year)")):
    p0, _, _ = M.clear(th, rl, np.full(len(rl), Lv)); ax.plot(rl, p0 / Lv, color=col, linewidth=2.2, label=lab)
ax.plot(Zl["K"], np.median(Zl["kn"], 0), color=INK, linewidth=1.6, linestyle=(0, (5, 2)), label="level 1 (one curve, any cost)"); ax.set_xlim(-15, 68); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("residual load, GW"); ax.set_ylabel("price / gas-plus-CO2 cost"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax)
ttl(ax, "Implied price against load (shocks at zero):\non level 1 from 15 to 60 GW, apart below")
# replicated against observed spread by load band (fitting period), prices and thermal output
pr, Tr, Xr = M.simulate(th, tr.rlreal.values, tr.L.values, rng); bands = np.arange(-20, 61, 10); cen = bands[:-1] + 5; qs = [0.05, 0.25, 0.5, 0.75, 0.95]
for c_, (obs, rep, yl, lim, nm) in enumerate(((tr.p.values / tr.L.values, pr / tr.L.values, "price / gas-plus-CO2 cost", (-1.0, 2.6), "prices"), (tr["T"].values, Tr, "thermal output, GW", (0, 42), "thermal output"), (tr.X.values, Xr, "net imports, GW", (-20, 20), "net imports"))):
    ax = fig.add_subplot(gs[1, c_]); qo = np.array([np.quantile(obs[(tr.rlreal.values >= a) & (tr.rlreal.values < a + 10)], qs) if ((tr.rlreal.values >= a) & (tr.rlreal.values < a + 10)).sum() > 60 else [np.nan] * 5 for a in bands[:-1]]); qr = np.array([np.quantile(rep[(tr.rlreal.values >= a) & (tr.rlreal.values < a + 10)], qs) if ((tr.rlreal.values >= a) & (tr.rlreal.values < a + 10)).sum() > 60 else [np.nan] * 5 for a in bands[:-1]])
    ax.fill_between(cen, qr[:, 0], qr[:, 4], color=BLUE, alpha=0.10, linewidth=0, label="simulated: 5-95%"); ax.fill_between(cen, qr[:, 1], qr[:, 3], color=BLUE, alpha=0.2, linewidth=0, label="simulated: 25-75%"); ax.plot(cen, qr[:, 2], color=BLUE, linewidth=2)
    for i in range(5): ax.plot(cen, qo[:, i], "-o", color=INK, linewidth=0.9, markersize=3.5, label="observed 5, 25, 50, 75, 95%" if i == 0 else None)
    ax.set_ylim(*lim); ax.set_xlabel("residual load, GW (10 GW bands)"); ax.set_ylabel(yl); clean(ax); ttl(ax, f"Simulated against observed, {nm},\nfitting period")
    if c_ == 0: ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    out[f"median absolute gap of the five quantiles, {nm}"] = float(np.nanmedian(np.abs(qo - qr)))
ax = fig.add_subplot(gs[1, 3]); obs_s = float((np.abs(tr.p.values) <= M.EPS).mean()); reps = [float((M.simulate(th, tr.rlreal.values, tr.L.values, rng)[0] == 0).mean()) for _ in range(40)]; obs_n = float((tr.p.values < -M.EPS).mean()); repn = [float((M.simulate(th, tr.rlreal.values, tr.L.values, rng)[0] < -M.EPS).mean()) for _ in range(40)]
for i, (nm, o, r_) in enumerate((("on the shelf (price zero)", obs_s, reps), ("below zero", obs_n, repn))):
    lo, md, hi = np.percentile(r_, [5, 50, 95]); ax.plot([100 * lo, 100 * hi], [i, i], color=BLUE, linewidth=5, solid_capstyle="round", alpha=0.4, label="simulated" if i == 0 else None); ax.scatter([100 * md], [i], s=40, color=BLUE, zorder=3, edgecolor=SURF, linewidth=0.8); ax.scatter([100 * o], [i], s=64, color=INK, zorder=4, edgecolor=SURF, linewidth=0.8, label="observed" if i == 0 else None); out[f"share of hours {nm}: observed / simulated"] = (o, float(md))
ax.set_yticks([0, 1]); ax.set_yticklabels(["on the shelf\n(price zero)", "below zero"]); ax.set_ylim(-0.7, 1.9); ax.set_xlim(0, None); ax.set_xlabel("% of hours, fitting period"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax); ttl(ax, "Hours at and below zero: does the simulated\nmarket have as many as the real one?")
# shocks against what is left out, with envelopes from whole-week shifts
def env(x, r, edges, nrep=150):
    c_ = pd.cut(x, edges)
    def bm(res):
        g_ = pd.Series(res).groupby(c_, observed=False); return np.where(g_.count() >= 80, g_.median(), np.nan)
    nw = len(r) // 168; e_ = np.array([bm(np.roll(r, int(k) * 168)) for k in rng.integers(4, nw - 4, nrep)]); return bm(r), np.nanpercentile(e_, 5, axis=0), np.nanpercentile(e_, 95, axis=0)
Hh = H[H.tr].copy(); Hh["hour"] = Hh.index.hour; Hh["dmean"] = Hh.groupby(Hh.index.date)["p"].transform("mean"); Hh["devp"] = (Hh.p - Hh.dmean) / Hh.L
panels = (("thermal shock against temperature", "teff", "eT", np.arange(-6, 31, 4), "temperature, C"), ("thermal shock against the hour's price minus the day's mean", "devp", "eT", np.array([-1.5, -0.6, -0.35, -0.2, -0.1, 0, 0.1, 0.2, 0.35, 0.6, 1.5]), "(price - day's mean price) / cost"), ("import shock against the neighbours' residual load", "nbs", "eX", np.arange(-45, 46, 10), "neighbours' residual load, GW from their mean"), ("demand shock against the hour of day", "hour", "d", np.arange(-0.5, 24, 1), "hour of day"))
for i, (nm, xc, rc, edges, xl) in enumerate(panels):
    ax = fig.add_subplot(gs[2, i]); m = Hh[xc].notna() & Hh[rc].notna(); o, lo, hi = env(Hh[xc][m].values, Hh[rc][m].values, edges); cen_ = (edges[:-1] + edges[1:]) / 2; ax.fill_between(cen_, lo, hi, color=NEUT, alpha=0.3, linewidth=0, label="what whole-week shifts give by chance"); ax.plot(cen_, o, "-o", color=BLUE, linewidth=2, markersize=4, label="observed (median shock)"); ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel(xl); ax.set_ylabel("shock, GW"); ax.set_ylim(-9, 9); clean(ax); out[f"{nm}: range / bins outside the envelope"] = (float(np.nanmax(o) - np.nanmin(o)), float(np.nansum((o > hi) | (o < lo))), float(np.sum(np.isfinite(o)))); ttl(ax, nm[0].upper() + nm[1:].replace(" against ", "\nagainst "))
    if i == 0: ax.legend(frameon=False, loc="lower left", fontsize=7.5)
fig.suptitle("Market model M0 against data: its three parts with their quantities, the price it implies, and where its shocks are not shocks", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.93)
fig.savefig(f"{SP}/out/market0_data.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0_data.png")
for k_, v in out.items(): print(f"  {k_}: " + (", ".join(f"{float(x):.3f}" for x in v) if isinstance(v, tuple) else f"{v:.3f}"))
