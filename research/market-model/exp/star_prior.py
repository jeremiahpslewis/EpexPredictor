"""Prior predictive check of the coupled-market model: draw every parameter from its prior, clear the market on the
actual residual loads and gas prices, and compare what comes out with the data on quantities one can recognise.
usage: star_prior.py PERIOD TAG [NDRAWS]"""
import sys, numpy as np, jax, jax.numpy as jnp, numpyro
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import star as S
period, tag = sys.argv[1], sys.argv[2]; ND = int(sys.argv[3]) if len(sys.argv) > 3 else 30
I = np.load(f"{S.lc.SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96
lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4; H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1)
sub = np.arange(0, len(H["P"]), 7)                                         # every 7th hour: all hours of the day, all seasons
ds = {k: (v[sub] if isinstance(v, np.ndarray) and v.shape[:1] == (len(H["P"]),) else v) for k, v in d.items()}; Hs = {k: v[sub] for k, v in H.items()}
dj = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
pri = numpyro.infer.Predictive(S.model, num_samples=ND)(jax.random.PRNGKey(1), dj, P=jnp.asarray(np.nan_to_num(Hs["P"])))
rl = Hs["rl"][:, 0]; ok = ds["ok"][:, 0]; P = Hs["P"]; sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
def summ(p0, pnl, ex, lig, gas):
    q = np.nanpercentile(p0 / 10, [5, 50, 95]); return (f"price 5/50/95% {q[0]:6.1f} {q[1]:6.1f} {q[2]:6.1f} ct | negative {100 * np.nanmean(p0 < 0):3.0f}% | above 20 ct {100 * np.nanmean(p0 > 200):3.0f}% | median at RL<15 / 25-35 / >45 GW: {np.nanmedian(p0[rl < 15]) / 10:5.1f} {np.nanmedian(p0[(rl >= 25) & (rl < 35)]) / 10:5.1f} {np.nanmedian(p0[rl > 45]) / 10:5.1f} | "
                                                   f"within 0.15 ct of NL {100 * np.nanmean(np.abs(pnl - p0) < 1.5):3.0f}% | net exports 5..95% {np.nanpercentile(ex, 5):+5.1f}..{np.nanpercentile(ex, 95):+5.1f} GW | lignite 5..95% {np.nanpercentile(lig, 5):4.1f}..{np.nanpercentile(lig, 95):4.1f} | gas {np.nanpercentile(gas, 5):4.1f}..{np.nanpercentile(gas, 95):4.1f}")
print("data:      " + summ(P[:, 0], P[:, 2], Hs["x"].sum(1), Hs["g"][:, 0], Hs["g"][:, 2]))
curves = []; e = np.arange(-10, 71, 5.0); rows = []
for i in range(ND):
    q = S.unpack({k: np.asarray(v[i]) for k, v in pri.items()}); pe, ex = S.equilibrium(q, ds); cap, mc, wd, sl, shift, ct = S.build(q, ds, np)
    blk = cap[:, 0, :] * sg((pe[:, :1] - mc[:, 0, :]) / wd[0][None]); lig = (S.floors(q, ds, np) * np.ones((len(rl), 1)))[:, 0] + blk[:, 1]; gas = (S.floors(q, ds, np) * np.ones((len(rl), 1)))[:, 2] + blk[:, 3] + blk[:, 4]
    rows.append(summ(pe[:, 0], pe[:, 2], ex.sum(1), lig, gas)); curves.append([np.median(pe[(rl >= a) & (rl < b), 0]) / 10 if ((rl >= a) & (rl < b)).sum() > 10 else np.nan for a, b in zip(e[:-1], e[1:])])
for i in range(min(ND, 8)): print(f"prior {i + 1:2d}:  " + rows[i])
C = np.array(curves); mid = 0.5 * (e[:-1] + e[1:]); obs = [np.nanmedian(P[(rl >= a) & (rl < b), 0]) / 10 if ((rl >= a) & (rl < b)).sum() > 10 else np.nan for a, b in zip(e[:-1], e[1:])]
print(f"across {ND} prior draws, median German price at RL 25-35 GW: 5%..95% of draws {np.nanpercentile(C[:, 7], 5):.1f}..{np.nanpercentile(C[:, 7], 95):.1f} ct (data {obs[7]:.1f}); at RL > 50: {np.nanpercentile(C[:, 12], 5):.1f}..{np.nanpercentile(C[:, 12], 95):.1f} (data {obs[12]:.1f}); at RL < 5: {np.nanpercentile(C[:, 2], 5):.1f}..{np.nanpercentile(C[:, 2], 95):.1f} (data {obs[2]:.1f})")
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, a = plt.subplots(figsize=(7.2, 4.6))
for i in range(ND): a.plot(mid, C[i], color=BLUE, linewidth=1, alpha=0.45, label="one draw from the prior" if i == 0 else None)
a.plot(mid, obs, "o", color=ORANGE, markersize=6, label="observed"); a.set_ylim(-10, 45); a.set_xlabel("German residual load, GW"); a.set_ylabel("median price, ct/kWh"); a.grid(True, color="#e9e8e3", linewidth=0.6); a.set_axisbelow(True); a.legend(frameon=False, loc="upper left")
for s_ in ("top", "right"): a.spines[s_].set_visible(False)
a.set_title("Before any fitting: German price against residual load under the priors", loc="left", fontsize=10, fontweight="bold", color=INK)
fig.tight_layout(); out = f"{S.lc.SP}/out/star_prior_{tag}.png"; fig.savefig(out, dpi=110); print("figure:", out)
