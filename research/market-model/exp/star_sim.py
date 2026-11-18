"""Simulated-data check in generative order: take a fitted parameter set as the truth, draw the shocks the model assumes
(balance of each zone, exchange per border, pumped storage, output by fuel), clear the market, read off prices and
quantities, refit from a fresh start, and compare what comes back with what went in.
usage: star_sim.py RUN PERIOD TAG [SIMSEED]"""
import sys, os, time, numpy as np
import star as S
run, period, tag = sys.argv[1], sys.argv[2], sys.argv[3]; ss = int(sys.argv[4]) if len(sys.argv) > 4 else 11
SP = S.lc.SP; z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); raw = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}; q = S.unpack(raw)
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96
lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4; H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1); n = len(H["P"])
rng = np.random.default_rng(ss); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60))); tdraw = lambda nu, size: rng.standard_t(nu, size)
# shocks, with the fitted scales
u = tdraw(10, (n, S.Z)) * q["s_q"][None]; ex = tdraw(4, (n, S.Z - 1)) * q["s_x"][None]; eps = tdraw(4, n) * float(q["s_ps"]); eg = tdraw(4, (n, 3)) * q["s_g"][None]
ex = np.clip(ex, -3 * q["s_x"][None] * 3, 3 * q["s_x"][None] * 3); eg = np.clip(eg, -9 * q["s_g"][None], 9 * q["s_g"][None]); eps = np.clip(eps, -9 * float(q["s_ps"]), 9 * float(q["s_ps"]))
rl = d["rl"].copy(); rl[:, 0] += u[:, 0] + ex.sum(1) - eps; rl[:, 1:] += u[:, 1:] - ex
t0 = time.time(); pe, e = S.equilibrium(q, d, rl=rl); pe0, _ = S.equilibrium(q, d)
cap, mc, wd, sl, shift, ct = S.build(q, d, np); pdm = np.column_stack([S.day_mean(pe[:, i], d["day"]) for i in range(S.Z)])
blk = cap[:, 0, :] * sg((pe[:, :1] + q["lam"][0][None] * (pdm[:, :1] - pe[:, :1]) - mc[:, 0, :]) / wd[0][None]); FL = S.floors(q, d, np) * np.ones((n, 1))
g = np.stack([FL[:, 0] + blk[:, 1], FL[:, 1] + blk[:, 2], FL[:, 2] + blk[:, 3] + blk[:, 4]], 1) + eg; x = e + ex; ps = S.equilibrium.ps + eps
okd = np.isfinite(H["P"]); P = np.where(okd, pe, np.nan); gm = np.where(np.isfinite(H["g"]), g, np.nan); xm = np.where(np.isfinite(H["x"]), x, np.nan); psm = np.where(np.isfinite(H["ps"]), ps, np.nan)
print(f"simulated world from {run} {tag}: {n} hours, cleared in {time.time() - t0:.0f}s. German price: data mean {np.nanmean(H['P'][:, 0]):.0f} sd {np.nanstd(H['P'][:, 0]):.0f}; simulated mean {np.nanmean(P[:, 0]):.0f} sd {np.nanstd(P[:, 0]):.0f}; "
      f"shocks move the German price by {np.mean(np.abs(pe[:, 0] - pe0[:, 0])) / 10:.2f} ct on average; share of hours within 0.5 EUR/MWh of each neighbour: data " + " ".join(f"{100 * np.nanmean(np.abs(H['P'][:, i] - H['P'][:, 0]) <= 0.5):.0f}" for i in range(1, S.Z)) + " | simulated " + " ".join(f"{100 * np.mean(np.abs(pe[:, i] - pe[:, 0]) <= 0.5):.0f}" for i in range(1, S.Z)), flush=True)
t0 = time.time(); q2, raw2, d2, H2, loss = S.fit(d0, seed=5, sim={"P": P, "g": gm, "x": xm, "ps": psm}); pe2, _ = S.equilibrium(q2, d2, rl=rl)
os.makedirs(f"{SP}/out/star_{run}_sim", exist_ok=True); np.savez_compressed(f"{SP}/out/star_{run}_sim/{tag}_sim{ss}.npz", **{f"raw_{k}": v for k, v in raw2.items()}, P=P.astype(np.float32), peq=pe2.astype(np.float32))
print(f"refit in {time.time() - t0:.0f}s, loss {loss:.1f}. Recovered model, cleared with the same shocks, against the simulated German price: mean abs difference {np.mean(np.abs(pe2[:, 0] - pe[:, 0])) / 10:.2f} ct")
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def row(name, a, b, pri=None): print(f"  {name:38s} true {np.round(np.atleast_1d(a), 2)}  recovered {np.round(np.atleast_1d(b), 2)}" + (f"  prior centre {np.round(np.atleast_1d(pri), 2)}" if pri is not None else ""))
cap2, mc2, wd2, sl2, shift2, ct2 = S.build(q2, d2, np)
print("German blocks (low, lignite, hard coal, combined cycle, peakers):")
row("mean capacity GW", cap[:, 0, :5].mean(0), cap2[:, 0, :5].mean(0)); row("mean marginal cost EUR/MWh", mc[:, 0, :5].mean(0), mc2[:, 0, :5].mean(0)); row("width", wd[0, :5], wd2[0, :5], S.WD0[:5])
row("gas link (heat rate)", q["beta"][0, :5], q2["beta"][0, :5], [0, S.BA0[0], S.BB0[0], 1.85, 2.7]); row("inertia weight", q["lam"][0, 1:5], q2["lam"][0, 1:5], [0.5] * 4)
row("must-run on a warm day", q["floor"], q2["floor"], [2.5, 0.6, 1.5]); row("must-run per heating degree", q["f1"], q2["f1"], S.F10)
row("elastic slope", q["sl"][0], q2["sl"][0], 0.5); row("pumped storage GW, price scale", [q["p_ps"], q["k_ps"]], [q2["p_ps"], q2["k_ps"]], [5, 25])
print("neighbours (fr nl be at pl cz ch dk):")
row("export capacity", q["Ce"], q2["Ce"], S.NTC); row("import capacity", q["Ci"], q2["Ci"], S.NTC); row("price-difference scale", q["kappa"], q2["kappa"], [10] * 8); row("offset", q["m"], q2["m"], [0] * 8)
row("wind effect on export capacity", q["ae"][0], q2["ae"][0], [0] * 8); row("sun effect on export capacity", q["ae"][1], q2["ae"][1], [0] * 8)
row("heat-led output per degree (de first)", q["hz"], q2["hz"], S.HZ0); row("combined-cycle heat rate", q["beta"][1:, 3], q2["beta"][1:, 3], [1.85] * 8); row("combined-cycle capacity", cap[:, 1:, 3].mean(0), cap2[:, 1:, 3].mean(0))
row("low-cost block capacity", cap[:, 1:, 0].mean(0), cap2[:, 1:, 0].mean(0)); row("balance noise scale GW (de first)", q["s_q"], q2["s_q"])
print(f"latent cost level: true range {ct.min():+.0f}..{ct.max():+.0f}, recovered {ct2.min():+.0f}..{ct2.max():+.0f}; correlation of the weekly paths {np.corrcoef(ct, ct2)[0, 1]:+.2f}; mean abs difference {np.mean(np.abs(ct - ct2)):.1f} EUR/t")
