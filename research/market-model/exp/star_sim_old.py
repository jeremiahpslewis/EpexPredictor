"""Simulated-data check: simulate prices, output by fuel and exchanges from the fitted model, refit from scratch,
and see which parameters come back. usage: star_sim.py RUN PERIOD TAG"""
import sys, numpy as np
import star as S
run, period, tag = sys.argv[1:4]; SP = S.lc.SP; z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); raw = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}; q = S.unpack(raw)
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96
lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4; H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1)
rng = np.random.default_rng(11); pe, e = S.equilibrium(q, d); n = len(pe); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
t = lambda df, size: rng.standard_t(df, size)
Ps = pe + np.exp(q["s"])[None] * t(max(float(q["nu"]), 2.1), pe.shape)                                   # prices: equilibrium plus the fitted price noise
cap, mc, wd, sl, shift, ct = S.build(q, d, np); blk = cap[:, 0, :] * sg((Ps[:, :1] - mc[:, 0, :]) / wd[0][None])
gs = np.stack([q["floor"][0] + blk[:, 1], q["floor"][1] + blk[:, 2], q["floor"][2] + blk[:, 3] + blk[:, 4]], 1) + q["s_g"][None] * t(4, (n, 3))
xs = S.exch(q, Ps[:, 1:] - Ps[:, :1], np) + q["s_x"][None] * t(4, (n, S.Z - 1))
Ps = np.where(d["ok"], Ps, np.nan); gs = np.where(d["gok"], gs, np.nan); xs = np.where(d["xok"], xs, np.nan)
q2, raw2, d2, H2, loss = S.fit(d0, seed=7, sim={"P": Ps, "g": gs, "x": xs}); np.savez_compressed(f"{SP}/out/star_{run}/{tag}_simfit.npz", **{f"raw_{k}": v for k, v in raw2.items()})
np.set_printoptions(precision=1, suppress=True, linewidth=220)
def line(lab, a, b): a, b = np.ravel(a), np.ravel(b); print(f"  {lab:<44} true {a}\n  {'':<44} back {b}")
print("simulated from the fitted model, refitted from the default start. Parameters used to simulate, and what came back:")
line("German block capacities (free part) [GW]", q["capf"][0], q2["capf"][0]); line("German marginal-cost intercepts", q["alpha"][0], q2["alpha"][0]); line("German gas links", q["beta"][0], q2["beta"][0]); line("must-run floors lignite, coal, gas", q["floor"], q2["floor"])
line("availability multipliers (lignite, coal, FR nuclear)", [q["kmeas"][0, 1], q["kmeas"][0, 2], q["kmeas"][1, 0]], [q2["kmeas"][0, 1], q2["kmeas"][0, 2], q2["kmeas"][1, 0]])
line("export capacities", q["Ce"], q2["Ce"]); line("import capacities", q["Ci"], q2["Ci"]); line("exchange offsets", q["m"], q2["m"]); line("exchange price scales", q["kappa"], q2["kappa"])
for zn in ("fr", "nl", "pl"): i = S.ZN.index(zn); line(f"{zn} block capacities", q["capf"][i], q2["capf"][i]); line(f"{zn} cost intercepts", q["alpha"][i], q2["alpha"][i])
c1 = S.build(q, d, np)[5]; c2 = S.build(q2, d, np)[5]; print(f"  common cost level: correlation of the two paths {np.corrcoef(c1, c2)[0, 1]:+.2f}; range true {c1.min():+.0f}..{c1.max():+.0f}, back {c2.min():+.0f}..{c2.max():+.0f}")
rel = lambda a, b: np.median(np.abs(np.ravel(b) - np.ravel(a)) / (np.abs(np.ravel(a)) + 1e-9))
print(f"  median relative error: all block capacities {100 * rel(q['capf'], q2['capf']):.0f}%, German ones {100 * rel(q['capf'][0], q2['capf'][0]):.0f}%; export/import capacities {100 * rel(np.r_[q['Ce'], q['Ci']], np.r_[q2['Ce'], q2['Ci']]):.0f}%; cost intercepts, mean abs error {np.abs(q2['alpha'] - q['alpha']).mean():.1f} EUR/MWh (German {np.abs(q2['alpha'][0] - q['alpha'][0]).mean():.1f})")
pe2, _ = S.equilibrium(q2, d); print(f"  German equilibrium price implied by the two parameter sets on the same inputs: mean abs difference {np.abs(pe2[:, 0] - pe[:, 0]).mean() / 10:.2f} ct")
