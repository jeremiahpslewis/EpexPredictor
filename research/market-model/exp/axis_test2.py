"""Regional axes: how tightly does the German price follow German residual load plus a weight on the neighbours', and plus other known shifters?
Same measure as axis_test.py (within quarter, 2 GW bins, mean absolute deviation from the bin median). Also a linear index fitted per axis family."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); loc = D.index.tz_convert("Europe/Berlin"); NB = ["fr", "nl", "be", "at", "pl", "cz", "ch", "dk"]
q = (loc.year * 4 + (loc.month - 1) // 3).values; p = D["p_de"].values; half = np.array([f"{y}H{1 if m <= 6 else 2}" for y, m in zip(loc.year, loc.month)]); keep = (loc >= "2024-07-01")
def mad(x, width=2.0):
    ok = np.isfinite(x) & np.isfinite(p) & keep; dev = np.full(len(x), np.nan); b = np.floor(x / width)
    for qq in np.unique(q[ok]):
        for bb in np.unique(b[ok & (q == qq)]):
            k = ok & (q == qq) & (b == bb)
            if k.sum() >= 8: dev[k] = np.abs(p[k] - np.median(p[k])) / 10
    return np.nanmean(dev), [np.nanmean(dev[half == h]) for h in ("2024H2", "2025H1", "2025H2", "2026H1", "2026H2")]
rl = D["rl_de"].values; nb = sum(D[f"rl_{z}"].values for z in NB); nbm = np.nanmean(nb)
print(f"{'axis (GW)':66s} {'MAD ct':>7s}   2024H2 2025H1 2025H2 2026H1 2026H2")
for w in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
    m, h = mad(rl + w * (nb - nbm), 2.0 * (1 + w)); print(f"{'German residual load + %.1f x (sum of eight neighbours)' % w:66s} {m:7.2f}   " + "  ".join(f"{v:.2f}" for v in h))
# a fitted linear index per quarter is too flexible for this measure; instead add shifters one at a time to the best simple axis
best = rl + 0.2 * (nb - nbm); dm = pd.Series(rl).groupby(np.asarray(loc.normalize())).transform("mean").values
for name, x in (("... minus German pumped storage net output (measured)", best - (D["g_de_psgen"] + D["g_de_pump"]).values), ("... plus scheduled net exports (measured)", best - D["x_sum"].values),
                ("... with 0.3 x (residual load - its day mean) taken out (storage and inertia)", best - 0.3 * (rl - dm)), ("... with 0.5 x that", best - 0.5 * (rl - dm)),
                ("... per neighbour weights: FR 0.1, NL 0.4, BE 0.3, AT 0.2, PL 0.2, CZ 0.5, CH 0.2, DK 0.3", rl + sum(w_ * (D[f'rl_{z}'].values - np.nanmean(D[f'rl_{z}'].values)) for z, w_ in zip(NB, (0.1, 0.4, 0.3, 0.2, 0.2, 0.5, 0.2, 0.3))))):
    m, h = mad(x, 2.4); print(f"{name:66s} {m:7.2f}   " + "  ".join(f"{v:.2f}" for v in h))
print("\nexogenous supply shifters added to German residual load + 0.3 x neighbours (availability = trailing 14-day maximum of output, one day late):")
av = lambda c: D[c].rolling(14 * 24, min_periods=7 * 24).max().shift(24).values
base = rl + 0.3 * (nb - nbm); frn = av("g_fr_nuclear"); sol = av("g_de_lignite") + av("g_de_coal"); gasav = av("g_de_gas")
import live_common as lc
T = lc.load()["DE"]; temp = T[[c for c in T.columns if c.startswith("temp_")]].mean(axis=1).resample("1h").mean().reindex(D.index).values; hdd = np.maximum(13.0 - pd.Series(temp).rolling(48, min_periods=12).mean().values, 0.0)
for name, x in (("base", base), ("- 0.3 x French nuclear availability", base - 0.3 * (frn - np.nanmean(frn))), ("- 1.0 x French nuclear availability", base - 1.0 * (frn - np.nanmean(frn))),
                ("- 0.5 x German lignite + hard coal availability", base - 0.5 * (sol - np.nanmean(sol))), ("- 1.0 x German lignite + hard coal availability", base - 1.0 * (sol - np.nanmean(sol))),
                ("- 0.4 GW per heating degree (heat-led must-run)", base - 0.4 * hdd), ("- 0.8 GW per heating degree", base - 0.8 * hdd),
                ("- 0.5 x solid availability - 0.4 per heating degree - 0.3 x French nuclear", base - 0.5 * (sol - np.nanmean(sol)) - 0.4 * hdd - 0.3 * (frn - np.nanmean(frn)))):
    m, h = mad(x, 2.6); print(f"{name:76s} {m:7.2f}   " + "  ".join(f"{v:.2f}" for v in h))
# across quarters: does the relative price (price / gas-fired cost) line up on the best axis without the within-quarter restriction?
E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); L = (1.85 * D["gas"] + 0.37 * ed.reindex(loc.normalize().tz_localize(None)).values).values
def mad_all(x, y, width=2.6):
    ok = np.isfinite(x) & np.isfinite(y) & keep; dev = np.full(len(x), np.nan); b = np.floor(x / width)
    for bb in np.unique(b[ok]):
        k = ok & (b == bb)
        if k.sum() >= 20: dev[k] = np.abs(y[k] - np.median(y[k]))
    return dev
print("\nno quarter restriction (one curve for Jul 2024 - Nov 2026): mean absolute deviation from the bin median")
for name, x in (("German residual load", rl), ("regional index (base)", base), ("regional index - 0.5 x solid availability - 0.4 per heating degree - 0.3 x French nuclear", base - 0.5 * (sol - np.nanmean(sol)) - 0.4 * hdd - 0.3 * (frn - np.nanmean(frn)))):
    d1 = mad_all(x, p); d2 = mad_all(x, p / L); print(f"  {name:90s} price {np.nanmean(d1) / 10:.2f} ct | price / gas-fired cost {np.nanmean(d2):.3f} (= {np.nanmean(d2 * L) / 10:.2f} ct); by half-year, relative, in ct: " + " ".join(f"{np.nanmean((d2 * L)[half == h]) / 10:.2f}" for h in ("2024H2", "2025H1", "2025H2", "2026H1", "2026H2")))
