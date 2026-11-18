"""Which quantity is the German price the tightest function of?  For each candidate axis: within each quarter, bin the quantity (2 GW bins) and take the
mean absolute deviation of the price from its bin median, in ct/kWh and for the price relative to gas-fired cost. Lower = tighter. Hourly."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index)
loc = D.index.tz_convert("Europe/Berlin"); ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); D["eua"] = ed.reindex(loc.normalize().tz_localize(None)).values; D["L"] = 1.85 * D["gas"] + 0.37 * D["eua"]
S = {n: pd.read_parquet(f"{SP}/ext/smard/series/{n}_DE_hour.parquet")[n].reindex(D.index) / 1000.0 for n in ("da_other", "da_total")}
X = pd.DataFrame({"published residual load": D["rl_de"], "realised residual load": D["rlreal_de"], "realised thermal (lignite + coal + gas)": D["g_de_lignite"] + D["g_de_coal"] + D["g_de_gas"],
                  "realised thermal + other + pumped generation": D["g_de_lignite"] + D["g_de_coal"] + D["g_de_gas"] + D["g_de_other"] + D["g_de_psgen"],
                  "scheduled day-ahead, other than wind and solar (SMARD)": S["da_other"]})
q = (loc.year * 4 + (loc.month - 1) // 3).values; p = D["p_de"].values; r = (D["p_de"] / D["L"]).values
print(f"{'axis':58s} {'MAD price ct':>12s} {'MAD price/L':>12s}   by half-year (ct): 2024H2 2025H1 2025H2 2026H1 2026H2")
half = np.array([f"{y}H{1 if m <= 6 else 2}" for y, m in zip(loc.year, loc.month)])
for name in X:
    x = X[name].values; ok = np.isfinite(x) & np.isfinite(p) & np.isfinite(r) & (loc >= "2024-07-01"); dev = np.full(len(x), np.nan); devr = np.full(len(x), np.nan); b = np.floor(x / 2.0)
    for qq in np.unique(q[ok]):
        for bb in np.unique(b[ok & (q == qq)]):
            k = ok & (q == qq) & (b == bb)
            if k.sum() >= 8: dev[k] = np.abs(p[k] - np.median(p[k])) / 10; devr[k] = np.abs(r[k] - np.median(r[k]))
    print(f"{name:58s} {np.nanmean(dev):12.2f} {np.nanmean(devr):12.3f}   " + "  ".join(f"{np.nanmean(dev[half == h]):.2f}" for h in ("2024H2", "2025H1", "2025H2", "2026H1", "2026H2")))
k = np.isfinite(X.iloc[:, 4].values) & np.isfinite(X.iloc[:, 2].values); a = X.iloc[:, 4].values[k]; t = X.iloc[:, 2].values[k]; o = (D["g_de_other"] + D["g_de_psgen"]).values[k]
print(f"\nscheduled other-than-wind-and-solar: mean {a.mean():.1f} GW; realised thermal {t.mean():.1f} + other and pumped generation {np.nanmean(o):.1f}; correlation with realised thermal + other {np.corrcoef(a, t + np.nan_to_num(o))[0, 1]:.3f}; mean abs difference {np.mean(np.abs(a - (t + np.nan_to_num(o)))):.2f} GW")
ident = (D["rl_de"] - D["x_sum"]).values[k]; print(f"scheduled quantity against published residual load plus scheduled net exports: correlation {np.corrcoef(a, ident)[0, 1]:.3f}, mean difference {np.mean(a - ident):+.1f} GW, sd of the difference {np.std(a - ident):.2f} GW")
