"""Check of the storage mechanism on the errors of the structural curve (no refit).
Per local day: a storage fleet (power P GW, energy E GWh charged, round-trip efficiency ETA) charges in the cheapest hours
and discharges in the dearest, as long as  price(dearest level) * ETA > price(cheapest level)  on the fitted curve.
Its net demand shifts the effective residual load; D = curve(x + shift) - curve(x) is what the mechanism predicts the
curve gets wrong. Compare with the curve's actual error r."""
import sys, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
P, E, ETA = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
def curve_fns(z):
    g = z["grid"].astype(float); b = z["b_curve"].astype(float); h = z["h_curve"].astype(float); db = np.diff(b) / np.diff(g); dh = np.diff(h) / np.diff(g)
    def val(x, gas):
        xc = np.clip(x, g[0], g[-1]); return (np.interp(xc, g, b) + gas * np.interp(xc, g, h) + np.minimum(x - g[0], 0) * db[0] + np.maximum(x - g[-1], 0) * (db[-1] + gas * dh[-1])) / 10
    return val
def level(x, q, up):
    """level L with sum(min(P, (L - x)+)) = q (valley filling, up) or sum(min(P, (x - L)+)) = q (peak shaving)"""
    lo, hi = x.min() - P - 1, x.max() + P + 1
    for _ in range(40):
        L = 0.5 * (lo + hi); s = np.minimum(P, np.maximum(L - x, 0)).sum() if up else np.minimum(P, np.maximum(x - L, 0)).sum()
        if (s < q) == up: lo = L
        else: hi = L
    return 0.5 * (lo + hi)
def dispatch(x, gas, val):
    def sol(q):
        Ll = level(x, q, True); Lh = level(x, ETA * q, False); return Ll, Lh
    lo, hi = 0.0, E
    Ll, Lh = sol(hi)
    if not (val(np.array([Lh]), gas)[0] * ETA > val(np.array([Ll]), gas)[0] and Lh > Ll):
        for _ in range(25):
            q = 0.5 * (lo + hi); Ll, Lh = sol(q)
            if val(np.array([Lh]), gas)[0] * ETA > val(np.array([Ll]), gas)[0] and Lh > Ll: lo = q
            else: hi = q
        Ll, Lh = sol(lo)
    return np.minimum(P, np.maximum(Ll - x, 0)) - np.minimum(P, np.maximum(x - Lh, 0))
for tag in ("2026-02-26", "2026-08-25", "2026-11-06"):
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); zz = np.load(f"{SP}/out/struct_dg/{tag}.npz"); val = curve_fns(zz)
    df = pd.DataFrame({"r": z["r"] / 10, "hour": z["hour"], "day": z["day"], "x": z["x"], "gas": z["gas"], "m": z["m"] / 10}); lo, hi = np.percentile(df.r, [1, 99]); df["r"] = df.r.clip(lo, hi)
    sh = np.zeros(len(df))
    for d, g in df.groupby("day"):
        if len(g) >= 23: sh[df.index.get_indexer(g.index)] = dispatch(g.x.to_numpy(), float(g.gas.iloc[0]), val)
    df["shift"] = sh; df["D"] = val(df.x.to_numpy() + sh, df.gas.to_numpy()) - val(df.x.to_numpy(), df.gas.to_numpy())
    df["season"] = np.where((pd.to_datetime(df.day, unit="D").dt.month.isin([11, 12, 1, 2])), "Nov-Feb", np.where(pd.to_datetime(df.day, unit="D").dt.month.isin([5, 6, 7, 8]), "May-Aug", "spring/autumn"))
    night = df.hour < 8; act = df.D.abs() > 0.3
    print(f"\nfit of {tag}: storage active (|predicted effect| > 0.3 ct) in {100 * act.mean():.0f}% of hours, {100 * (act & night).sum() / night.sum():.0f}% of night hours; mean shift when charging {df.loc[df['shift'] > 0.5, 'shift'].mean():.1f} GW")
    for lab, m in (("night, storage charging", night & (df.D > 0.3)), ("night, storage idle", night & ~act), ("day, storage charging", ~night & (df.D > 0.3)), ("any, storage discharging", df.D < -0.3)):
        s = df[m]; 
        if len(s) < 30: continue
        beta = float((s.r * s.D).sum() / (s.D ** 2).sum()) if (s.D ** 2).sum() > 0 else float("nan")
        print(f"   {lab:<26} n {len(s):5d}  curve's error: mean {s.r.mean():+.2f}  predicted by storage: mean {s.D.mean():+.2f}  slope of error on prediction {beta:+.2f}  MAE {s.r.abs().mean():.2f} -> after subtracting the prediction {(s.r - s.D).abs().mean():.2f}, half of it {(s.r - 0.5 * s.D).abs().mean():.2f}")
    s = df[night & (df.D > 0.3)]
    print("   night & charging by season: " + "; ".join(f"{k}: n {len(v)}, error {v.r.mean():+.2f}, predicted {v.D.mean():+.2f}" for k, v in s.groupby("season")))
    s = df[night & (df.x < 15)]; print("   night & x < 15 GW by season: " + "; ".join(f"{k}: n {len(v)}, error {v.r.mean():+.2f}, predicted {v.D.mean():+.2f}, share with storage charging {100 * (v.D > 0.3).mean():.0f}%" for k, v in s.groupby("season")))
