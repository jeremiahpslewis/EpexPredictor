"""3. Distillation / grey-box experiments (EXPECTATIONS.md K1, K2, R1-R3). Usage: distill_exp.py [K2_PRED.pkl]
Training-period structural forecasts: SP/distill/train11.pkl (variant b, posterior draws as oos11s, records built by
../predict11_build.py for 2026-01-01..2026-08-31, hourly; published TSO residual-load inputs, i.e. E-type, and the
posterior was fitted on these months, so its training errors are in-sample for the curves).
K1  p_lag + kappa (E_s - p_lag), kappa fixed before scoring: (i) the tree/structural slope ratio from trees_slope.md,
    (ii) the OLS coefficient of (p - p_lag) on (p_E - p_lag) over the training run.
R1  LightGBM on training-run errors (actual - p_E, ct/kWh), features: p_E, p_lag, p_E - p_lag, German RL forecast and its
    change against the lag MTU, local hour, weekday, month-free; frame calendar + eu weather; applied to E_s and E.
R2  as R1, refitted for each holdout day on training rows plus strict holdout errors of earlier days (expanding).
Writes distill_results.md."""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import *
KAPPA_TREE = float(os.environ.get("KAPPA_TREE", "nan"))
fr = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
def feats(recs_path, pred_path):
    recs = pickle.load(open(recs_path, "rb"))["recs"]
    df = pd.DataFrame({"t": [r["t"] for r in recs], "D0": [r["D"][0] / 1000 for r in recs], "lagD0": [r["lag"]["D"][0] / 1000 for r in recs],
                       "plag": [r["lag"]["p"][0] / 10 for r in recs], "Pg": [r["de"]["Pg"] for r in recs], "hdd": [r["de"]["hdd"] for r in recs],
                       "Lc": [r["Lc"] for r in recs], "Dsum_nb": [np.sum(r["D"][1:]) / 1000 for r in recs]}).set_index("t")
    R = pd.read_pickle(pred_path); R = R[R.variant == "b"]
    df["pE"] = pd.Series(R.p_draws.apply(lambda d: np.nanmedian(d) if np.isfinite(d).any() else np.nan).to_numpy() / 10, index=pd.DatetimeIndex(R.t)).reindex(df.index)
    return df
def design(df):
    X = pd.DataFrame(index=df.index)
    X["pE"] = df.pE; X["plag"] = df.plag; X["dpE"] = df.pE - df.plag; X["D0"] = df.D0; X["dD0"] = df.D0 - df.lagD0; X["hdd"] = df.hdd; X["Dnb"] = df.Dsum_nb
    loc = df.index.tz_convert("Europe/Berlin"); X["hour"] = loc.hour + loc.minute / 60; X["dow"] = loc.dayofweek
    for c in ("holiday", "eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature", "sunelevation"): X[c] = fr[c].reindex(df.index).to_numpy()
    return X
PAR = dict(objective="l1", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbosity=-1, num_threads=1, force_col_wise=True, seed=1)
def fit(X, y, n=300):
    import lightgbm as lgb
    ok = np.isfinite(y) & np.isfinite(X["pE"].to_numpy()); return lgb.train(PAR, lgb.Dataset(X[ok], label=y[ok]), n)

if __name__ == "__main__":
    M = load_members(); a, cut = M["a"], M["cut"]
    tr = feats(f"{SP}/distill/pred_train.pkl", f"{SP}/distill/train11.pkl"); tr["y"] = fr.price.reindex(tr.index).to_numpy()
    tr = tr[tr.index < pd.Timestamp("2026-09-01", tz="UTC")]
    hs = feats(f"{E11}/pred_holdout_strict.pkl", f"{E11}/oos11s.pkl"); hs["y"] = fr.price.reindex(hs.index).to_numpy()
    hp = feats(f"{E11}/pred_holdout.pkl", f"{E11}/oos11.pkl")
    out = ["# Distillation / grey-box experiments (61 holdout days, ct/kWh, pre-gate inputs unless stated)", ""]
    ok = tr.pE.notna() & tr.y.notna()
    out.append(f"Training run: {int(ok.sum())} hourly MTUs 2026-01-01..2026-08-31 solved of {len(tr)}; in-sample structural MAE {np.abs(tr.pE - tr.y)[ok].mean():.3f}, "
               f"persistence (p_lag) MAE {np.abs(tr.plag - tr.y)[ok].mean():.3f}.")
    # K1 kappa from the training run: OLS of (p - p_lag) on (p_E - p_lag), no intercept
    x_ = (tr.pE - tr.plag)[ok].to_numpy(); y_ = (tr.y - tr.plag)[ok].to_numpy(); k_tr = float(x_ @ y_ / (x_ @ x_))
    kgrid = np.linspace(0.3, 1.6, 27); kmae = [np.abs(tr.plag + k * (tr.pE - tr.plag) - tr.y)[ok].mean() for k in kgrid]; k_l1 = float(kgrid[int(np.argmin(kmae))])
    out.append(f"kappa from training: OLS {k_tr:.3f}; MAE-optimal on training grid {k_l1:.2f}; tree/structural slope ratio (trees_slope.md) {KAPPA_TREE:.3f}.")
    grid = lambda s: series_to_grid(s, cut)
    plag = grid(hs.plag); Es = M["Es"]; E = M["E"]
    F = {"T": M["T"], "P": M["P"], "A": M["A"], "E_s": Es, "E": E}; F["M3"] = (F["T"] + F["P"] + F["A"]) / 3
    fill = lambda X, base: np.where(np.isfinite(X), X, base)
    for nm, k in (("K1 tree-slope kappa", KAPPA_TREE), ("K1 training-OLS kappa", k_tr), ("K1 training-MAE kappa", k_l1)):
        if np.isfinite(k): F[f"{nm} ({k:.2f})"] = fill(plag + k * (Es - plag), Es)
    # R1 frozen residual learner
    Xtr = design(tr); m1 = fit(Xtr, (tr.y - tr.pE).to_numpy())
    Xhs = design(hs); cs = grid(pd.Series(m1.predict(Xhs), index=hs.index)); F["R1 E_s + learned correction"] = fill(Es + cs, Es)
    Xhp = design(hp); cp_ = grid(pd.Series(m1.predict(Xhp), index=hp.index)); F["R1 E + learned correction (post-gate)"] = fill(E + cp_, E)
    imp = pd.Series(m1.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False); imp = (imp / imp.sum()).round(3)
    # R2 expanding: refit per holdout day on training rows + strict holdout errors of earlier days
    hs_day = hs.index.tz_convert("Europe/Berlin").normalize(); R2 = np.full_like(Es, np.nan)
    for j, c in enumerate(cut):
        dloc = c.tz_convert("Europe/Berlin").normalize(); prev = hs[hs_day < dloc]
        X2 = pd.concat([Xtr, design(prev)]) if len(prev) else Xtr; y2 = np.r_[(tr.y - tr.pE).to_numpy(), (prev.y - prev.pE).to_numpy()]
        m2 = fit(X2, y2); today = hs[hs_day == dloc]
        if len(today): R2[j] = series_to_grid(pd.Series(m2.predict(design(today)), index=today.index), [c])[0]
    F["R2 E_s + expanding correction"] = fill(Es + R2, Es)
    # Addendum 1: fuel-scaled residual learners (price-level features and target divided by the fuel index Lc)
    def design_s(df):
        X = design(df)
        for c in ("pE", "plag", "dpE"): X[c] = X[c] / df.Lc * 100
        return X
    ys = lambda df: ((df.y - df.pE) / df.Lc * 100).to_numpy()
    m1s = fit(design_s(tr), ys(tr)); F["R1s E_s + fuel-scaled correction (add. 1)"] = fill(Es + grid(pd.Series(m1s.predict(design_s(hs)) * hs.Lc.to_numpy() / 100, index=hs.index)), Es)
    R2s = np.full_like(Es, np.nan)
    for j, c in enumerate(cut):
        dloc = c.tz_convert("Europe/Berlin").normalize(); prev = hs[hs_day < dloc]; today = hs[hs_day == dloc]
        X2 = pd.concat([design_s(tr), design_s(prev)]) if len(prev) else design_s(tr); y2 = np.r_[ys(tr), ys(prev)] if len(prev) else ys(tr)
        m2 = fit(X2, y2)
        if len(today): R2s[j] = series_to_grid(pd.Series(m2.predict(design_s(today)) * today.Lc.to_numpy() / 100, index=today.index), [c])[0]
    F["R2s E_s + fuel-scaled expanding correction (add. 1)"] = fill(Es + R2s, Es)
    # Addendum 1: band-dependent slope distillation, band on the pre-gate E_s forecast (EUR/MWh)
    KB = [(-1e9, 0, 0.40), (0, 40, 0.69), (40, 120, 1.06), (120, 200, 1.14), (200, 1e9, 0.41)]
    kap = np.ones_like(Es)
    for lo, hi, kk in KB: kap[(Es * 10 > lo) & (Es * 10 <= hi)] = kk
    F["K3 band kappa (add. 1)"] = fill(plag + kap * (Es - plag), Es)
    if len(sys.argv) > 1:
        K2 = struct_member(sys.argv[1], cut); F[f"K2 re-clear, German dRL x kappa ({os.environ.get('KAPPA_K2', '?')})"] = np.where(np.isfinite(K2), K2, FALLBACK)
    base = ["T", "P", "A", "E", "E_s", "M3"]
    rows = {}
    for k_, v in F.items():
        r = {"MAE": f"{day_mae(v, a).mean():.3f}", "bias": f"{(v - a).mean():+.2f}"}
        if k_ not in base:
            r["vs E_s"] = paired(v, Es, a); r["vs own base (E for post-gate rows)"] = paired(v, E if "post-gate" in k_ else Es, a); r["vs T"] = paired(v, F["T"], a); r["M3+it (4 equal) vs M3"] = paired((3 * F["M3"] + v) / 4, F["M3"], a)
        rows[k_] = r
    out += ["", pd.DataFrame(rows).T.fillna("").to_markdown(), "", "R1 feature importance (gain share): " + ", ".join(f"{k} {v}" for k, v in imp.items()), ""]
    # where the R1 correction acts: by window
    loc_h = M["loc_h"]; tab = {}
    for k_ in ("E_s", "R1 E_s + learned correction", "R2 E_s + expanding correction", "R1s E_s + fuel-scaled correction (add. 1)", "R2s E_s + fuel-scaled expanding correction (add. 1)", "K3 band kappa (add. 1)", "T"):
        for lo, hi in [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24)]:
            mm = (loc_h >= lo) & (loc_h < hi); tab.setdefault(k_, {})[f"{lo:02d}-{hi:02d}"] = round(float(np.abs(F[k_] - a)[mm].mean()), 2)
        tab[k_]["price<=0"] = round(float(np.abs(F[k_] - a)[a <= 0].mean()), 2)
    out += ["MAE by local window and at non-positive prices:", "", pd.DataFrame(tab).T.to_markdown(), ""]
    np.savez(f"{SP}/distill/distill_fc.npz", **{k_.split(" ")[0] + str(i): v for i, (k_, v) in enumerate(F.items())})
    open(os.path.dirname(os.path.abspath(__file__)) + "/distill_results.md", "w").write("\n".join(out)); print("\n".join(out))
