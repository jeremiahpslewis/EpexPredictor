"""The candidate's online layer on the market model (notes 21.15). Writes, per origin of each scoring period, a file in the format dyn2.py reads:
R1 / XH1 = history of the hourly price error and of the load axis (200 days, the last row being D+1, whose prices are known at issue and whose load is
the D+1 forecast the candidate uses), x_known / known_shape = load axis and bare market-model price for the 153 target quarter-hours (true load given).
AXIS=pub: demand built from the PUBLISHED residual load (the series the candidate and Timestral use), history and targets alike.
usage: mm_layer.py RUN [carry|gw]     then: dyn2.py PERIOD RUN known x+xh"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import importlib; MOD = os.environ.get("MOD", "market0d"); M = importlib.import_module(MOD)
SP = M.SP; RUN = sys.argv[1]; CARRY = len(sys.argv) > 2 and sys.argv[2] == "carry"; GW = len(sys.argv) > 2 and sys.argv[2] == "gw"; LAGMODE = int(os.environ.get("LAGMODE", 1)); AXIS = os.environ.get("AXIS", "real"); th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/{MOD}_post.npz").items() if np.ndim(v) == 1}
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["p_de", "rlreal_de", "rl_de", "g_de_other"]]; S.index = S.index.tz_convert("UTC"); S["D"] = (S.rlreal_de - S.g_de_other).interpolate(limit=8)
R = pd.read_parquet(f"{SP}/out/{MOD}_rows.parquet"); R.index = R.index.tz_convert("UTC"); full = pd.date_range(R.index.min().floor("D"), R.index.max().ceil("D"), freq="1h", tz="UTC"); G = R.reindex(full); co2 = G.co2.ffill()
EUW = os.environ.get("EUW", "0") in ("1", "wind", "model"); EUM = os.environ.get("EUW", "0") == "model";   # model: coefficients from the fit itself (market0g)
EUSEL = [0, 1] if os.environ.get("EUW", "0") == "wind" else [0, 1, 2, 3]   # Europe-wide weather: the supply shift is first regressed on the four features over the 200-day history (per origin), the layer learns the rest
if EUW:
    import live_common as lc; EUC = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]; EUF = lc.load()["DE"][EUC].resample("1h").mean(); EUF.index = EUF.index.tz_convert("UTC")
os.makedirs(f"{SP}/out/struct_{RUN}", exist_ok=True); NH = 200; chk = []; REFIT = os.environ.get("REFIT", "0") == "1"; TH0 = th; used = []
if REFIT:   # walk-forward: each origin uses the latest refit whose data end no later than midnight starting its issue day
    import glob; FITS = sorted((pd.Timestamp(f.split("_r")[-1][:8]).tz_localize("Europe/Berlin"), {k: float(np.median(v)) for k, v in np.load(f).items() if np.ndim(v) == 1}) for f in glob.glob(f"{SP}/out/{MOD}_r*_post.npz"))
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{per}.npz"); GEN = np.load(f"{SP}/out/gen_ts_{per}.npz")
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); tag = str(c)[:10]
        if REFIT: day = (t0 - pd.Timedelta(hours=34)).tz_convert("Europe/Berlin").normalize(); e_, th = [f for f in FITS if f[0] <= day][-1]; used.append(str(e_.date()))
        Lj = 1.85 * float(I["gas"][j]) + 0.37 * float(co2.asof(t0 - pd.Timedelta(hours=34))); tt = t0 + pd.Timedelta(minutes=15) * np.arange(153); x = S.D.reindex(tt).ffill().bfill().values
        hh = pd.date_range(t0 - pd.Timedelta(hours=24 * NH), t0 - pd.Timedelta(hours=1), freq="1h"); g = G.reindex(hh); P = g.p.values.reshape(NH, 24); X = g.D.values.reshape(NH, 24).copy(); P0 = g.p0.values.reshape(NH, 24).copy(); oth = np.nanmean(g.other.values.reshape(NH, 24)[-9:-2], 0)   # other plant: mean by hour of the 7 days ending the day before the issue day
        if AXIS == "pub":
            X = (g.rl.values - g.other.values).reshape(NH, 24).copy(); X[-2] = g.rl.values.reshape(NH, 24)[-2] - oth; x = S.rl_de.reindex(tt).ffill().bfill().values - oth[(np.arange(153) // 4) % 24]
        d1 = D1["rl_d1"][j].astype(float).reshape(24, 4).mean(1) - oth; chk.append(np.corrcoef(D1["rl_d1_true"][j].astype(float), S.rlreal_de.reindex(t0 - pd.Timedelta(hours=24) + pd.Timedelta(minutes=15) * np.arange(96)).ffill().bfill().values)[0, 1])
        e = {}
        if CARRY:   # shocks carried: mean of the last 7 known days by hour (days up to D; D+1 has prices but no quantities)
            e = {a: np.nanmean(g[cn].values.reshape(NH, 24)[-8:-1], 0) for a, cn in (("eps", "eps"), ("eT", "eT"), ("eX", "eX"), ("eS", "eS"))}
            for r in range(8, NH - 1):   # history of the same forecast: each past day predicted from the 7 days ending two days before it
                ee = {a: np.nanmean(g[cn].values.reshape(NH, 24)[r - 8:r - 1], 0) for a, cn in (("eps", "eps"), ("eT", "eT"), ("eX", "eX"), ("eS", "eS"))}; ok = ~np.isnan(X[r])
                if ok.any(): P0[r, ok] = M.clear(th, X[r][ok], g.L.values.reshape(NH, 24)[r][ok], **{a: np.nan_to_num(v[ok]) for a, v in ee.items()})[0]
            P0[:8] = np.nan
        X[-1] = d1; P0[-1] = M.clear(th, d1, np.full(24, Lj), **{a: np.nan_to_num(v) for a, v in e.items()})[0]
        hod = (np.arange(153) // 4) % 24; shape = M.clear(th, x, np.full(153, Lj), **{a: np.nan_to_num(v)[hod] for a, v in e.items()})[0] / 10
        if LAGMODE == 1 and AXIS != "pub": X[-2, 14:] = np.nan     # issue day D: quantities known up to 14:00 only
        if LAGMODE == 2: X[-2, :] = np.nan       # issue day D: no quantities at all (feed a day late)
        if GW:   # the error carried in GW: each known price inverted into the supply shift it implies (demand minus what the curves supply at that price)
            Lh = g.L.values.reshape(NH, 24).copy(); Lh[-1] = Lj; ok = ~np.isnan(P + X + Lh); u = np.full((NH, 24), np.nan); gT, gX, gS = M.curves(th, np.where(ok, P, 0.0), np.where(ok, Lh, 90.0)); wd = th["K_s"] * (P < -M.EPS) + 0.5 * th["K_s"] * (np.abs(P) <= M.EPS)
            u[ok] = (X - (gT + gX + gS + th["B_g"] - wd))[ok]; eus = np.zeros(153)
            if EUW:
                Zh = EUF.reindex(hh).values.reshape(NH, 24, 4); mu = np.nanmean(Zh, (0, 1)); sdv = np.nanstd(Zh, (0, 1))
                if EUM: PZ = np.load(f"{SP}/out/{MOD}_post.npz"); mu, sdv = PZ["MU"], PZ["SD"]; Zs = (Zh - mu) / sdv; kap = np.median(PZ["kT"], 0) + np.median(PZ["kX"], 0)
                else: Zs = (Zh - mu) / sdv; v_ = ok & ~np.isnan(Zs).any(2); A_ = np.column_stack([Zs[v_][:, EUSEL], np.ones(int(v_.sum()))]); kap = np.zeros(4); kap[EUSEL] = np.linalg.lstsq(A_, u[v_], rcond=None)[0][:len(EUSEL)]
                u = u - np.nan_to_num(Zs) @ kap; eus = ((I["eu"][j].astype(float) - mu) / sdv) @ kap
            xts = GEN["rl_pes"][j].astype(float) - oth[(np.arange(153) // 4) % 24]; np.savez_compressed(f"{SP}/out/struct_{RUN}/{tag}.npz", R1=(10 * u).astype(np.float32), XH1=X.astype(np.float32), x_known=x.astype(np.float32), known_shape=np.zeros(153, np.float32), x_ts=xts.astype(np.float32), ts_shape=np.zeros(153, np.float32), actual=I["actual"][j], Lj=Lj, thk=np.array(list(th.keys())), thv=np.array(list(th.values())), eu_shift=eus.astype(np.float32)); continue
        np.savez_compressed(f"{SP}/out/struct_{RUN}/{tag}.npz", R1=(P - P0).astype(np.float32), XH1=X.astype(np.float32), x_known=x.astype(np.float32), known_shape=shape.astype(np.float32), actual=I["actual"][j])
    if GW: print(per, "written (GW mode)" + (f"; fits used: {sorted(set(used))}" if REFIT else "")); used = []; continue
    a = I["actual"].astype(float); sh = np.stack([np.load(f"{SP}/out/struct_{RUN}/{str(c)[:10]}.npz")["known_shape"] for c in I["cutoffs"]]); print(f"{per}: {len(a)} origins written; bare shape night / first 14 h {np.abs(sh - a)[:, :32].mean():.2f} / {np.abs(sh - a)[:, :56].mean():.2f}")
print("alignment check, correlation of the D+1 file's true residual load with mine over the D+1 day: min", round(float(np.min(chk)), 3), "median", round(float(np.median(chk)), 3))
