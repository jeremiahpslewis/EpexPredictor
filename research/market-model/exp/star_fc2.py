"""Forecasts from the coupled-market model with an online level correction, from the weekly fits saved by star_fc.py.
For each origin (14:00 on D, target D+2 00:00 onwards) the market is cleared in one pass for D-1, D, D+1 and the forecast hours:
  D-1, D: published residual loads (known), gas, carbon and availability as they were
  D+1:    prices are known (auction at noon on D); residual loads are the arm's (ts: Timestral day-ahead forecasts, known: published)
  D+2..:  the arm's residual loads; gas, carbon, availability frozen at the origin
Online correction: the model's error on the 72 known hours, weighted by age (memory 2 days) and by distance in hour of day (width 1 h),
is added to the forecast. The two settings are the ones the earlier structural model's correction layer settled on; nothing is tuned here.
usage: star_fc2.py RUN PERIOD [FITRUN]"""
import sys, os, glob, numpy as np
import star as S
run, period = sys.argv[1], sys.argv[2]; fitrun = sys.argv[3] if len(sys.argv) > 3 else run; SP = S.lc.SP; FD = f"{SP}/out/starfc_{fitrun}_{period}"
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
fits = sorted(glob.glob(f"{FD}/fit_*.npz")); ftag = [os.path.basename(f)[4:14] for f in fits]; MEM, HW = 2.0, 1.0
res = {a: np.full((n, 153), np.nan) for a in ("known", "ts", "known_oc", "ts_oc")}; a_ = I["actual"].astype(float)
hod = lambda v: np.asarray(v) % 24
for j, t in enumerate(tags):
    fi = max(i for i, ft in enumerate(ftag) if ft <= t); z = np.load(fits[fi]); raw = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}; q = S.unpack(raw); nw = raw["c_e"].shape[0] + 1
    d0 = int(I["cuts"][j]); lo = d0 - 288; H = S.window(lo, d0 + 156); nh = len(H["P"]); d = S.data_of(H, np.full(nh, nw - 1), nw); past = np.arange(nh) < 48; fut = np.arange(nh) >= 48        # hours of D-1, D | D+1 and later
    d["gas"] = np.where(past, np.nan_to_num(H["gas"], nan=float(I["gas"][j])), float(I["gas"][j])); d["eua"] = np.where(past, d["eua"], float(S.eua_all[d0 - 192]) * (1.0 if S.USE_EUA else 0.0))
    av = np.nan_to_num(H["avail"]).copy()
    for (zz_, k), arr in S.AV.items(): av[fut, S.ZN.index(zz_), k] = arr[d0 - 96]
    d["avail"] = np.nan_to_num(av); d["drl"] = np.where(np.arange(nh) < 24, d["drl"], 0.0); d["ok"] = np.ones((nh, S.Z), bool)
    for arm in ("known", "ts"):
        rl = np.nan_to_num(H["rl"]).copy()
        if arm == "ts":
            zf = f"{NBD}/{t}.npz"
            if not os.path.exists(zf): continue
            zz = np.load(zf); zl = [str(x) for x in zz["zones"]]
            q1 = np.column_stack([D1["rl_d1"][j]] + [zz["d1_z"][zl.index(zn)] for zn in S.ZN[1:]]).astype(float).reshape(24, 4, S.Z).mean(1)                       # D+1
            q2 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(zn)][:152] for zn in S.ZN[1:]]).astype(float).reshape(38, 4, S.Z).mean(1)            # D+2 ..
            q3 = np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(zn)][152:153] for zn in S.ZN[1:]]).astype(float)
            rl[48:72] = q1; rl[72:] = np.vstack([q2, q3])[:nh - 72]
        pe, _ = S.equilibrium(q, d, rl=rl, blend=True); p = pe[:, 0] / 10.0; act = H["P"][:72, 0] / 10.0; r = act - p[:72]; hk = hod(H["hour"][:72]); age = 2 - np.arange(72) // 24      # age 0 for D+1, 1 for D, 2 for D-1
        fc = p[72:]; hf = hod(H["hour"][72:]); dist = np.abs(hf[:, None] - hk[None]); dist = np.minimum(dist, 24 - dist); w = np.exp(-age[None] / MEM) * np.exp(-0.5 * (dist / HW) ** 2) * np.isfinite(r)[None]
        corr = (w * np.nan_to_num(r)[None]).sum(1) / np.maximum(w.sum(1), 1e-9)
        res[arm][j] = np.repeat(fc, 4)[:153]; res[arm + "_oc"][j] = np.repeat(fc + corr, 4)[:153]
for arm, v in res.items():
    np.savez_compressed(f"{SP}/out/starfc2_{run}_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); e = np.abs(v - a_)
    print(f"{run} {period} {arm:9s}: MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a_ - v)[:, :32]):+.2f}, first 14 h {np.nanmean((a_ - v)[:, :56]):+.2f}")
