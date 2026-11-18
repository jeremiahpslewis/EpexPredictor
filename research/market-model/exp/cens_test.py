"""Mechanism check: fit once, apply unchanged to the following five months, and look at the level evidence of the usable hours by month.
Arms by environment: CENS (prices <= 0 censored in the likelihood), TROUGH (trough-day term on/off).   usage: cens_test.py FITDATE ENDDATE"""
import sys, os, numpy as np, pandas as pd
os.environ["OCXV"] = "t"
import oc, ocxt
from ocx_state import implied, daily
from ocxt import *
h1, h2 = hx(sys.argv[1]), hx(sys.argv[2]); A, d = fit_t(h1); ok = d["ok"] > 0; e = (d["p"] - pred_t(A, d))[ok] / 10
sl = slice(h1, h2); f = rows_t(sl, drl=np.zeros(h2 - h1)); dx, code = implied(A, f); z = daily(dx, oc.DAY[sl]); ff = {k: (v if k == "G" else np.nan_to_num(v)) for k, v in f.items()}; ff["okx"] = np.isfinite(f["rl"]).all(1) & np.isfinite(f["sol"]); ph = pred_t(A, ff)
idx = pd.DatetimeIndex(S.idx[: oc.n15 // 4 * 4: 4])[sl].tz_convert("Europe/Berlin"); mo = idx.strftime("%y-%m"); hr = oc.HOUR[sl]; err = (f["p"] - ph) / 10; dmo = pd.Series(mo).groupby(oc.DAY[sl]).first().reindex(z.index)
lab = f"censored {'yes' if ocxt.CENS else 'no '}, trough term {'yes' if ocxt.TROUGH else 'no '}"
print(f"{lab} | fit {sys.argv[1]}: in sample MAE {np.mean(np.abs(e)):.2f}, kc {float(A['kc']):.1f} qc {float(A['qc']):.1f} wc {float(A['wc']):.1f} qa {float(A['qa']):.1f} wa {float(A['wa']):.1f} s0 {float(A['s0']):.1f}")
print("    forward, by month: evidence GW " + " ".join(f"{m} {z[dmo.values == m].mean():+.1f}" for m in sorted(set(mo))) + " | night price error ct " + " ".join(f"{np.nanmean(err[(mo == m) & (hr < 8)]):+.2f}" for m in sorted(set(mo))) + " | night MAE " + " ".join(f"{np.nanmean(np.abs(err[(mo == m) & (hr < 8)])):.2f}" for m in sorted(set(mo))) + f" | all: night {np.nanmean(np.abs(err[hr < 8])):.2f}, 0-14 h {np.nanmean(np.abs(err[hr < 14])):.2f}, 0-14 h with prices floored at 0 for scoring {np.nanmean(np.abs(np.maximum(f['p'], 0) / 10 - ph / 10)[hr < 14]):.2f}")
