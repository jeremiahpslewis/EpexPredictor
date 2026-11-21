"""
Checks of the GASCURVE switch on a few days (notes/gas_in_slice.md, E1 and E2): seconds per clear, master rounds and
cuts, the capacity accounting, and zone A's cleared gas output against the curve evaluated at the slice's own cleared
prices with the same posterior draw (with the draw's lambda and with lambda = 0); slice gas by 4-hour block against observed
output, and the share of quarters priced at or below the lowest curve slice and at or above the backstop. Run with the same environment as
hist_window.py plus GASCURVE=1 GASCURVE_DRAWS=...; usage: gas_slice_check.py DATA_DIR OUT_DIR [DAY_INDEX ...] (draws k=0..N-1, N = CHECK_DRAWS, 4).
"""
import os, sys, time
from pathlib import Path
idx_days = [int(a) for a in sys.argv[3:]] or [0, 7]; sys.argv = sys.argv[:3]
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import hist_window as H
import vslice as V

assert H.GASCURVE
rows = []
for di in idx_days:
    d = H.DAYS[di]
    for k in range(int(os.environ.get("CHECK_DRAWS", 4))):
        info, idx = H.day_inputs(d); kk = di * 1000 + k
        rng = np.random.default_rng(3_000_000 + kk); th = V.draw_theta(rng); day = V.draw_day(rng, info)
        info.coal_price = float(rng.lognormal(np.log(12.0), 0.25)); info.peak_mar = 0.0
        j = int(np.random.default_rng(7_000_000 + kk).integers(len(H.GC["A"]))); hb = np.digitize(V.HOURS, [6, 10, 16, 20])
        gc = dict(F=1000 * float(H.GC["F_last"][j]), A=1000 * float(H.GC["A"][j]), lam=float(H.GC["lam"][j]), mu=H.GC["mu_b"][j][hb], s=H.GC["s_b"][j][hb], nq=H.GC_NQ, mar=H.GC_MAR)
        info.gas_curve = {"A": gc}
        t0 = time.time(); m, acc, r = V.simulate(th, day, info); sec = time.time() - t0
        gi = acc["A"]["gas_orders"]
        out = acc["A"]["split"]["gas"]["netted"].copy()
        for jj in gi: out[m.orders[jj].t] += r.x[jj] * m.orders[jj].q
        for b, blk in enumerate(m.blocks):
            if "gas curve" in blk.name: out += r.a[b] * np.array(blk.q)
        # the curve at the slice's prices: hourly margins, centred +-3 h mean (within the day), quarter-hour output
        p = r.prices[0]; ref = 1.85 * info.gas + 0.37 * info.co2; mh = (p - ref).reshape(24, 4).mean(1)
        mw = np.array([mh[max(0, h - 3):h + 4].mean() for h in range(24)])
        def curve(lam):
            me = np.repeat(mh + lam * (mw - mh), 4)
            return gc["F"] + gc["A"] / (1 + np.exp(-(me - gc["mu"]) / gc["s"]))
        qs = (np.arange(gc["nq"]) + 0.5) / gc["nq"]; lo_p = ref + gc["mu"] + gc["s"] * np.log(qs[0] / (1 - qs[0]))
        hi_p = ref + gc["mu"] + gc["s"] * np.log(0.995 / 0.005)
        obs_g = H.Xr["gasg"].reindex(idx.tz_convert("UTC").floor("h")).to_numpy(float)       # observed gas output, GW (hourly)
        hb4 = (V.HOURS // 4).astype(int)
        within = out <= gc["F"] + gc["A"] + 1                      # leave out quarters where the backstop beyond F + A runs
        rows.append(dict(day=str(d.date()), k=k, sec=round(sec, 2), rounds=r.rounds, cuts=len(r.cuts), acct=V.accounting(acc),
                         lam=round(gc["lam"], 2), mae_lam=float(np.abs(out - curve(gc["lam"]))[within].mean()) / 1000,
                         mae_0=float(np.abs(out - curve(0.0))[within].mean()) / 1000, mean_out=out.mean() / 1000,
                         backstop_q=int((~within).sum()), pmin=round(float(p.min())), pmax=round(float(p.max())),
                         at_or_below_lowest=float((p <= lo_p + 0.01).mean()), at_or_above_backstop=float((p >= hi_p - 0.01).mean()),
                         **{f"gas_{4*b:02d}": float(out[hb4 == b].mean()) / 1000 for b in range(6)}, **{f"obs_{4*b:02d}": float(np.nanmean(obs_g[hb4 == b])) for b in range(6)}))
        print(rows[-1], flush=True)
import pandas as pd
D = pd.DataFrame(rows); print(D.describe().round(3).to_string())
print("by 4h, slice gas GW:", [round(D[f"gas_{4*b:02d}"].mean(), 2) for b in range(6)], "observed:", [round(D[f"obs_{4*b:02d}"].mean(), 2) for b in range(6)])
print("share of quarters priced at/below the lowest slice", round(D.at_or_below_lowest.mean(), 3), "at/above the backstop", round(D.at_or_above_backstop.mean(), 3))
print("mean sec", round(D.sec.mean(), 2), "max rounds", D.rounds.max(), "cuts", D.cuts.sum(), "max acct", D.acct.max(),
      "MAE lam", round(D.mae_lam.mean(), 3), "MAE lam=0", round(D.mae_0.mean(), 3))
