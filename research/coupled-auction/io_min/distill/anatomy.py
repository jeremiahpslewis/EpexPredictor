"""1. Error anatomy on the 61 holdout days (EXPECTATIONS.md A1-A5, X0). Writes anatomy.md next to this file."""
import numpy as np, pandas as pd
from common import *
M = load_members(); a, cut, loc_h = M["a"], M["cut"], M["loc_h"]
F = {"T": M["T"], "P": M["P"], "A": M["A"], "E": M["E"], "E_s": M["Es"]}; F["M3"] = (F["T"] + F["P"] + F["A"]) / 3
out = ["# Error anatomy (61 days, ct/kWh)", ""]
out.append("X0 reproduction: " + ", ".join(f"{k} {day_mae(v, a).mean():.3f}" for k, v in F.items()))
# persistence-plus-slope diagnostic (A5): yesterday's same-QH price plus slope x change in the deployed RL forecast
fr = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")["price"]
G = np.load(f"{O}/gen_lwA_autumn.npz", allow_pickle=True); rl = G["rl_pes"].astype(float)
plag = np.stack([fr.reindex(c - pd.Timedelta(days=1) + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy() for c in cut])
# forecast for D (index j) minus forecast for D-1 (index j-1, issued a day earlier) at the same clock time (24 h earlier)
drl = np.stack([rl[j, :96] - rl[j - 1, :96] for j in range(5, 66)])
for s in (0.215, 0.315):
    F[f"plag+{s*10:.2f}dRL"] = plag + s * drl
F["plag (naive)"] = plag
np.savez(f"{SP}/distill/anat_aux.npz", plag=plag, drl=drl)
e = {k: v - a for k, v in F.items()}
rows = {}
for k, v in e.items():
    dm = v.mean(1); lev = (dm ** 2).mean(); shp = ((v - dm[:, None]) ** 2).mean()
    rows[k] = dict(MAE=np.abs(v).mean(), bias=v.mean(), RMSE=np.sqrt((v ** 2).mean()), level_MSE=lev, shape_MSE=shp, level_share=lev / (lev + shp),
                   MAE_daily_level=np.abs(dm).mean(), MAE_after_removing_daily_mean=np.abs(v - dm[:, None]).mean())
out += ["## MSE split: daily level (daily mean error squared) vs within-day shape", "", pd.DataFrame(rows).T.round(3).to_markdown(), ""]
win = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24)]
tab = {k: {f"{lo:02d}-{hi:02d}": f"{v[(loc_h >= lo) & (loc_h < hi)].mean():+.2f} / {np.abs(v[(loc_h >= lo) & (loc_h < hi)]).mean():.2f}" for lo, hi in win} for k, v in e.items() if k in ("T", "P", "A", "E", "E_s", "M3")}
out += ["## By local hour window: bias / MAE", "", pd.DataFrame(tab).T.to_markdown(), ""]
# per-window: E_s excess over T split into level and shape: MAE of daily-demeaned errors
tab = {}
for lo, hi in win:
    m = (loc_h >= lo) & (loc_h < hi)
    for k in ("T", "E_s"):
        v = e[k]; dm = np.array([v[d][m[d]].mean() for d in range(len(v))])
        tab.setdefault(k, {})[f"{lo:02d}-{hi:02d} level"] = np.abs(dm).mean()
        tab[k][f"{lo:02d}-{hi:02d} shape"] = np.mean([np.abs(v[d][m[d]] - dm[d]).mean() for d in range(len(v))])
out += ["## By window: MAE of the window's daily mean error (level) and MAE around it (shape)", "", pd.DataFrame(tab).T.round(3).to_markdown(), ""]
bands = [(-99, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 999)]
tab = {k: {f"{lo}..{hi}": np.abs(v[(a > lo) & (a <= hi)]).mean() for lo, hi in bands} for k, v in e.items() if k in ("T", "P", "A", "E_s", "M3")}
tab["n QH"] = {f"{lo}..{hi}": int(((a > lo) & (a <= hi)).sum()) for lo, hi in bands}
out += ["## MAE by observed price band (ct/kWh; conditions on the outcome, so smooth forecasts look worst at the ends)", "", pd.DataFrame(tab).T.round(2).to_markdown(), ""]
rlf = np.stack([rl[j, :96] for j in range(5, 66)]); q = np.quantile(rlf, [0, .2, .4, .6, .8, 1])
tab = {k: {f"Q{i+1} ({q[i]:.0f}-{q[i+1]:.0f} GW)": f"{v[(rlf >= q[i]) & (rlf <= q[i+1])].mean():+.2f} / {np.abs(v[(rlf >= q[i]) & (rlf <= q[i+1])]).mean():.2f}" for i in range(5)} for k, v in e.items() if k in ("T", "P", "A", "E_s", "M3")}
out += ["## By quintile of the deployed (pre-gate) RL forecast: bias / MAE", "", pd.DataFrame(tab).T.to_markdown(), ""]
ks = ["T", "P", "A", "E", "E_s"]
C = pd.DataFrame({k: e[k].ravel() for k in ks}).corr(); Cd = pd.DataFrame({k: e[k].mean(1) for k in ks}).corr()
Cs = pd.DataFrame({k: (e[k] - e[k].mean(1, keepdims=True)).ravel() for k in ks}).corr()
out += ["## Error correlations", "", "QH errors:", "", C.round(2).to_markdown(), "", "daily mean errors:", "", Cd.round(2).to_markdown(), "", "within-day (demeaned) errors:", "", Cs.round(2).to_markdown(), ""]
# how much of each error is the realised day-over-day change that the forecast missed (anchoring diagnostic)
chg = a - plag; tab = {}
for k in ("T", "P", "A", "E", "E_s", "M3"):
    v = e[k].ravel(); c_ = chg.ravel(); b_ = (v @ c_) / (c_ @ c_); tab[k] = {"corr(error, actual day-over-day change)": np.corrcoef(v, c_)[0, 1],
        "share of error variance explained": np.corrcoef(v, c_)[0, 1] ** 2, "corr(error, yesterday's same-QH error)": np.nan}
    E_ = e[k]; tab[k]["corr(error, yesterday's same-QH error)"] = np.corrcoef(E_[1:].ravel(), E_[:-1].ravel())[0, 1]
out += ["## Anchoring: errors against the realised change from yesterday's same quarter-hour", "", pd.DataFrame(tab).T.round(2).to_markdown(), ""]
# missed solves (scored with the 10.07 fallback): where they sit and the key numbers without them
miss = ~np.isfinite(M["Esraw"]); tab = {}
for lo, hi in bands:
    mb = (a > lo) & (a <= hi); ok_ = mb & ~miss
    tab[f"{lo}..{hi}"] = {"missed QH": int((mb & miss).sum()), "QH": int(mb.sum()), "E_s MAE all": np.abs(e["E_s"][mb]).mean(),
                          "E_s MAE solved only": np.abs(e["E_s"][ok_]).mean(), "T MAE same slots": np.abs(e["T"][ok_]).mean()}
out += [f"## Missed E_s solves ({int(miss.sum())} QH) by observed price band", "", pd.DataFrame(tab).T.round(2).to_markdown(), "",
        "by local window (missed QH): " + ", ".join(f"{lo:02d}-{hi:02d} {int((miss & (loc_h >= lo) & (loc_h < hi)).sum())}" for lo, hi in win),
        f"E_s MAE on solved slots only {np.abs(e['E_s'][~miss]).mean():.3f} vs T on the same slots {np.abs(e['T'][~miss]).mean():.3f}; shape MSE solved only "
        f"{np.mean([((e['E_s'][d][~miss[d]] - e['E_s'][d][~miss[d]].mean()) ** 2).mean() for d in range(len(a))]):.2f}", ""]
out += ["## Paired differences (first minus second, 90% block bootstrap)", ""]
for x, y in [("E_s", "T"), ("E_s", "plag+2.15dRL"), ("E_s", "plag+3.15dRL"), ("plag+2.15dRL", "plag+3.15dRL"), ("plag+2.15dRL", "T")]:
    out.append(f"- {x} - {y}: {paired(F[x], F[y], a)}")
open(__file__.rsplit("/", 1)[0] + "/anatomy.md", "w").write("\n".join(out)); print("\n".join(out))
