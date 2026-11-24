"""
Scorer for the out-of-sample test (prereg/OOS_PROTOCOL.md). Usage:
    score_oos.py PANEL_DIR TARGETS.pkl OUT.md LABEL=KIND:PATH [LABEL=KIND:PATH ...]
KIND is 11 (predict11.py output, MTU-level draws) or 2 (predict2.py output, hourly draws).
- Misses: every missed solve is replaced by the training-window mean (price, net import) BEFORE the median over
  draws, so steep cases that fail are scored, not dropped. An hour with no eleven-zone record scores the fallback.
- Eleven-zone hourly forecast: mean over the hour's MTUs of the per-MTU median over draws.
- Naive: the same hour of the previous day.
- Metrics: MAE and RMSE of the hourly German price; MAE of the hourly German net import (SDAC10, GW); by observed price
  band; share of missed solves. Paired differences in absolute error: block bootstrap by day, 1,000 resamples, 90%.
"""
import sys
import numpy as np, pandas as pd
PANEL, TG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; SPECS = sys.argv[4:]
TRAIN_END = pd.Timestamp("2026-09-08", tz="UTC")
Pn = pd.read_parquet(f"{PANEL}/panel.parquet"); T = pd.read_pickle(TG)
FB_P, FB_M = float(Pn.p[Pn.index < TRAIN_END].mean()), float(Pn.m[Pn.index < TRAIN_END].mean())
fc, miss = {}, {}
for spec in SPECS:
    label, rest = spec.split("="); kind, path = rest.split(":", 1); R = pd.read_pickle(path)
    for v in sorted(R.variant.unique()):
        S = R[R.variant == v]
        if kind == "11":
            p = S.p_draws.apply(lambda d: np.median(np.where(np.isnan(d), FB_P, d))); m = S.m_draws.apply(lambda d: np.median(np.where(np.isnan(d), FB_M, d)))
            miss[f"{label}{v}"] = float(np.mean([np.isnan(d).mean() for d in S.p_draws]))
            H = pd.DataFrame({"h": pd.DatetimeIndex(S.t).floor("h"), "p": p.to_numpy(), "m": m.to_numpy()}).groupby("h").mean()
        else:
            miss[f"{label}{v}"] = float(S.p.isna().mean())
            H = S.assign(p=S.p.fillna(FB_P), m=S.m.fillna(FB_M)).groupby("h")[["p", "m"]].median()
        fc[f"{label}{v}"] = H.reindex(T.index).fillna({"p": FB_P, "m": FB_M})
lag = T.index - pd.Timedelta(hours=24)
fc["naive"] = pd.DataFrame({"p": Pn.p.reindex(lag).to_numpy(), "m": Pn.m.reindex(lag).to_numpy()}, index=T.index).fillna({"p": FB_P, "m": FB_M}); miss["naive"] = 0.0
err = {k: pd.DataFrame({"ep": v.p - T.p, "em": v.m - T.m}) for k, v in fc.items()}
band = pd.cut(T.p, [-1e9, 0, 40, 120, 200, 1e9], labels=["<=0", "0-40", "40-120", "120-200", ">200"])
rows = [dict(forecast=k, price_MAE=e.ep.abs().mean(), price_RMSE=np.sqrt((e.ep ** 2).mean()), import_MAE=e.em.abs().mean(), missed_solves=miss[k]) for k, e in err.items()]
out = [f"# Out-of-sample scores ({len(T)} hours, {T.index.min():%Y-%m-%d} to {T.index.max():%Y-%m-%d})", "",
       f"Fallback for misses: training-window means, price {FB_P:.1f} EUR/MWh, net import {FB_M:.2f} GW.", "",
       pd.DataFrame(rows).set_index("forecast").round(3).to_markdown(), "", "## Price MAE by observed price band", "",
       pd.DataFrame({k: e.ep.abs().groupby(band, observed=False).mean() for k, e in err.items()}).round(2).assign(hours=band.value_counts().reindex(band.cat.categories)).to_markdown(), ""]
day = T.index.floor("D"); days = day.unique(); rng = np.random.default_rng(0)
idx = {d: np.flatnonzero(day == d) for d in days}; B = [np.concatenate([idx[d] for d in rng.choice(days, len(days))]) for _ in range(1000)]
def paired(a, b, col):
    d = (err[a][col].abs() - err[b][col].abs()).to_numpy(); bs = [d[ii].mean() for ii in B]
    return f"{d.mean():+.3f} [{np.quantile(bs, .05):+.3f}, {np.quantile(bs, .95):+.3f}]"
keys = [k for k in fc if k != "naive"]
pairs = [(a, "naive") for a in keys] + [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
out += ["## Paired differences in absolute error (first minus second; negative = first better), 90% block-bootstrap band", "",
        pd.DataFrame([dict(pair=f"{a} − {b}", price=paired(a, b, "ep"), net_import=paired(a, b, "em")) for a, b in pairs]).set_index("pair").to_markdown(), ""]
open(OUT, "w").write("\n".join(out)); print("\n".join(out))
