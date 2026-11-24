"""
Eleven-zone external slope check (EXPECTATIONS_11ZONE.md, "External slope check"). Usage:
    slope11.py RECS.pkl PANEL_DIR OUT_PREFIX KIND NMAX POST.npz:DRAW [POST.npz:DRAW ...]
KIND = fit (random fit-sample MTUs, the check) or outage (sanity check E4 against the outage study).
Per MTU and draw: recover shocks, re-clear the baseline on the replay domain, re-clear with German demand + 1 GW
(neighbour demand, rest-of-Europe exchanges and inferred rows held), record dpDE. Output: OUT_PREFIX.pkl (rows) and,
for KIND = fit, OUT_PREFIX.json with the weighted slopes against the empirical net slope (cf.py recipe).
"""
import sys, time, pickle, json
import numpy as np, pandas as pd
from multiprocessing import Pool
RECS, PANEL, OUTP, KIND, NMAX = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]); SPEC = [a for a in sys.argv[6:] if a != "--polish"]
POLISH = "--polish" in sys.argv
sys.argv = ["cf11.py", RECS, "/dev/null", "1"] + [s.split(":")[0] for s in SPEC]   # cf11 reads its argv at import
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import cf11 as C, net11 as N

SOLVER = "ip"   # "ip": cold-start interior point (solve11.solve_ip); "polish": warm-start active set from the observed point

def run_ip(th, nb, r):
    import solve11 as V
    const, terms = V.curves_logistic(th, nb, r); S0, _ = V.supply_fn(const, terms)(r["p"]); load = S0 - r["n"]
    A, b, lo, hi, _ = C.domain(r, None)
    pb, yb, okb, ib = V.solve_ip(const, terms, load, r["H"], A, b, r["e"], r["e0"], lo, hi)
    l1 = load.copy(); l1[0] += 1000.0
    pd_, yd, okd, _ = V.solve_ip(const, terms, l1, r["H"], A, b, r["e"], r["e0"], lo, hi)
    if not (okb and okd): return dict(ok=False, cap=False, dp=float("nan"), dimp=float("nan"), rep_err=float("nan"))
    return dict(ok=bool(pd_.max() < 4000), cap=bool(pd_.max() >= 4000), dp=float(pd_[0] - pb[0]),
                dimp=float(-((r["H"] @ yd)[0] - (r["H"] @ yb)[0]) / 1000), rep_err=float(abs(pb[0] - r["p"][0])))

def run(args):
    th, nb, r = args
    if SOLVER == "ip":
        try: return run_ip(th, nb, r)
        except Exception as ex: return dict(ok=False, cap=False, dp=float("nan"), dimp=float("nan"), error=repr(ex)[:120])
    try:
        S0, _ = C.make_supply(th, nb, r)(r["p"]); load = S0 - r["n"]
        A, b, lo, hi, dflag = C.domain(r, None)
        act0 = list(range(len(b) - len(r["A_sp"]) - len(r["A_inf"]) - dflag["eq_rows"], len(b)))
        bnd0 = [(j, s) for j, s in r["bnd"]] + [(j, "lo") for j in range(N.IA + 1, N.NY) if r["y"][j] < 0.5 and (j, "lo") not in r["bnd"]]
        if abs(r["y"][N.IA]) >= 999.5: bnd0.append((N.IA, "hi" if r["y"][N.IA] > 0 else "lo"))
        dl = time.time() + C.BUDGET
        pb, yb, ab, bb, okb = N.polish(C.make_supply(th, nb, r), load, r["H"], A, b, r["e"], r["e0"], lo, hi, r["p"], r["y"], act0, list(set(bnd0)), deadline=dl)
        l1 = load.copy(); l1[0] += 1000.0
        pd_, yd, _, _, okd = N.polish(C.make_supply(th, nb, r), l1, r["H"], A, b, r["e"], r["e0"], lo, hi, pb, yb, ab, bb, outer=C.OUTER_OUT, deadline=dl)
        return dict(ok=bool(okb and okd) and pd_.max() < 4000, cap=bool(pd_.max() >= 4000), dp=float(pd_[0] - pb[0]),
                    dimp=float(-((r["H"] @ yd)[0] - (r["H"] @ yb)[0]) / 1000))
    except Exception as ex:
        return dict(ok=False, cap=False, dp=float("nan"), dimp=float("nan"), error=repr(ex)[:120])

if POLISH: SOLVER = "polish"

if __name__ == "__main__":
    d = [r for r in pickle.load(open(RECS, "rb"))["recs"] if r["kind"] == KIND]
    if KIND == "fit": d = [r for r in d if not any(v["missing"] for v in r["links"].values())]
    d = d[:: max(1, len(d) // NMAX)][:NMAX]
    rows, t0 = [], time.time()
    with Pool(4) as pool:
        for spec in SPEC:
            post, i = spec.split(":"); i = int(i); P = dict(np.load(post, allow_pickle=True))
            th = {k: float(P[k][i]) for k in [str(x) for x in P["de_keys"]]}
            nb = dict(sbar=P["sbar"][1:], al=P["al"][i], be=P["be"][i], aa=P["aa"][i], bb=P["bb"][i], sg=P["sg"][i])
            res = pool.map(run, [(th, nb, r) for r in d], chunksize=4)
            basin = post.split("post_")[-1].split(".")[0]
            rows += [dict(basin=basin, draw=i, t=r["t"], ahc=len(r["active_vh"]) > 0, generic="generic" in (r["aug"] or {}).get("added", []), p=float(r["p"][0]), **o)
                     for r, o in zip(d, res)]
            print(f"{basin} draw {i}: {len(d)} MTUs, {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); pickle.dump(R, open(OUTP + ".pkl", "wb"))
    print("converged share", R.ok.mean().round(3), " capped", int(R.cap.sum()), " errors", int(R.get("error", pd.Series(dtype=str)).notna().sum()))
    if KIND != "fit": sys.exit()
    # weights and price terciles exactly as in cf.py: residualised German demand of the hour (day and hour-month-weekend cells)
    import os
    Pn = pd.read_parquet(f"{PANEL}/panel.parquet")
    TRAIN_END = os.environ.get("TRAIN_END")                     # restrict the empirical side to training hours (prereg/HOLDOUT.md)
    if TRAIN_END: Pn = Pn[Pn.index < pd.Timestamp(TRAIN_END, tz="UTC")]
    dfe = pd.DataFrame({"p": Pn.p.to_numpy(), "D": Pn.DDE.to_numpy(), "DN": Pn.DN.to_numpy(), "day": Pn.day.to_numpy(),
                        "hour": Pn.hour.to_numpy(), "month": Pn.index.month, "wk": (Pn.day.dt.dayofweek >= 5).astype(int)}, index=Pn.index)
    cell = dfe.hour.astype(str) + "_" + dfe.month.astype(str) + "_" + dfe.wk.astype(str)
    r_ = dfe.D - dfe.D.groupby(dfe.day).transform("mean"); Dr = r_ - r_.groupby(cell).transform("mean")
    lev = pd.qcut(dfe.p.groupby(dfe.day).transform("mean"), 3, labels=["low", "middle", "high"])
    hr = pd.DatetimeIndex(R.t).floor("h"); R["w"] = Dr.reindex(hr).to_numpy() ** 2; R["lev"] = lev.reindex(hr).to_numpy()
    EMP = {"all": (2.383, 2.174, 2.599), "low": (2.612, 2.317, 2.931), "middle": (2.097, 1.788, 2.419), "high": (2.370, 1.936, 2.761)}
    if TRAIN_END:                                               # recompute the empirical net slope (cf.py recipe) on training hours
        rp_ = dfe.p - dfe.p.groupby(dfe.day).transform("mean"); pr = rp_ - rp_.groupby(cell).transform("mean")
        rn_ = dfe.DN - dfe.DN.groupby(dfe.day).transform("mean"); DNr = rn_ - rn_.groupby(cell).transform("mean")
        rng_ = np.random.default_rng(0); EMP = {}
        for name, mk in [("all", np.ones(len(dfe), bool))] + [(l, (lev == l).to_numpy()) for l in ("low", "middle", "high")]:
            X_ = np.c_[Dr[mk], DNr[mk]]; y_ = pr[mk].to_numpy(); b_ = np.linalg.lstsq(X_, y_, rcond=None)[0][0]
            days_ = dfe.day[mk].to_numpy(); ud = np.unique(days_); ix = pd.Series(np.arange(mk.sum())).groupby(days_).apply(np.array).to_dict()
            bs = [np.linalg.lstsq(X_[ii], y_[ii], rcond=None)[0][0] for ii in (np.concatenate([ix[d] for d in rng_.choice(ud, len(ud))]) for _ in range(300))]
            EMP[name] = (float(b_), float(np.quantile(bs, .05)), float(np.quantile(bs, .95)))
        print("empirical slope on training hours:", {k: round(v[0], 3) for k, v in EMP.items()})
    out = {}
    for name, mk in [("all", np.ones(len(R), bool))] + [(l, (R.lev == l).to_numpy()) for l in ("low", "middle", "high")] + \
                    [("ahc", R.ahc.to_numpy()), ("non-ahc", ~R.ahc.to_numpy()), ("generic", R.generic.to_numpy())]:
        S = R[mk & R.ok.to_numpy() & R.w.notna().to_numpy()]
        per = S.groupby(["basin", "draw"]).apply(lambda g: np.average(g.dp, weights=g.w)) if len(S) else pd.Series(dtype=float)
        e = EMP.get(name, EMP["all"])
        out[name] = dict(model_weighted=float(per.mean()), draw_min=float(per.min()), draw_max=float(per.max()), model_median=float(S.dp.median()),
                         imports_share=float(S.dimp.mean()), empirical=e[0], emp_q05=e[1], emp_q95=e[2], ratio=float(e[0] / per.mean()),
                         n=int(len(S)), failed=int((mk & ~R.ok.to_numpy()).sum()))
    json.dump(out, open(OUTP + ".json", "w"), indent=1); print(pd.DataFrame(out).T.round(3).to_string())
