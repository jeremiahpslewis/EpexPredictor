"""
Issue-time predictor for the CHP floor of German gas output (notes/gas_floor.md).
The block curve of gas_curve2.py, gasg ~ StudentT(F + A sigmoid((m_eff - mu_b)/s_b)), with three floors:
  rw      monthly random-walk floor (gas_curve2.py), predicted with the last fitted month carried forward;
  hdd     F_t = f0 + f1 HDD_d(t), HDD base 15 C from the simple mean of the six German land points;
  hdd_rw  the same plus a slow monthly random walk (scale HalfNormal(0.3)), frozen at its last value out of sample.
Two season-crossing tests (Europe/Berlin dates, end inclusive). Fits write to OUT_DIR (not the repo).
Usage: gas_floor.py DATA_DIR OUT_DIR fit   -> OUT_DIR/gas_floor.json, gas_floor_pred_*.parquet
       gas_floor.py DATA_DIR OUT_DIR fig "TITLE"   -> research/coupled-auction/gas_floor.png (TITLE states the finding)
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd

SP, OUT, STAGE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
TZ = "Europe/Berlin"; BASE = 15.0; BLOCKS = [0, 6, 10, 16, 20, 24]
TESTS = {"into_winter": ("2024-08-01", "2025-11-01", "2026-03-01"),      # fit [a, b), test [b, c)
         "out_of_winter": ("2025-03-01", "2026-04-01", "2026-08-12")}
MODELS = ("rw", "hdd", "hdd_rw")


def load():
    X = pd.read_parquet(SP / "out/recon_hourly.parquet").tz_convert(TZ)
    with gzip.open(SP / "appdata/gasprices_the_v1_DE_LU.json.gz", "rt") as f: g = json.load(f)["gasprice"]
    gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in g.items()}).sort_index().tz_convert(TZ)
    eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
    X["gas"] = gas.reindex(X.index, method="ffill"); X["co2"] = eua.reindex(X.index.tz_localize(None).normalize(), method="ffill").to_numpy()
    X["m"] = X.p - (1.85 * X.gas + 0.37 * X.co2)
    X = X.dropna(subset=["m", "gasg"]).copy()
    grid = X.m.resample("1h").mean()
    X["m_win"] = grid.rolling(7, center=True, min_periods=5).mean().reindex(X.index)
    X = X.dropna(subset=["m_win"])
    X["b"] = np.digitize(X.index.hour, BLOCKS[1:-1])
    # weather store: 15-min UTC rows, temp_0..temp_5 are the DE_LU land points (temp_6..10 are North Sea points)
    w = pd.read_json(SP / "appdata/weather_v2_DE_LU.json.gz", compression="gzip")
    if pd.api.types.is_integer_dtype(w.index.dtype): w.index = pd.to_datetime(w.index, unit="ms", utc=True)
    elif w.index.tz is None: w.index = w.index.tz_localize("UTC")        # the store writes UTC times
    t = w[[f"temp_{i}" for i in range(6)]].mean(axis=1).tz_convert(TZ)
    tday = t.groupby(t.index.tz_localize(None).normalize()).mean()
    hdd = (BASE - tday).clip(lower=0)
    X["day"] = X.index.tz_localize(None).normalize()
    X["hdd"] = hdd.reindex(X.day).to_numpy(); X["tday"] = tday.reindex(X.day).to_numpy()
    return X.dropna(subset=["hdd"])


def fit_and_predict(fit, test, kind, seed=0):
    import jax, jax.numpy as jnp
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    mcode = pd.factorize(fit.index.to_period("M"), sort=True)[0]; NM = int(mcode.max() + 1)

    def model(m, mw, b, mo, hdd, y=None):
        lam = numpyro.sample("lam", dist.Beta(1.0, 1.0)); me = m + lam * (mw - m)
        mu = numpyro.sample("mu", dist.Normal(0.0, 50.0)); log_s = numpyro.sample("log_s", dist.Normal(np.log(30.0), 1.0))
        tau_mu = numpyro.sample("tau_mu", dist.HalfNormal(30.0)); tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.5))
        mu_b = numpyro.deterministic("mu_b", mu + tau_mu * numpyro.sample("z_mu", dist.Normal(0, 1).expand([5])))
        s_b = numpyro.deterministic("s_b", jnp.exp(log_s + tau_s * numpyro.sample("z_s", dist.Normal(0, 1).expand([5]))))
        A = numpyro.sample("A", dist.HalfNormal(15.0))
        if kind == "rw":
            f0 = numpyro.sample("f0", dist.HalfNormal(5.0)); tau_f = numpyro.sample("tau_f", dist.HalfNormal(1.0))
            Fm = numpyro.deterministic("Fm", jnp.clip(f0 + jnp.cumsum(jnp.concatenate([jnp.zeros(1), tau_f * numpyro.sample("z_f", dist.Normal(0, 1).expand([NM - 1]))])), 0.0, None))
            F = Fm[mo]
        else:
            f0 = numpyro.sample("f0", dist.Normal(0.0, 5.0)); f1 = numpyro.sample("f1", dist.Normal(0.0, 1.0))
            F = f0 + f1 * hdd
            if kind == "hdd_rw":
                tau_f = numpyro.sample("tau_f", dist.HalfNormal(0.3))
                r = numpyro.deterministic("r", jnp.cumsum(jnp.concatenate([jnp.zeros(1), tau_f * numpyro.sample("z_f", dist.Normal(0, 1).expand([NM - 1]))])))
                F = F + r[mo]
            F = jnp.clip(F, 0.0, None)
        mean = F + A * jax.nn.sigmoid((me - mu_b[b]) / s_b[b])
        numpyro.sample("y", dist.StudentT(numpyro.sample("nu", dist.Gamma(4.0, 0.5)), mean, numpyro.sample("sigma", dist.HalfNormal(2.0))), obs=y)

    a = lambda d, c: d[c].to_numpy()
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=800, num_samples=600, num_chains=4, progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), a(fit, "m"), a(fit, "m_win"), a(fit, "b"), mcode, a(fit, "hdd"), a(fit, "gasg"))
    S = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    rhat = max(float(np.nanmax(np.asarray(v["r_hat"]))) for v in summ.values())
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())

    def floor_of(d, outside):
        h = a(d, "hdd")
        if kind == "rw":
            return S["Fm"][:, -1][:, None] * np.ones(len(d)) if outside else S["Fm"][:, pd.factorize(d.index.to_period("M"), sort=True)[0]]
        F = S["f0"][:, None] + S["f1"][:, None] * h
        if kind == "hdd_rw":
            F = F + (S["r"][:, -1][:, None] if outside else S["r"][:, mcode])
        return np.clip(F, 0.0, None)

    def predict(d, outside):
        rng = np.random.default_rng(1); lam = S["lam"][:, None]; b = a(d, "b")
        me = a(d, "m") + lam * (a(d, "m_win") - a(d, "m"))
        F = floor_of(d, outside)
        mean = F + S["A"][:, None] / (1 + np.exp(-(me - S["mu_b"][:, b]) / S["s_b"][:, b]))
        pred = mean + S["sigma"][:, None] * rng.standard_t(S["nu"][:, None], size=mean.shape)
        return pd.DataFrame({"y": a(d, "gasg"), "med": np.median(pred, 0), "lo": np.quantile(pred, 0.05, 0), "hi": np.quantile(pred, 0.95, 0),
                             "floor": np.median(F, 0), "hdd": a(d, "hdd"), "m": a(d, "m"), "mu_b": np.median(S["mu_b"], 0)[b],
                             "s_b": np.median(S["s_b"], 0)[b]}, index=d.index)

    q = lambda v: np.quantile(v, [0.05, 0.5, 0.95]).round(3).tolist()
    r = {"divergences": div, "max_rhat": round(rhat, 4), "A": q(S["A"]), "lam": q(S["lam"]), "sigma": q(S["sigma"]), "nu": q(S["nu"]),
         "mu_b": np.median(S["mu_b"], 0).round(1).tolist(), "s_b": np.median(S["s_b"], 0).round(1).tolist()}
    if kind == "rw":
        r["F_month"] = np.median(S["Fm"], 0).round(2).tolist(); r["F_last"] = q(S["Fm"][:, -1])
    else:
        r["f0"] = q(S["f0"]); r["f1"] = q(S["f1"])
        if kind == "hdd_rw": r["tau_f"] = q(S["tau_f"]); r["r_month"] = np.median(S["r"], 0).round(2).tolist()
    return r, predict(fit, False), predict(test, True)


def score(P):
    return {"mae": round(float((P.y - P.med).abs().mean()), 3), "cov90": round(float(((P.y >= P.lo) & (P.y <= P.hi)).mean()), 3),
            "bias": round(float((P.y - P.med).mean()), 3), "hours": int(len(P))}


def monthly(P):
    low = (P.med - P.floor) < 0.5          # floor hours: the curve adds less than 0.5 GW to the floor
    g = P.assign(err=P.y - P.med, low=low).groupby(P.index.to_period("M"))
    return {str(k): {"mae": round(float(v.err.abs().mean()), 2), "bias": round(float(v.err.mean()), 2),
                     "floor_hours_bias": round(float(v.err[v.low].mean()), 2) if v.low.any() else None, "n_floor_hours": int(v.low.sum()),
                     "hdd": round(float(v.hdd.mean()), 1), "y": round(float(v.y.mean()), 2)} for k, v in g}


if STAGE == "fit":
    import numpyro; numpyro.set_host_device_count(4)
    X = load(); res = {}
    for name, (a0, b0, c0) in TESTS.items():
        ts = lambda s: pd.Timestamp(s, tz=TZ)
        fit = X[(X.index >= ts(a0)) & (X.index < ts(b0))]; test = X[(X.index >= ts(b0)) & (X.index < ts(c0))]
        res[name] = {"fit_hours": int(len(fit)), "test_hours": int(len(test))}
        for kind in MODELS:
            r, Pf, Pt = fit_and_predict(fit, test, kind)
            r["test"] = score(Pt); r["in_sample"] = score(Pf); r["test_monthly"] = monthly(Pt); r["fit_monthly"] = monthly(Pf)
            res[name][kind] = r
            Pt.to_parquet(OUT / f"gas_floor_pred_{name}_{kind}.parquet"); Pf.to_parquet(OUT / f"gas_floor_fit_{name}_{kind}.parquet")
            print(name, kind, json.dumps({k: r[k] for k in r if k not in ("test_monthly", "fit_monthly")}), flush=True)
            json.dump(res, open(OUT / "gas_floor.json", "w"), indent=1)

if STAGE == "fig":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = json.load(open(OUT / "gas_floor.json"))
    for name in TESTS:                      # monthly tables from the saved predictions, with the current floor-hour rule
        for kind in MODELS:
            R[name][kind]["test_monthly"] = monthly(pd.read_parquet(OUT / f"gas_floor_pred_{name}_{kind}.parquet"))
            R[name][kind]["fit_monthly"] = monthly(pd.read_parquet(OUT / f"gas_floor_fit_{name}_{kind}.parquet"))
    json.dump(R, open(OUT / "gas_floor.json", "w"), indent=1)
    BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#222222", "#555555", "#e6e6e6", "#fbfbf9"
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.facecolor": BG, "figure.facecolor": BG,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, name in enumerate(TESTS):
        a = ax[0, j]
        H = pd.read_parquet(OUT / f"gas_floor_pred_{name}_hdd.parquet"); W = pd.read_parquet(OUT / f"gas_floor_pred_{name}_rw.parquet")
        dH = H.groupby(H.index.tz_localize(None).normalize()).mean(); dW = W.groupby(W.index.tz_localize(None).normalize()).mean()
        a.fill_between(dH.index, dH.lo, dH.hi, color=BLUE, alpha=0.15, lw=0, label="HDD floor 90% band (daily mean of hourly bounds)")
        a.plot(dH.index, dH.y, color=INK, lw=1.4, label="observed")
        a.plot(dH.index, dH.med, color=BLUE, lw=2, label=f"HDD floor: MAE {R[name]['hdd']['test']['mae']:.2f} GW, cov90 {R[name]['hdd']['test']['cov90']:.0%}")
        a.plot(dW.index, dW.med, color=ORANGE, lw=2, label=f"carried forward: MAE {R[name]['rw']['test']['mae']:.2f} GW, cov90 {R[name]['rw']['test']['cov90']:.0%}")
        a.set_title(f"{name.replace('_', ' ')}: daily mean of hourly gas output (GW), test window", loc="left", fontsize=10)
        a.set_ylim(0, 1.75 * max(dH.y.max(), dH.hi.max(), dW.med.max())); a.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.4, fontsize=8)
        a.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    # floor vs HDD: daily mean gas output in hours where the curve sits on its floor, all fitted and test hours of both tests
    a = ax[1, 0]
    P = pd.concat([pd.read_parquet(OUT / f"gas_floor_{k}_{n}_hdd.parquet") for n in TESTS for k in ("fit", "pred")])
    P = P[~P.index.duplicated()].sort_index(); low = P[(P.med - P.floor) < 0.5]
    d = low.groupby(low.index.tz_localize(None).normalize()).agg(y=("y", "mean"), hdd=("hdd", "first"), n=("y", "size"))
    d = d[d.n >= 3]; d = d[d.index < pd.Timestamp("2026-08-12")]
    a.scatter(d.hdd, d.y, s=9, color=INK2, alpha=0.45, lw=0, label="daily mean in floor hours (curve adds < 0.5 GW)")
    hx = np.linspace(0, max(20, d.hdd.max()), 50)
    for name, ls in zip(TESTS, ("-", "--")):
        f0, f1 = R[name]["hdd"]["f0"][1], R[name]["hdd"]["f1"][1]
        a.plot(hx, np.clip(f0 + f1 * hx, 0, None), color=BLUE, ls=ls, lw=2, label=f"fitted floor F, {name.replace('_', ' ')} fit: {f0:.2f} + {f1:.2f}·HDD")
    a.set_xlabel("heating degree days (base 15 °C), mean of six German points"); a.set_ylabel("GW")
    a.set_title("Floor-hour gas output against HDD, fit and test hours of both tests", loc="left", fontsize=10); a.legend(frameon=False, fontsize=8.5, loc="upper left")
    # monthly mean test error, all hours, both models
    a = ax[1, 1]; labels, xs = [], 0
    for name in TESTS:
        for mo, v in R[name]["hdd"]["test_monthly"].items():
            w = R[name]["rw"]["test_monthly"][mo]
            a.bar(xs - 0.2, v["bias"], 0.38, color=BLUE); a.bar(xs + 0.2, w["bias"], 0.38, color=ORANGE)
            labels.append(mo[2:]); xs += 1
        xs += 0.6; labels.append("")
    pos = []; xs = 0
    for name in TESTS:
        for mo in R[name]["hdd"]["test_monthly"]: pos.append((xs, mo[2:])); xs += 1
        xs += 0.6
    a.set_xticks([p for p, _ in pos], [l for _, l in pos], rotation=45, fontsize=8)
    a.axhline(0, color=INK2, lw=0.8)
    from matplotlib.patches import Patch
    a.set_ylabel("observed − predicted median (GW)")
    a.legend(handles=[Patch(color=BLUE, label="HDD floor"), Patch(color=ORANGE, label="carried-forward floor")], frameon=False, fontsize=8.5)
    a.set_title("Monthly mean test error (left group: into winter, right: out of winter)", loc="left", fontsize=10)
    fig.suptitle(sys.argv[4] if len(sys.argv) > 4 else "gas floor", fontsize=12, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(Path(__file__).parent / "gas_floor.png", dpi=130)
