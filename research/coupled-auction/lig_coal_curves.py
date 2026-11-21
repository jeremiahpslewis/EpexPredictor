"""
Latent supply curves for lignite and hard coal with a drifting location (notes/lig_coal_curves.md).
output = F_k + A sigmoid((m_eff - mu_k)/s), m_eff = m + lam (m_win - m), m_win = centred +-3 h mean margin (as gas_curve2.py).
F_k: random walk over 4-week blocks counted back from the fit end (last block complete; it is carried forward).
Model "static": one mu. Model "drift": mu_k random walk over the same blocks (non-centred). A, s, lam shared in both.
Lignite m = p - (1.1 co2 + 8); coal m = p - (2.5*12 + 0.85 co2 + 10): coal price fixed at 12, drifting mu absorbs it.
Usage: lig_coal_curves.py DATA_DIR OUT_DIR fit FUEL WINDOW MODEL [SPEC]   (WINDOW in aug, oct, val_aug, val_oct, full)
SPEC v1 (registered first): F walks, A constant. SPEC v2 (after v1 failed in-sample checks): F constant, A walks.
SPEC v0 (baseline added after the tests): nothing walks. SPEC v3 (coal option): F constant, one scale walk multiplying F + A. Validation windows end 14 days before each test window.
       lig_coal_curves.py DATA_DIR OUT_DIR fig FIGPATH
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
WIN = {"aug": "2026-08-12", "oct": "2026-10-15", "val_aug": "2026-07-29", "val_oct": "2026-10-01"}; TD = 14; BL = pd.Timedelta(days=28)


def load(fuel):
    X = pd.read_parquet(SP / "out/recon_hourly.parquet").tz_convert("Europe/Berlin")
    eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
    X["co2"] = eua.reindex(X.index.tz_localize(None).normalize(), method="ffill").to_numpy()
    X["m"] = X.p - (1.1 * X.co2 + 8) if fuel == "lig" else X.p - (2.5 * 12 + 0.85 * X.co2 + 10)
    X["y"] = X[fuel]
    X = X.dropna(subset=["m", "y"]).copy()
    grid = X.m.resample("1h").mean()
    X["m_win"] = grid.rolling(7, center=True, min_periods=5).mean().reindex(X.index)
    return X.dropna(subset=["m_win"])


def blocks(idx, end):
    k = np.floor(np.asarray((end - idx) / BL, dtype=float)).astype(int)   # 0 = most recent 4 weeks
    nb = int(np.floor((end - idx.min()) / BL))            # merge a partial oldest block into the next one
    k = np.minimum(k, max(nb - 1, 0)); NB = int(k.max()) + 1
    return NB - 1 - k, NB


def fit_one(fuel, window, model_name, spec="v1"):
    import jax, jax.numpy as jnp
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    numpyro.set_host_device_count(4)
    X = load(fuel)
    if window == "full":
        end = X.index.max() + pd.Timedelta(hours=1); fit = X; test = X.iloc[:0]
    else:
        end = pd.Timestamp(WIN[window], tz="Europe/Berlin")
        fit = X[(X.index >= end - pd.DateOffset(months=12)) & (X.index < end)]
        test = X[(X.index >= end) & (X.index < end + pd.Timedelta(days=TD))]
    bk, NB = blocks(fit.index, end)
    drift = model_name == "drift"

    def model(m, mw, b, y=None):
        lam = numpyro.sample("lam", dist.Beta(1.0, 1.0)); me = m + lam * (mw - m)
        mu0 = numpyro.sample("mu0", dist.Normal(0.0, 50.0))
        if drift:
            tau_mu = numpyro.sample("tau_mu", dist.HalfNormal(10.0 if spec == "v1" else 5.0))      # EUR/MWh per 4 weeks
            z = numpyro.sample("z_mu", dist.Normal(0, 1).expand([NB - 1]))
            mu = numpyro.deterministic("mu", mu0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_mu * z)]))
        else:
            mu = numpyro.deterministic("mu", jnp.full(NB, mu0))
        s = numpyro.deterministic("s", jnp.exp(numpyro.sample("log_s", dist.Normal(np.log(30.0), 1.0))))
        A = numpyro.sample("A", dist.HalfNormal(15.0)); f0 = numpyro.sample("f0", dist.HalfNormal(8.0))
        tau_f = numpyro.sample("tau_f", dist.HalfNormal(1.0))
        walk = lambda nm, sc: jnp.concatenate([jnp.zeros(1), jnp.cumsum(sc * numpyro.sample(nm, dist.Normal(0, 1).expand([NB - 1])))])
        if spec == "v1":
            F = numpyro.deterministic("F", jnp.clip(f0 + walk("z_f", tau_f), 0.0, None)); Ak = numpyro.deterministic("Ak", jnp.full(NB, A))
        elif spec == "v0":      # fixed curve: nothing walks (baseline added after the tests)
            F = numpyro.deterministic("F", jnp.full(NB, f0)); Ak = numpyro.deterministic("Ak", jnp.full(NB, A))
        elif spec == "v2":
            F = numpyro.deterministic("F", jnp.full(NB, f0)); Ak = numpyro.deterministic("Ak", jnp.clip(A + walk("z_a", tau_f), 0.0, None))
        else:
            g = jnp.exp(walk("z_g", 0.2 * tau_f))      # scale walk, ~0.2 log units per block at tau_f = 1
            F = numpyro.deterministic("F", f0 * g); Ak = numpyro.deterministic("Ak", A * g)
        mean = F[b] + Ak[b] * jax.nn.sigmoid((me - mu[b]) / s)
        numpyro.sample("y", dist.StudentT(numpyro.sample("nu", dist.Gamma(4.0, 0.5)), mean, numpyro.sample("sigma", dist.HalfNormal(2.0))), obs=y)

    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=600, num_samples=600, num_chains=4, progress_bar=False)
    mc.run(jax.random.PRNGKey(0), fit.m.to_numpy(), fit.m_win.to_numpy(), bk, fit.y.to_numpy())
    S = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    rh = {k: float(np.nanmax(np.asarray(v["r_hat"]))) for k, v in summ.items()}
    lam, mu, F, Ak, s = S["lam"], S["mu"], S["F"], S["Ak"], S["s"]
    # in-sample residuals in the saturated region, by block (does A need to drift?)
    me_f = fit.m.to_numpy() + np.median(lam) * (fit.m_win.to_numpy() - fit.m.to_numpy())
    sig_f = 1 / (1 + np.exp(-(me_f - np.median(mu, 0)[bk]) / np.median(s)))
    resid = fit.y.to_numpy() - (np.median(F, 0)[bk] + np.median(Ak, 0)[bk] * sig_f)
    low = sig_f < 0.2
    sat = sig_f > 0.8
    sat_res = [round(float(resid[sat & (bk == k)].mean()), 2) if (sat & (bk == k)).sum() > 20 else None for k in range(NB)]
    low_res = [round(float(resid[low & (bk == k)].mean()), 2) if (low & (bk == k)).sum() > 20 else None for k in range(NB)]
    bstart = [str((end - (NB - k) * BL).date()) for k in range(NB)]; bstart[0] = str(fit.index.min().date())
    r = dict(fuel=fuel, window=window, model=model_name, spec=spec, fit_hours=int(len(fit)), test_hours=int(len(test)),
             divergences=div, draws=int(len(lam)), max_rhat=round(max(rh.values()), 3),
             worst_rhat=sorted(rh.items(), key=lambda kv: -kv[1])[:3],
             lam=np.quantile(lam, [0.05, 0.5, 0.95]).round(3).tolist(), s=round(float(np.median(s)), 1),
             A_med=np.median(Ak, 0).round(2).tolist(), sigma=round(float(np.median(S["sigma"])), 3), nu=round(float(np.median(S["nu"])), 1),
             tau_mu=round(float(np.median(S["tau_mu"])), 2) if drift else None,
             block_start=bstart, mu_med=np.median(mu, 0).round(1).tolist(),
             mu_lo=np.quantile(mu, 0.05, 0).round(1).tolist(), mu_hi=np.quantile(mu, 0.95, 0).round(1).tolist(),
             F_med=np.median(F, 0).round(2).tolist(), sat_resid_by_block=sat_res, low_resid_by_block=low_res,
             corr_lastF_lastmu=round(float(np.corrcoef(F[:, -1], mu[:, -1])[0, 1]), 3) if drift else None,
             corr_lastA_lastmu=round(float(np.corrcoef(Ak[:, -1], mu[:, -1])[0, 1]), 3) if drift and spec != "v0" else None)
    sfx = "" if spec == "v1" else "_" + spec
    if len(test):
        rng = np.random.default_rng(1)
        mt = test.m.to_numpy() + lam[:, None] * (test.m_win.to_numpy() - test.m.to_numpy())
        mean = F[:, -1][:, None] + Ak[:, -1][:, None] / (1 + np.exp(-(mt - mu[:, -1][:, None]) / s[:, None]))
        pred = mean + S["sigma"][:, None] * rng.standard_t(S["nu"][:, None], size=mean.shape)
        lo, hi, med = np.quantile(pred, 0.05, 0), np.quantile(pred, 0.95, 0), np.median(pred, 0); yt = test.y.to_numpy()
        sig_t = 1 / (1 + np.exp(-(np.median(mt, 0) - np.median(mu[:, -1])) / np.median(s)))
        r.update(test_share_upper=round(float((sig_t > 0.8).mean()), 3), test_mae=round(float(np.abs(yt - med).mean()), 3), test_bias=round(float((med - yt).mean()), 3),
                 test_cov90=round(float(((yt >= lo) & (yt <= hi)).mean()), 3), test_mean_actual=round(float(yt.mean()), 2))
        pd.DataFrame({"y": yt, "med": med, "lo": lo, "hi": hi}, index=test.index).to_parquet(OUT / f"pred_{fuel}_{window}_{model_name}{sfx}.parquet")
    json.dump(r, open(OUT / f"fit_{fuel}_{window}_{model_name}{sfx}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("block_start", "mu_lo", "mu_hi")}), flush=True)


def fig(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = lambda f, w, m: json.load(open(OUT / f"fit_{f}_{w}_{m}_v2.json"))      # figure shows the v2 specification
    C = {"static": "#8a8a8a", "drift": "#2a6fb0", "full": "#c0602a", "fixed": "#b8a05a"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    for i, (f, nm) in enumerate([("lig", "Lignite"), ("coal", "Hard coal")]):
        a = ax[i, 0]
        full = OUT / f"fit_{f}_full_drift_v2.json"
        r = json.load(open(full)) if full.exists() else None
        if r and r["max_rhat"] < 1.05:      # the coal all-data fit did not converge; not drawn
            t = pd.to_datetime(r["block_start"]) + pd.Timedelta(days=14)
            a.fill_between(t, r["mu_lo"], r["mu_hi"], color=C["full"], alpha=0.15, lw=0)
            a.plot(t, r["mu_med"], color=C["full"], lw=1.5, label="all data, descriptive")
        for w, ls in (("aug", ":"), ("oct", "-")):
            r = R(f, w, "drift"); t = pd.to_datetime(r["block_start"]) + pd.Timedelta(days=14)
            a.plot(t, r["mu_med"], color=C["drift"], ls=ls, lw=1.8, label=f"fit ending {WIN[w]}")
            rs = R(f, w, "static"); a.hlines(rs["mu_med"][0], t[0], t[-1], color=C["static"], ls=ls, lw=1.2,
                                              label=f"static μ, fit ending {WIN[w]}")
        lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.6 * (hi_ - lo_))
        a.axhline(0, color="k", lw=0.5); a.set_ylabel("μ, half-utilisation margin (EUR/MWh)")
        a.set_title(f"{nm}: location μ by 4-week block" + (" (90% band: all-data fit, descriptive)" if f == "lig" else " (all-data fit did not converge)"), fontsize=10, loc="left")
        a.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
        for j, w in enumerate(("aug", "oct")):
            a = ax[i, j + 1]
            ps, pdr = pd.read_parquet(OUT / f"pred_{f}_{w}_static_v2.parquet"), pd.read_parquet(OUT / f"pred_{f}_{w}_drift_v2.parquet")
            t = ps.index.tz_localize(None)
            a.fill_between(t, pdr.lo, pdr.hi, color=C["drift"], alpha=0.15, lw=0)
            a.plot(t, ps.y, color="k", lw=0.9, label="actual")
            rs, rd = R(f, w, "static"), R(f, w, "drift")
            p0 = pd.read_parquet(OUT / f"pred_{f}_{w}_static_v0.parquet"); r0 = json.load(open(OUT / f"fit_{f}_{w}_static_v0.json"))
            a.plot(t, p0.med, color=C["fixed"], lw=1.0, label=f"fixed curve MAE {r0['test_mae']:.2f}, cov {r0['test_cov90']:.2f}")
            a.plot(t, ps.med, color=C["static"], lw=1.1, label=f"A walk  MAE {rs['test_mae']:.2f}, cov {rs['test_cov90']:.2f}")
            a.plot(t, pdr.med, color=C["drift"], lw=1.1, label=f"A + μ walk MAE {rd['test_mae']:.2f}, cov {rd['test_cov90']:.2f}")
            lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
            a.set_title(f"{nm}, test {WIN[w]} + 14 d (GW)", fontsize=10, loc="left"); a.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
            a.tick_params(axis="x", labelsize=7, rotation=30)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    title = sys.argv[5] if len(sys.argv) > 5 else "Lignite and hard coal supply curves"
    fig.suptitle(title, fontsize=12, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    if sys.argv[3] == "fit":
        fit_one(sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] if len(sys.argv) > 7 else "v1")
    else:
        fig(sys.argv[4])
