"""
Are the hour-specific gas supply widths (notes 21.168) intertemporal constraints in disguise? (notes 21.169)
(1) residual checks of the static block model; (2) a dynamic curve on m_eff = m + lambda (m_win - m), m_win = centred
+-3 h mean margin, pooled and by block, fitted on the same 12 months and scored on the same test window.
Usage: gas_curve2.py DATA_DIR OUT_DIR (OUT_DIR holds gas_curve_draws.npz from gas_curve.py). Writes gas_curve2.json.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2]); TF = pd.Timestamp("2026-08-12", tz="Europe/Berlin"); TD = 14
X = pd.read_parquet(SP / "out/recon_hourly.parquet").tz_convert("Europe/Berlin")
with gzip.open(SP / "appdata/gasprices_the_v1_DE_LU.json.gz", "rt") as f: g = json.load(f)["gasprice"]
gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in g.items()}).sort_index().tz_convert("Europe/Berlin")
eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
X["gas"] = gas.reindex(X.index, method="ffill"); X["co2"] = eua.reindex(X.index.tz_localize(None).normalize(), method="ffill").to_numpy()
X["m"] = X.p - (1.85 * X.gas + 0.37 * X.co2)
X = X.dropna(subset=["m", "gasg"]).copy()
# centred +-3 h mean margin on a regular hourly grid (gaps stay gaps), and the direction of the margin
grid = X.m.resample("1h").mean()
X["m_win"] = grid.rolling(7, center=True, min_periods=5).mean().reindex(X.index)
X["dm"] = (grid.shift(-1) - grid.shift(1)).reindex(X.index)
X = X.dropna(subset=["m_win", "dm"])
BLOCKS = [0, 6, 10, 16, 20, 24]; X["b"] = np.digitize(X.index.hour, BLOCKS[1:-1])
fit = X[(X.index >= TF - pd.DateOffset(months=12)) & (X.index < TF)]; test = X[(X.index >= TF) & (X.index < TF + pd.Timedelta(days=TD))]
mcode = pd.factorize(fit.index.to_period("M"), sort=True)[0]; NM = mcode.max() + 1
res = {"fit_hours": int(len(fit)), "test_hours": int(len(test))}

# (1) residuals of the static block model (posterior medians from gas_curve.py)
D = np.load(OUT / "gas_curve_draws.npz"); mu_b, s_b = np.median(D["blocks_mu_b"], 0), np.median(D["blocks_s_b"], 0)
A, F = float(np.median(D["blocks_A"])), np.median(D["blocks_F"], 0)
fit_static = F[mcode] + A / (1 + np.exp(-(fit.m.to_numpy() - mu_b[fit.b]) / s_b[fit.b]))
resid = fit.gasg.to_numpy() - fit_static; gap = (fit.m_win - fit.m).to_numpy()
mid = np.abs(fit.m.to_numpy() - mu_b[fit.b]) < 2 * s_b[fit.b]        # where the curve is not flat
slope = float(np.polyfit(gap[mid], resid[mid], 1)[0])
falling, rising = mid & (fit.dm.to_numpy() < -5), mid & (fit.dm.to_numpy() > 5)
res["residual_checks"] = {"slope_resid_on_window_gap_GW_per_EUR": round(slope, 4), "hours_used": int(mid.sum()),
                          "resid_falling_minus_rising_GW": round(float(resid[falling].mean() - resid[rising].mean()), 3),
                          "n_falling": int(falling.sum()), "n_rising": int(rising.sum()),
                          "slope_by_block": {f"{BLOCKS[k]:02d}-{BLOCKS[k+1]:02d}": round(float(np.polyfit(gap[mid & (fit.b.to_numpy() == k)], resid[mid & (fit.b.to_numpy() == k)], 1)[0]), 4) for k in range(5)}}
print(json.dumps(res["residual_checks"]), flush=True)

# (2) dynamic curve
def model(m, mw, b, mo, y=None, blocks=False):
    lam = numpyro.sample("lam", dist.Beta(1.0, 1.0)); me = m + lam * (mw - m)
    mu = numpyro.sample("mu", dist.Normal(0.0, 50.0)); log_s = numpyro.sample("log_s", dist.Normal(np.log(30.0), 1.0))
    if blocks:
        tau_mu = numpyro.sample("tau_mu", dist.HalfNormal(30.0)); tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.5))
        mu_b = numpyro.deterministic("mu_b", mu + tau_mu * numpyro.sample("z_mu", dist.Normal(0, 1).expand([5])))
        s_b = numpyro.deterministic("s_b", jnp.exp(log_s + tau_s * numpyro.sample("z_s", dist.Normal(0, 1).expand([5]))))
    else:
        mu_b = numpyro.deterministic("mu_b", jnp.full(5, mu)); s_b = numpyro.deterministic("s_b", jnp.full(5, jnp.exp(log_s)))
    A = numpyro.sample("A", dist.HalfNormal(15.0)); f0 = numpyro.sample("f0", dist.HalfNormal(5.0)); tau_f = numpyro.sample("tau_f", dist.HalfNormal(1.0))
    F = numpyro.deterministic("F", jnp.clip(f0 + jnp.cumsum(jnp.concatenate([jnp.zeros(1), tau_f * numpyro.sample("z_f", dist.Normal(0, 1).expand([NM - 1]))])), 0.0, None))
    mean = F[mo] + A * jax.nn.sigmoid((me - mu_b[b]) / s_b[b])
    numpyro.sample("y", dist.StudentT(numpyro.sample("nu", dist.Gamma(4.0, 0.5)), mean, numpyro.sample("sigma", dist.HalfNormal(2.0))), obs=y)

for blocks in (False, True):
    name = "dynamic_blocks" if blocks else "dynamic_pooled"
    mc = MCMC(NUTS(model), num_warmup=600, num_samples=600, num_chains=4, progress_bar=False)
    mc.run(jax.random.PRNGKey(0), fit.m.to_numpy(), fit.m_win.to_numpy(), fit.b.to_numpy(), mcode, fit.gasg.to_numpy(), blocks)
    S = mc.get_samples(); summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    rng = np.random.default_rng(1); lam = np.asarray(S["lam"]); mub, sb = np.asarray(S["mu_b"]), np.asarray(S["s_b"])
    mt = test.m.to_numpy() + lam[:, None] * (test.m_win.to_numpy() - test.m.to_numpy()); bt = test.b.to_numpy()
    mean = np.asarray(S["F"])[:, -1][:, None] + np.asarray(S["A"])[:, None] / (1 + np.exp(-(mt - mub[:, bt]) / sb[:, bt]))
    pred = mean + np.asarray(S["sigma"])[:, None] * rng.standard_t(np.asarray(S["nu"])[:, None], size=mean.shape)
    lo, hi, med = np.quantile(pred, 0.05, 0), np.quantile(pred, 0.95, 0), np.median(pred, 0); yt = test.gasg.to_numpy()
    smed = np.median(sb, 0); mmed = np.median(mub, 0)
    r = dict(divergences=div, max_rhat=round(float(max(np.nanmax(np.asarray(v["r_hat"])) for v in summ.values())), 3),
             lam=np.quantile(lam, [0.05, 0.5, 0.95]).round(3).tolist(), mu_b=mmed.round(1).tolist(), s_b=smed.round(1).tolist(),
             width_ratio=round(float(smed.max() / smed.min()), 2), location_spread=round(float(mmed.max() - mmed.min()), 1),
             A=round(float(np.median(S["A"])), 2), test_mae=round(float(np.abs(yt - med).mean()), 3), test_cov90=round(float(((yt >= lo) & (yt <= hi)).mean()), 3))
    res[name] = r; print(name, json.dumps(r), flush=True)
json.dump(res, open(OUT / "gas_curve2.json", "w"), indent=1)
