"""
Latent gas supply curve with hour-specific price elasticity (notes 21.167). Direct likelihood on hourly realised gas
output against the margin over gas-fired cost; fit before a test window, predict the window. Usage:
gas_curve.py DATA_DIR OUT_DIR [TEST_FIRST TEST_DAYS]. Writes gas_curve.json and gas_curve_draws.npz.
"""
import gzip, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TF = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2026-08-12", tz="Europe/Berlin"); TD = int(sys.argv[4]) if len(sys.argv) > 4 else 14
X = pd.read_parquet(SP / "out/recon_hourly.parquet").tz_convert("Europe/Berlin")
with gzip.open(SP / "appdata/gasprices_the_v1_DE_LU.json.gz", "rt") as f: g = json.load(f)["gasprice"]
gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in g.items()}).sort_index().tz_convert("Europe/Berlin")
eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
X["gas"] = gas.reindex(X.index, method="ffill"); X["co2"] = eua.reindex(X.index.tz_localize(None).normalize(), method="ffill").to_numpy()
X["m"] = X.p - (1.85 * X.gas + 0.37 * X.co2)
X = X.dropna(subset=["m", "gasg"])
BLOCKS = [0, 6, 10, 16, 20, 24]; X["b"] = np.digitize(X.index.hour, BLOCKS[1:-1])
fit = X[(X.index >= TF - pd.DateOffset(months=12)) & (X.index < TF)]; test = X[(X.index >= TF) & (X.index < TF + pd.Timedelta(days=TD))]
months = fit.index.to_period("M"); mcode = pd.factorize(months, sort=True)[0]; NM = mcode.max() + 1

def model(m, b, mo, y=None, pooled=False):
    mu = numpyro.sample("mu", dist.Normal(0.0, 50.0)); log_s = numpyro.sample("log_s", dist.Normal(np.log(30.0), 1.0))
    if pooled:
        mu_b = jnp.full(5, mu); s_b = jnp.full(5, jnp.exp(log_s))
    else:
        tau_mu = numpyro.sample("tau_mu", dist.HalfNormal(30.0)); tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.5))
        mu_b = numpyro.deterministic("mu_b", mu + tau_mu * numpyro.sample("z_mu", dist.Normal(0, 1).expand([5])))
        s_b = numpyro.deterministic("s_b", jnp.exp(log_s + tau_s * numpyro.sample("z_s", dist.Normal(0, 1).expand([5]))))
    A = numpyro.sample("A", dist.HalfNormal(15.0))
    f0 = numpyro.sample("f0", dist.HalfNormal(5.0)); tau_f = numpyro.sample("tau_f", dist.HalfNormal(1.0))
    F = numpyro.deterministic("F", jnp.clip(f0 + jnp.cumsum(jnp.concatenate([jnp.zeros(1), tau_f * numpyro.sample("z_f", dist.Normal(0, 1).expand([NM - 1]))])), 0.0, None))
    mean = F[mo] + A * jax.nn.sigmoid((m - mu_b[b]) / s_b[b])
    numpyro.sample("y", dist.StudentT(numpyro.sample("nu", dist.Gamma(4.0, 0.5)), mean, numpyro.sample("sigma", dist.HalfNormal(2.0))), obs=y)

res = {"fit_hours": int(len(fit)), "test_hours": int(len(test)), "test_first": str(TF.date()), "models": {}}
draws = {}
for pooled in (True, False):
    name = "pooled" if pooled else "blocks"
    mc = MCMC(NUTS(model), num_warmup=600, num_samples=600, num_chains=4, progress_bar=False)
    mc.run(jax.random.PRNGKey(0), fit.m.to_numpy(), fit.b.to_numpy(), mcode, fit.gasg.to_numpy(), pooled)
    S = mc.get_samples(); ex = mc.get_extra_fields()
    div = int(np.asarray(mc.get_extra_fields(group_by_chain=False).get("diverging", np.zeros(1))).sum())
    summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    rhat = float(max(np.nanmax(np.asarray(v["r_hat"])) for v in summ.values())); ess = float(min(np.nanmin(np.asarray(v["n_eff"])) for v in summ.values()))
    # test: floor of the last fitted month carried forward; predictive draws with the Student-t noise
    rng = np.random.default_rng(1); n = len(S["A"]); F_last = np.asarray(S["F"])[:, -1]
    mu_b = np.asarray(S["mu_b"]) if not pooled else np.repeat(np.asarray(S["mu"])[:, None], 5, 1)
    s_b = np.asarray(S["s_b"]) if not pooled else np.repeat(np.exp(np.asarray(S["log_s"]))[:, None], 5, 1)
    mt, bt = test.m.to_numpy(), test.b.to_numpy()
    mean = F_last[:, None] + np.asarray(S["A"])[:, None] / (1 + np.exp(-(mt[None] - mu_b[:, bt]) / s_b[:, bt]))
    pred = mean + np.asarray(S["sigma"])[:, None] * rng.standard_t(np.asarray(S["nu"])[:, None], size=mean.shape)
    lo, hi, med = np.quantile(pred, 0.05, 0), np.quantile(pred, 0.95, 0), np.median(pred, 0)
    yt = test.gasg.to_numpy(); h = test.index.hour
    r = dict(divergences=div, max_rhat=round(rhat, 3), min_ess=round(ess), test_mae=round(float(np.abs(yt - med).mean()), 3),
             test_bias=round(float((yt - med).mean()), 3), test_cov90=round(float(((yt >= lo) & (yt <= hi)).mean()), 3),
             test_mae_by_block={f"{BLOCKS[k]:02d}-{BLOCKS[k+1]:02d}": round(float(np.abs(yt - med)[test.b.to_numpy() == k].mean()), 2) for k in range(5)},
             A=np.quantile(S["A"], [0.05, 0.5, 0.95]).round(2).tolist(), F_last=np.quantile(F_last, [0.05, 0.5, 0.95]).round(2).tolist(),
             mu_b=np.quantile(mu_b, [0.05, 0.5, 0.95], 0).round(1).tolist(), s_b=np.quantile(s_b, [0.05, 0.5, 0.95], 0).round(1).tolist())
    res["models"][name] = r; draws[name] = dict(mu_b=mu_b, s_b=s_b, A=np.asarray(S["A"]), F=np.asarray(S["F"]))
    print(name, json.dumps(r), flush=True)
res["test_mae_gain_blocks_vs_pooled"] = round(1 - res["models"]["blocks"]["test_mae"] / res["models"]["pooled"]["test_mae"], 3)
json.dump(res, open(OUT / "gas_curve.json", "w"), indent=1)
np.savez(OUT / "gas_curve_draws.npz", **{f"{k}_{n}": v for k, d in draws.items() for n, v in d.items()})
print("gain", res["test_mae_gain_blocks_vs_pooled"])
