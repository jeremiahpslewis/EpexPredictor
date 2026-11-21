"""
Posterior draws of the dynamic gas supply curve by hour block (the "dynamic_blocks" model of gas_curve2.py, notes 21.169),
fitted on the 12 months before a cut-off date and saved for the slice's GASCURVE switch (notes/gas_in_slice.md).
Differences from gas_curve2.py: the cut-off is an argument; the data are truncated BEFORE the centred +-3 h window mean is
built (gas_curve2 built it on the whole series, so the last fitted hours saw up to 3 h of prices after the cut-off); the
draws are saved. The floor saved is the last fitted month's F (the month of the cut-off, partly observed).
Usage: gas_curve_fit.py DATA_DIR OUT.npz FIRST   (FIRST = first delivery day of the window; data used are < FIRST 00:00)
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)

SP, OUT, TF = Path(sys.argv[1]), Path(sys.argv[2]), pd.Timestamp(sys.argv[3], tz="Europe/Berlin")
X = pd.read_parquet(SP / "out/recon_hourly.parquet").tz_convert("Europe/Berlin")
X = X[(X.index >= TF - pd.DateOffset(months=12)) & (X.index < TF)].copy()          # nothing at or after the cut-off
with gzip.open(SP / "appdata/gasprices_the_v1_DE_LU.json.gz", "rt") as f: g = json.load(f)["gasprice"]
gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in g.items()}).sort_index().tz_convert("Europe/Berlin")
eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
X["gas"] = gas.reindex(X.index, method="ffill"); X["co2"] = eua.reindex(X.index.tz_localize(None).normalize(), method="ffill").to_numpy()
X["m"] = X.p - (1.85 * X.gas + 0.37 * X.co2)
X = X.dropna(subset=["m", "gasg"]).copy()
grid = X.m.resample("1h").mean()
X["m_win"] = grid.rolling(7, center=True, min_periods=5).mean().reindex(X.index)
X = X.dropna(subset=["m_win"])
BLOCKS = [0, 6, 10, 16, 20, 24]; X["b"] = np.digitize(X.index.hour, BLOCKS[1:-1])
mcode = pd.factorize(X.index.to_period("M"), sort=True)[0]; NM = mcode.max() + 1

def model(m, mw, b, mo, y=None):
    lam = numpyro.sample("lam", dist.Beta(1.0, 1.0)); me = m + lam * (mw - m)
    mu = numpyro.sample("mu", dist.Normal(0.0, 50.0)); log_s = numpyro.sample("log_s", dist.Normal(np.log(30.0), 1.0))
    tau_mu = numpyro.sample("tau_mu", dist.HalfNormal(30.0)); tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.5))
    mu_b = numpyro.deterministic("mu_b", mu + tau_mu * numpyro.sample("z_mu", dist.Normal(0, 1).expand([5])))
    s_b = numpyro.deterministic("s_b", jnp.exp(log_s + tau_s * numpyro.sample("z_s", dist.Normal(0, 1).expand([5]))))
    A = numpyro.sample("A", dist.HalfNormal(15.0)); f0 = numpyro.sample("f0", dist.HalfNormal(5.0)); tau_f = numpyro.sample("tau_f", dist.HalfNormal(1.0))
    F = numpyro.deterministic("F", jnp.clip(f0 + jnp.cumsum(jnp.concatenate([jnp.zeros(1), tau_f * numpyro.sample("z_f", dist.Normal(0, 1).expand([NM - 1]))])), 0.0, None))
    mean = F[mo] + A * jax.nn.sigmoid((me - mu_b[b]) / s_b[b])
    numpyro.sample("y", dist.StudentT(numpyro.sample("nu", dist.Gamma(4.0, 0.5)), mean, numpyro.sample("sigma", dist.HalfNormal(2.0))), obs=y)

mc = MCMC(NUTS(model), num_warmup=600, num_samples=600, num_chains=4, progress_bar=False)
mc.run(jax.random.PRNGKey(0), X.m.to_numpy(), X.m_win.to_numpy(), X.b.to_numpy(), mcode, X.gasg.to_numpy())
S = mc.get_samples(); summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
d = dict(lam=np.asarray(S["lam"]), mu_b=np.asarray(S["mu_b"]), s_b=np.asarray(S["s_b"]), A=np.asarray(S["A"]), F_last=np.asarray(S["F"])[:, -1])
np.savez(OUT, **d)
q = lambda v: np.quantile(v, [0.05, 0.5, 0.95], axis=0).round(2).tolist()
print(json.dumps(dict(cutoff=str(TF.date()), fit_hours=int(len(X)), first=str(X.index.min()), last=str(X.index.max()), last_month=str(X.index[-1].to_period("M")),
                      divergences=div, max_rhat=round(float(max(np.nanmax(np.asarray(v["r_hat"])) for v in summ.values())), 3),
                      lam=q(d["lam"]), mu_b=np.median(d["mu_b"], 0).round(1).tolist(), s_b=np.median(d["s_b"], 0).round(1).tolist(),
                      A=q(d["A"]), F_last=q(d["F_last"]))))
