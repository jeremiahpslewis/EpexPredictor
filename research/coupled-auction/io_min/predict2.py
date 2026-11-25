"""
Two-zone benchmark out-of-sample forecasts (prereg/OOS_PROTOCOL.md). Usage:
    predict2.py PANEL_DIR POST.npz OUT.pkl START END NPOST NSHOCK SEED
Hourly. Inputs for hour h are those known before the gate:
- residual-load forecasts rl_h and DN_h;
- fuel and carbon prices and availability of h;
- German other plant / storage output and the CH exchange from h - 24 h (DDE = rl - other - psgen - pump + ch).
Shocks: variant a, NSHOCK draws from the fitted stationary distribution (AR(1) innovations scaled by 1/sqrt(1-rho^2));
variant b, the shock recovered at h - 24 h from that hour's observed (p_DE, p_N, m) and inputs.
Output: every draw's DE price and net import m (GW) per hour and variant (NaN for a miss).
"""
import sys
import numpy as np, pandas as pd, jax, jax.numpy as jnp
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import model as Mo
jax.config.update("jax_enable_x64", True)
PANEL, POST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
START, END = pd.Timestamp(sys.argv[4], tz="UTC"), pd.Timestamp(sys.argv[5], tz="UTC"); NPOST, NSHOCK, SEED = map(int, sys.argv[6:9])
Pn = pd.read_parquet(f"{PANEL}/panel.parquet"); S = dict(np.load(POST)); nS = len(S["h_G"]); DR = np.linspace(0, nS - 1, NPOST).astype(int)
hrs = Pn.index[(Pn.index >= START) & (Pn.index < END)]; lag = hrs - pd.Timedelta(hours=24)
ok = lag.isin(Pn.index); hrs, lag = hrs[ok], lag[ok]
Xh = Pn.loc[hrs].copy(); Xl = Pn.loc[lag]
Xh["DDE"] = Xh.rl.to_numpy() - (Xl.other + Xl.psgen + Xl.pump - Xl.ch).to_numpy()       # variant b: cancels against the lag's DDE
lag2 = hrs - pd.Timedelta(hours=48); Xl2 = Pn.reindex(lag2)
Xa = Xh.copy(); Xa["DDE"] = Xh.rl.to_numpy() - (Xl2.other + Xl2.psgen + Xl2.pump - Xl2.ch).to_numpy()   # variant a: D-2 realised output
Xa["DDE"] = Xa.DDE.fillna(Xh.DDE)
xh, _, _ = Mo.to_inputs(Xh); xa, _, _ = Mo.to_inputs(Xa); xl, obs_l, _ = Mo.to_inputs(Xl)
rng = np.random.default_rng(SEED); rows = []
for i in DR:
    th = Mo.theta_from(S, i); mu, rho, sd, nu = S["mu"][i], S["rho"][i], S["sd_u"][i], float(S["nu"][i]); L = S["Lcorr"][i]
    ub, _, _ = Mo.invert(th, xl, obs_l["pDE"], obs_l["pN"], obs_l["m"])                  # variant b
    us = {"b": [np.asarray(ub)]}
    us["a"] = []
    for _ in range(NSHOCK):
        z = rng.standard_normal((len(hrs), 3)); w = rng.chisquare(nu, (len(hrs), 1)) / nu
        us["a"].append(mu + (z @ (sd[:, None] * L).T) / np.sqrt(w) / np.sqrt(1 - rho ** 2))
    for v, ul in us.items():
        for u in ul:
            c = Mo.clear(th, xa if v == "a" else xh, jnp.asarray(u)); p, m = np.asarray(c["pDE"]), np.asarray(c["m"])
            bad = (p <= Mo.PMIN + 1) | (p >= Mo.PMAX - 1) | ~np.isfinite(p)
            rows.append(pd.DataFrame({"h": hrs, "variant": v, "draw": int(i), "p": np.where(bad, np.nan, p), "m": np.where(bad, np.nan, m)}))
R = pd.concat(rows); R.to_pickle(OUT); print("two-zone forecasts", len(R), "rows; miss share", R.p.isna().groupby(R.variant).mean().round(3).to_dict())
