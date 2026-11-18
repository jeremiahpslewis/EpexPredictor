"""Shared pieces of the live-setup backtest: forecast issued at 14:00 German time on day D, prices known until the end
of D+1, first unknown slot d0 = local midnight of D+2 (34 h after issue). Forecast rows d0 .. d0+38h."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
ZONES = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]; NS = ["DE", "NL", "BE", "DK1", "DK2"]
S, HZ, L, ISSUE = 96, 153, 540 * 96, 34 * 4
VARS = ("wind", "temp", "irradiance", "pressure", "humidity")
def load():
    F = {z: pd.read_parquet(f"{SP}/frames_real/frame_{z}.parquet") for z in ZONES}
    return F
def vintage(F):
    """per zone: frames (same index) with weather columns as forecast 2 and 3 days earlier, 15-min interpolated"""
    idx = F["DE"].index; V = {}
    for z in ZONES:
        p = pd.read_parquet(f"{SP}/prev_all/prev_{z}.parquet")
        p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="15min")).interpolate(limit=3).reindex(idx)
        V[z] = {k: p[[c for c in p.columns if c.endswith(f"_d{k}")]].rename(columns=lambda c: c[:-3]) for k in (2, 3)}
    return V
def eu_feats(W):
    """W: zone -> DataFrame with wind_i / irradiance_i / temp_i columns. Same definition as predictor/model/europeweather.py"""
    zw, zi, zt = {}, {}, {}
    for z, w in W.items():
        ws = w[[c for c in w.columns if c.startswith("wind_")]].clip(lower=0, upper=45.0)
        zw[z] = ((ws / 45.0) ** 3).mean(axis=1); zi[z] = w[[c for c in w.columns if c.startswith("irradiance_")]].mean(axis=1); zt[z] = w[[c for c in w.columns if c.startswith("temp_")]].mean(axis=1)
    return pd.DataFrame({"eu_wind": pd.concat(zw, axis=1).mean(axis=1), "eu_wind_northsea": pd.concat({z: zw[z] for z in NS}, axis=1).mean(axis=1),
                         "eu_irradiance": pd.concat(zi, axis=1).mean(axis=1), "eu_temperature": pd.concat(zt, axis=1).mean(axis=1)})
def cutoffs(index, start, end, stride):
    out = []; d = pd.Timestamp(start, tz="Europe/Berlin")
    while d <= pd.Timestamp(end, tz="Europe/Berlin"):
        out.append(index.get_loc(d.tz_convert("UTC"))); d += pd.DateOffset(days=stride)
    return out
def test_matrix(F, V, z, d0, cols, use_vintage):
    """feature rows d0..d0+HZ for zone z; weather (and Europe-wide inputs) archived or as forecast at the issue time"""
    X = F[z][cols].iloc[d0:d0 + HZ].astype(float)
    if use_vintage:
        lead_h = np.arange(HZ) / 4.0            # hours after d0; the issue was 34 h before d0
        for zz_cols, k_rows in ((2, lead_h <= 14), (3, lead_h > 14)):
            v = V[z][zz_cols].iloc[d0:d0 + HZ]
            for c in v.columns:
                if c in X.columns: X.loc[X.index[k_rows], c] = v[c].to_numpy()[k_rows]
    return X
