"""Timestral zero-shot on the German price, same rolling-origin setup: context = prices before the cutoff, forecast 289 steps,
weather / gas / calendar as covariates. usage: ts_test.py NAME MODEL CONTEXT START END STRIDE [nocov]"""
import sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
name, model, ctx, start, end, stride = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
nocov = len(sys.argv) > 7
H = 289
df = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
wind = df[[c for c in df.columns if c.startswith("wind_")]]
cov = pd.DataFrame({
    "wind_land": wind.iloc[:, :6].mean(axis=1), "wind_sea": wind.iloc[:, 6:].mean(axis=1),
    "wind_power": ((wind.clip(upper=45) / 45) ** 3).mean(axis=1), "eu_wind": df["eu_wind"], "eu_ns": df["eu_wind_northsea"],
    "irr": df[[c for c in df.columns if c.startswith("irradiance_")]].mean(axis=1), "eu_irr": df["eu_irradiance"],
    "temp": df[[c for c in df.columns if c.startswith("temp_")]].mean(axis=1), "holiday": (df["holiday"] >= 0.5).astype(float),
}).to_numpy(float).T
gas = df["gasprice"].to_numpy(float); wd = df["weekday"].to_numpy(float); y = df["price"].to_numpy(float); idx = df.index
cuts = []; d = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
while d < e - pd.Timedelta(days=3): cuts.append(idx.get_loc(d)); d += pd.Timedelta(days=stride)
f = TimestralForecaster.from_pretrained(model)
preds, q10, q90 = [], [], []; t0 = time.time()
for n, d0 in enumerate(cuts):
    a, b = d0 - ctx, d0 + H
    context = y[a:d0][None, None, :]
    kw = {}
    if not nocov:
        g = gas[a:b].copy(); g[ctx:] = gas[d0]                      # forecasts only know the gas price as of the cutoff
        kw["known_covariates"] = np.concatenate([cov[:, a:b], g[None]], axis=0)[None]
        kw["categorical_known_covariates"] = wd[a:b][None, None, :]
    with torch.no_grad():
        fc = f.forecast(context, prediction_length=H, **kw)
    preds.append(np.asarray(fc.median)[0, 0]); q10.append(np.asarray(fc.quantile(0.1))[0, 0]); q90.append(np.asarray(fc.quantile(0.9))[0, 0])
    if n == 0: print("first forecast took %.1fs" % (time.time() - t0), flush=True)
np.savez_compressed(f"{SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), actual=np.stack([y[d0:d0 + H] for d0 in cuts]).astype(np.float32),
                    **{name: np.stack(preds).astype(np.float32), name + "_q10": np.stack(q10).astype(np.float32), name + "_q90": np.stack(q90).astype(np.float32)})
print(name, len(cuts), "cutoffs in %.0fs" % (time.time() - t0), flush=True)
