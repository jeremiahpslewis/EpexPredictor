"""Offline Timestral variants on dumped inputs (dump_ts.py). Each variant returns the forecast on D's slots.
Covariate rows as TimestralBlend.covariates builds them: wind onshore mean, wind power proxy, irradiance, temp, holiday,
[offshore wind], [eu_wind, eu_wind_northsea, eu_irradiance, gasprice], [residual_load] - names from the df columns."""
import argparse, glob, os, time, json
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
REV = {"mistralai/Timestral-28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
       "mistralai/Timestral-123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
       "mistralai/Timestral-209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"}
_M = {}
def model(name):
    if name not in _M: _M[name] = TimestralForecaster.from_pretrained(name, revision=REV[name], local_files_only=True)
    return _M[name]

def load(path):
    z = dict(np.load(path, allow_pickle=True)); z["n_hist"] = int(z["n_hist"]); return z

def _unused_cov_names(z):
    df = pd.read_parquet(path_parquet(z)); n = ["wind_on", "wind_pow", "irr", "temp", "holiday"]
    wind = df.filter(regex=r"^wind_\d+$")
    if z["cov"].shape[0] > 5 + 4 + 1 and False: pass
    extra = [c for c in ("eu_wind", "eu_wind_northsea", "eu_irradiance", "gasprice") if c in df.columns]
    k = z["cov"].shape[0] - 5 - len(extra) - (1 if "residual_load" in df.columns else 0)
    return n + ["wind_off"] * k + extra + (["residual_load"] if "residual_load" in df.columns else [])

def path_parquet(z): return z["_path"].replace(".npz", ".parquet")

def run(z, ctx=8192, name="mistralai/Timestral-123M", drop=(), rl_override=None, past=None, sym=True, extra_known=None, median=True):
    """rl_override: array for D's slots of the RL covariate; past: (C, n_hist) past-only covariate rows"""
    n = z["n_hist"]; prices = z["prices"]; cov = z["cov"].copy(); wd = z["wd"]; names = z["names"]; H = cov.shape[1] - n
    keep = [i for i, nm in enumerate(names) if nm not in drop]
    D = pd.DatetimeIndex(pd.to_datetime(z["D"], unit="us", utc=True)); idx = pd.DatetimeIndex(pd.to_datetime(z["idx"], unit="us", utc=True))
    if rl_override is not None:
        ri = list(names).index("residual_load"); pos = idx.get_indexer(D); cov[ri, pos] = rl_override
    if extra_known is not None: cov = np.vstack([cov, extra_known])
    cov = cov[keep + list(range(len(names), cov.shape[0]))]
    s = n - ctx
    kw = {}
    if past is not None: kw["past_covariates"] = past[None, :, s:n]
    with torch.no_grad():
        f = model(name).forecast(prices[None, None, s:n], prediction_length=H, known_covariates=cov[None, :, s:],
                                 categorical_known_covariates=wd[None, None, s:], symmetric_averaging=sym, **kw)
    out = np.asarray(f.median)[0, 0]
    fut = idx[n:n + H]; return pd.Series(out, index=fut).reindex(D).to_numpy()

def run_joint(z, nbp, ctx=8192, name="mistralai/Timestral-123M"):
    """DE and the neighbours as jointly forecast variates (shared known covariates); DE row returned"""
    n = z["n_hist"]; s = n - ctx; H = z["cov"].shape[1] - n
    series = np.vstack([z["prices"][None, :n], nbp])[:, s:n]
    D = pd.DatetimeIndex(pd.to_datetime(z["D"], unit="us", utc=True)); idx = pd.DatetimeIndex(pd.to_datetime(z["idx"], unit="us", utc=True))
    with torch.no_grad():
        f = model(name).forecast(series[None], prediction_length=H, known_covariates=z["cov"][None, :, s:],
                                 categorical_known_covariates=z["wd"][None, None, s:])
    return pd.Series(np.asarray(f.median)[0, 0], index=idx[n:n + H]).reindex(D).to_numpy()

NB_PAST = ("AT", "FR", "NL", "BE", "DK1", "DK2", "SE4")
def leak_check(z):
    """the context (DE and neighbours, same index) ends at the slot before D's first; no neighbour value after it"""
    n = z["n_hist"]; idx = pd.to_datetime(z["idx"], unit="us", utc=True); D0 = pd.to_datetime(z["D"][0], unit="us", utc=True)
    assert idx[n - 1] + pd.Timedelta(minutes=15) == D0, (idx[n - 1], D0)
    assert z["nb"].shape[1] == n, z["nb"].shape
    if "cutoff" in z: assert int(z["cutoff"]) == idx[n - 1].value

def rl_matched_D(z):
    day = pd.to_datetime(z["D"][0], unit="us", utc=True).tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
    for d in ("out_strict_rlmatched", "out_fresh_strict_rlmatched"):
        f = f"{SP}/faithful/{d}/{day}.npz"
        if os.path.exists(f): return np.load(f, allow_pickle=True)["rl_mixed"].astype(float)
    raise FileNotFoundError(day)

def variants(z):
    leak_check(z)
    n = z["n_hist"]; names = list(z["nb_names"]); nb = z["nb"]
    past = np.stack([nb[names.index(c)] for c in NB_PAST if c in names])[:, :n]
    return {
        "base": lambda: run(z),
        "noRL": lambda: run(z, drop=("residual_load",)),
        "nbpast": lambda: run(z, past=past),
        "m209": lambda: run(z, name="mistralai/Timestral-209M"),
        "m28": lambda: run(z, name="mistralai/Timestral-28M"),
        "nosym": lambda: run(z, sym=False),
        "rlm": lambda: run(z, rl_override=rl_matched_D(z)),            # matched RL on D's slots only (approximation)
        "nbpast_rlm": lambda: run(z, past=past, rl_override=rl_matched_D(z)),
        "nbpast209": lambda: run(z, past=past, name="mistralai/Timestral-209M"),
        "nbpast3": lambda: run(z, past=np.stack([nb[names.index(c)] for c in ("AT", "FR", "NL")])[:, :n]),
        "nbjoint": lambda: run_joint(z, past),
        "nogas_eu": lambda: run(z, drop=("gasprice", "eu_wind", "eu_wind_northsea", "eu_irradiance")),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", required=True); ap.add_argument("--dirs", default="ctx8k,smoke")
    ap.add_argument("--budget", type=float, default=1080); a = ap.parse_args()
    torch.set_num_threads(1); t0 = time.time(); outd = f"{SP}/C/tsvar"; os.makedirs(outd, exist_ok=True)
    files = sorted(sum([glob.glob(f"{SP}/C/tsdump/{d}/*.npz") for d in a.dirs.split(",")], []))
    for v in a.variants.split(","):
        for f in files:
            day = os.path.basename(f)[:10]; fo = f"{outd}/{v}_{day}.npy"
            if os.path.exists(fo): continue
            if time.time() - t0 > a.budget: print("budget reached"); raise SystemExit
            z = load(f); z["_path"] = f
            if "names" not in z:   # smoke dump predates the names field
                df = pd.read_parquet(f.replace(".npz", ".parquet"))
                nm = ["wind_on", "wind_pow", "irr", "temp", "holiday"] + (["wind_off"] if z["cov"].shape[0] == 11 else []) + \
                     [c for c in ("eu_wind", "eu_wind_northsea", "eu_irradiance", "gasprice") if c in df.columns] + ["residual_load"]
                z["names"] = np.array(nm)
            ts = time.time(); y = variants(z)[v](); np.save(fo, y)
            print(f"{v} {day} MAE {np.nanmean(np.abs(y - z['actual'])):.3f} {time.time()-ts:.0f}s", flush=True)
