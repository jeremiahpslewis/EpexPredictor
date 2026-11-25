"""Shared loaders for the distillation study: the same members, mask, fallback and bootstrap as ../ensemble_bt.py."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
E11 = "/tmp/claude-0/-home-user-EpexPredictor/1711225d-3f79-42cf-ba0d-e4080065c4b2/scratchpad/e11"
O = f"{SP}/out"; DAYS = slice(5, 66); FALLBACK = 10.07
sys.path.insert(0, "/home/user/EpexPredictor/research/market-model/exp")

def struct_member(path, cut, variant="b"):
    R = pd.read_pickle(path); R = R[R.variant == variant]
    s_ = pd.Series(R.p_draws.apply(lambda d: np.nanmedian(d) if np.isfinite(d).any() else np.nan).to_numpy() / 10, index=pd.DatetimeIndex(R.t))
    return np.stack([s_.reindex(c + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy() for c in cut])

def series_to_grid(s, cut):
    return np.stack([s.reindex(c + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy() for c in cut])

def load_members():
    import market1 as M
    I = np.load(f"{O}/inputs_autumn.npz", allow_pickle=True); cut = pd.DatetimeIndex(I["cutoffs"]).tz_localize("UTC")[DAYS]
    a = I["actual"].astype(float)[DAYS, :96]
    T = np.load(f"{O}/rl_E_opt_T.npz")["vint"].astype(float)[DAYS, :96]
    P = np.load(f"{O}/tsm_rllwA_autumn.npz")["median"].astype(float)[DAYS, :96]
    Pm = np.load(f"{O}/market1n_post.npz"); th = {k: float(np.median(Pm[k])) for k in M.PN}; S = np.load(f"{O}/tsshift_lwA_lite_autumn.npz"); gas = I["gas"]
    A = np.stack([np.asarray(M.clear(th, S["xh"][j].astype(float)[:96], np.full(96, 1.85 * float(gas[j]) + 0.37 * 75.0), np.full(96, 75.0), eps=S["median"][j].astype(float)[:96])) / 10
                  for j in range(5, 66)])
    Er = struct_member(f"{E11}/oos11.pkl", cut); Esr = struct_member(f"{E11}/oos11s.pkl", cut)
    E = np.where(np.isnan(Er), FALLBACK, Er); Es = np.where(np.isnan(Esr), FALLBACK, Esr)
    loc = [(c + pd.to_timedelta(np.arange(96) * 15, "min")).tz_convert("Europe/Berlin") for c in cut]
    loc_h = np.stack([l.hour for l in loc])
    return dict(cut=cut, a=a, T=T, P=P, A=A, E=E, Es=Es, Eraw=Er, Esraw=Esr, loc_h=loc_h, I=I, gas=np.asarray(gas, float)[DAYS])

def day_mae(F, a, m=None):
    m = np.ones_like(a, bool) if m is None else m
    return np.array([np.abs(F[d] - a[d])[m[d]].mean() if m[d].any() else np.nan for d in range(len(a))])

_rng = np.random.default_rng(0)
def boot(x, blk=7, n=4000):
    """circular moving-block bootstrap by day (7-day blocks) over all days, so the interval refers to the reported mean
    (ensemble_bt.py drops the last len % 7 days from the resampling instead)"""
    x = np.asarray(x); x = x[np.isfinite(x)]; m = len(x); nb = int(np.ceil(m / blk)); xx = np.r_[x, x[:blk]]
    st = _rng.integers(0, m, (n, nb)); idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(n, -1)[:, :m] % m
    return np.percentile(x[idx].mean(1), [5, 95])

def paired(F1, F2, a, m=None):
    d = day_mae(F1, a, m) - day_mae(F2, a, m); b = boot(d); return f"{d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]"
