"""
Shared pieces for the scarcity-response study (notes: research/market-model/notes/scarcity_response.md).

Control = the application's market model in its slim form (predictor/model/marketroute.py): demand = published residual load - 8 GW,
carbon 75 EUR/t, gas price lagged 192 quarter hours, no neighbours, prices in EUR/MWh. The supply curve is rebuilt here block by block
so that the import term can be swapped or held fixed; check() verifies it against predictor.model.marketmodel.

Reads only $SP/out/star_data.parquet (ends 2026-11-09 00:00 UTC). Nothing after 2026-11-08 local is used.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from predictor.model import marketmodel as MM  # noqa: E402

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
P = MM.PARAMETERS
CARBON = 75.0
OTHER = 8.0
GAS_LAG = 192
LEVEL_SLOTS = 672
PMIN, PMAX = -500.0, 4000.0  # auction price range; bisection is confined to it (fixed on FIT before any held-out run)
FIT = ("2024-03-01", "2025-11-30")      # local dates, inclusive
HELD = ("2025-12-01", "2026-11-08")
NB_COLS = ["rl_fr", "rl_nl", "rl_be", "rl_at", "rl_pl", "rl_cz", "rl_ch", "rl_dk"]
LIMIT_Q = (0.01, 0.99)  # trailing-28-day quantiles of net imports for the bounded block's export / import limits


def load():
    """Quarter-hour frame with slim-form inputs, local calendar fields, regime labels and the bounded block's limits."""
    S = pd.read_parquet(f"{SP}/out/star_data.parquet")
    S = S[S.index < pd.Timestamp("2026-11-08 23:00", tz="UTC")]  # = 2026-11-09 00:00 local; nothing later
    f = pd.DataFrame(index=S.index)
    f["p"] = S.p_de.astype(float)
    f["D"] = S.rl_de.astype(float) - OTHER
    gas = S.gas.astype(float)
    f["gas"] = gas.shift(GAS_LAG).fillna(gas.dropna().iloc[0]).to_numpy()
    f["L"] = MM.cost_index(f.gas.to_numpy(), CARBON)
    f["x"] = S.x_sum.astype(float)
    f["nb"] = S[NB_COLS].sum(axis=1, min_count=8)
    loc = f.index.tz_convert("Europe/Berlin")
    f["date"] = loc.tz_localize(None).normalize()
    f["hour"] = loc.hour
    f["month"] = loc.month
    f["wkend"] = loc.weekday >= 5
    f["period"] = np.where((f.date >= FIT[0]) & (f.date <= FIT[1]), "FIT",
                           np.where((f.date >= HELD[0]) & (f.date <= HELD[1]), "HELD", ""))
    f["hkey"] = f.index.floor("1h")
    regimes(f)
    limits(f)
    return f


def regimes(f):
    """Fixed regime rule; thresholds per calendar month from FIT quarter hours only, applied to both periods."""
    fit = f.period == "FIT"
    eve = f.hour.between(17, 20)
    mid = f.hour.between(11, 14)
    q_eve = f[fit & eve].groupby("month").D.quantile(0.8)
    m_nb = f[fit & eve].groupby("month").nb.median()
    q_mid = f[fit & mid].groupby("month").D.quantile(0.2)
    f["thr_eve"] = f.month.map(q_eve)
    f["thr_nb"] = f.month.map(m_nb)
    f["thr_mid"] = f.month.map(q_mid)
    tight = eve & (f.D > f.thr_eve) & (f.nb > f.thr_nb)
    surplus = mid & (f.D < f.thr_mid)
    f["regime"] = np.where(tight, "tight", np.where(surplus, "surplus", "ordinary"))


def limits(f):
    """Export / import limits per local day from net imports of the 28 days ending one day before it (days d-29 .. d-2)."""
    daily = f.groupby("date").x
    days = pd.Index(sorted(f.date.unique()))
    lo, hi = {}, {}
    vals = {d: g.dropna().to_numpy() for d, g in daily}
    for d in days:
        w = [vals[k] for k in pd.date_range(d - pd.Timedelta(days=29), d - pd.Timedelta(days=2)) if k in vals]
        w = np.concatenate(w) if w else np.array([])
        if len(w) >= 96 * 21:
            lo[d], hi[d] = np.quantile(w, LIMIT_Q)
        else:
            lo[d], hi[d] = np.nan, np.nan
    f["xlo"] = f.date.map(lo)
    f["xhi"] = f.date.map(hi)


# ---------------------------------------------------------------- supply curves
def other_blocks(p, L):
    """Everything in marketmodel.supply except the import term (fuels, storage, unmeasured), GW."""
    p = np.asarray(p, float)
    carbon = CARBON
    cost = {"lignite": P["lignite_cost"] + carbon, "coal": P["coal_cost"] + MM.COAL_CARBON * carbon, "gas": P["gas_cost_factor"] * L}
    tot = 0.0
    for fuel, cap in MM.CAPACITY.items():
        m = P[f"{fuel}_mustrun"]
        tot = tot + cap * (m + (1 - m) * P[f"{fuel}_responsive"] * ndtr((p / cost[fuel] - 1.0) / P[f"{fuel}_width"]))
    ratio = p / L
    storage = P["storage_range"] * (2.0 * ndtr((ratio - P["storage_centre"]) / P["storage_width"]) - 1.0)
    return tot + storage + P["unmeasured"]


def step(p):
    return P["import_step"] * (1.0 - ndtr((np.asarray(p, float) - P["import_step_price"]) / P["import_step_width"]))


def imports_control(p, L, **_):
    return P["import_level"] + P["import_slope"] * (np.asarray(p, float) / L - 1.0) - step(p)


def imports_bounded(p, L, lo, hi, c, w):
    return lo + (hi - lo) * ndtr((np.asarray(p, float) / L - c) / w) - step(p)


def stopped(p):
    p = np.asarray(p, float)
    return P["stops_below_zero"] * ((p < -MM.ZERO_BAND) + 0.5 * (np.abs(p) <= MM.ZERO_BAND))


def make_supply(kind, L, lo=None, hi=None, c=None, w=None, fixed_imports=None):
    """Supply function of price for one version: kind 'control' or 'bounded'; fixed_imports holds the import term (way b)."""
    if fixed_imports is not None:
        return lambda p: other_blocks(p, L) + fixed_imports
    if kind == "control":
        return lambda p: other_blocks(p, L) + imports_control(p, L)
    return lambda p: other_blocks(p, L) + imports_bounded(p, L, lo, hi, c, w)


def shift(sup, D, p):
    """Supply shift at the observed price (as marketmodel.supply_shift)."""
    return np.asarray(D, float) - (sup(p) - stopped(p))


def clear(sup, D, extra):
    """Price where sup(p) + extra - stopped = D, with marketmodel's zero step, confined to [PMIN, PMAX]. Returns (price, hit_bound)."""
    D = np.asarray(D, float)
    extra = np.broadcast_to(np.asarray(extra, float), D.shape)
    K = P["stops_below_zero"]
    zb = MM.ZERO_BAND
    ex = lambda p, s: sup(p) + extra - s - D
    above = ex(np.full(D.shape, zb), 0.0) < 0
    below = ex(np.full(D.shape, -zb), K) > 0
    low = np.where(above, zb, PMIN)
    high = np.where(above, PMAX, -zb)
    st = np.where(below, K, 0.0)
    for _ in range(60):
        mid = 0.5 * (low + high)
        short = ex(mid, st) < 0
        low = np.where(short, mid, low)
        high = np.where(short, high, mid)
    price = np.where(above | below, 0.5 * (low + high), 0.0)
    hit = (above & (price > PMAX - 1e-3)) | (below & (price < PMIN + 1e-3))
    return price, hit


def check(f, n=5000, seed=0):
    """Rebuilt control supply and clearing against predictor.model.marketmodel on a random sample."""
    rng = np.random.default_rng(seed)
    i = rng.choice(len(f), n, replace=False)
    g = f.iloc[i]
    p, L, D, gas = g.p.to_numpy(), g.L.to_numpy(), g.D.to_numpy(), g.gas.to_numpy()
    sup = make_supply("control", L)
    d_sup = np.abs(sup(p) - MM.supply(p, gas, CARBON)).max()
    s_mm = MM.supply_shift(D, p, gas, CARBON)
    s_me = shift(sup, D, p)
    d_shift = np.abs(s_mm - s_me).max()
    lvl = rng.normal(0, 4, n)
    c_mm = MM.clearing_price(D, gas, CARBON, shift=lvl)
    c_me, _ = clear(sup, D, lvl)
    ok = (c_mm > PMIN) & (c_mm < PMAX)
    d_clear = np.abs(c_mm - c_me)[ok].max()
    back, _ = clear(sup, D, s_me)
    tgt = np.where(np.abs(p) <= MM.ZERO_BAND, 0.0, p)
    d_back = np.abs(back - tgt)
    return dict(supply=d_sup, shift=d_shift, clear=d_clear, anchor_q99=np.quantile(d_back, 0.99), anchor_share_gt_0p5=(d_back > 0.5).mean())
