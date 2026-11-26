"""Agent P: offline market-route lab on the stored faithful arrays (no Timestral rerun).

The market route prices D as clearing_price(demand, gas_lag2, 75, shift_ts), demand = rl_mixed - 8 GW. Because the
clearing price is monotone in the shift, the Timestral shift forecast is recovered exactly (outside the zero band) as
supply_shift(demand, market price, gas_lag2, 75). The history of the shift is worked out from known prices with the
published TSO residual load (as ResidualLoad.get_data serves it for the past). Variants change the shift forecast or
the clearing inputs and are scored paired against the stored market route.
Usage: marketlab.py  -> prints a table; writes SP/P/market/<variant>.npz
"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/user/EpexPredictor")
sys.path.insert(0, os.path.join(HERE, "..", "..", "distill"))
from predictor.model import marketmodel as mm  # noqa: E402
from common import boot  # noqa: E402

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
ST = f"{SP}/fresh_data/stores"
CARBON, OTHER, EUR = 75.0, 8.0, 10.0


def series(name, col):
    s = pd.read_json(f"{ST}/{name}.json.gz")
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    return s.sort_index()[col] if isinstance(col, str) else s.sort_index()


def load_days():
    out = []
    for o, d0, d1 in [("out_strict", "2026-09-08", "2026-11-07"), ("out_fresh_strict", "2026-11-09", "2026-11-26")]:
        for fn in sorted(glob.glob(f"{SP}/faithful/{o}/*.npz")):
            d = os.path.basename(fn)[:10]
            if not d0 <= d <= d1:
                continue
            z = np.load(fn)
            iv = np.asarray(z["idx"]); idx = pd.to_datetime(iv, unit="us" if np.abs(iv).max() < 1e17 else "ns", utc=True)
            out.append({"day": d, "set": o, "idx": idx, **{k: np.asarray(z[k], float) for k in
                        ("trees", "timestral", "market", "core", "actual", "rl_mixed", "rl_true", "rl_trees")}})
    return out


def main():
    pr = series("prices_v3_DE_LU", "price")
    gas = series("gasprices_the_v1_DE_LU", "gasprice")
    gen = series("generation_v1_DE_LU", None)
    rl_pub = gen.tso_load - gen.tso_wind - gen.tso_solar
    days = load_days()
    V = {}
    for r in days:
        idx = r["idx"]; n = len(idx)
        cutoff = idx[0] - pd.Timedelta(minutes=15)
        g2 = gas.reindex(idx - pd.Timedelta(days=2), method="ffill").to_numpy()          # the app's lag (192 qh)
        g1 = gas.loc[:cutoff].iloc[-1] * np.ones(n)                                       # gas day D-1, known at gate
        dem = r["rl_mixed"] - OTHER
        s_ts = mm.supply_shift(dem, r["market"] * EUR, g2, CARBON)
        back = mm.clearing_price(dem, g2, CARBON, shift=s_ts) / EUR
        # history: 28 days of shift from known prices and the published RL
        H = pd.date_range(cutoff - pd.Timedelta(days=28) + pd.Timedelta(minutes=15), cutoff, freq="15min")
        hg = gas.reindex(H - pd.Timedelta(days=2), method="ffill").to_numpy()
        hs = pd.Series(mm.supply_shift(rl_pub.reindex(H).to_numpy() - OTHER, pr.reindex(H).to_numpy() * EUR, hg, CARBON), index=H)
        loc = H.tz_convert("Europe/Berlin"); slot = loc.hour * 4 + loc.minute // 15
        dl = idx.tz_convert("Europe/Berlin"); dslot = dl.hour * 4 + dl.minute // 15
        wdD = dl.weekday[0]
        prof7 = hs[-672:].groupby(slot[-672:]).mean().reindex(dslot).to_numpy()        # last 7 days, slot profile
        same = hs[(loc.weekday >= 5) == (wdD >= 5)]
        lsame = same.index.tz_convert("Europe/Berlin")
        profsame = same.groupby(lsame.hour * 4 + lsame.minute // 15).median().reindex(dslot).to_numpy()
        lvl1 = hs[-96:].mean()
        m7 = np.nanmean(rl_pub.reindex(H[-672:]).to_numpy()) - OTHER
        def price(shift, g=g2, d=dem):
            return mm.clearing_price(d, g, CARBON, shift=shift) / EUR
        cand = {
            "stored": r["market"], "recovered": back,
            "prof7": price(prof7), "profsame": price(profsame),
            "ts+prof7 /2": price(0.5 * s_ts + 0.5 * prof7),
            "ts shape, D-1 level": price(s_ts - np.nanmean(s_ts) + lvl1),
            "ts, gas D-1": price(s_ts, g1),
            "ts, demand rl_trees": price(s_ts, g2, r["rl_trees"] - OTHER),
            "ts shrink 0.8 to prof7": price(0.8 * s_ts + 0.2 * prof7),
            "damp 0.7 to D mean": price(s_ts, g2, dem.mean() + 0.7 * (dem - dem.mean())),
            "damp 0.5 to D mean": price(s_ts, g2, dem.mean() + 0.5 * (dem - dem.mean())),
            "damp 0.7 to 7d mean": price(s_ts, g2, m7 + 0.7 * (dem - m7)),
            "damp 0.5 to 7d mean": price(s_ts, g2, m7 + 0.5 * (dem - m7)),
            "ORACLE ts, demand rl_true": price(s_ts, g2, r["rl_true"] - OTHER),
            "ORACLE half rl_true": price(s_ts, g2, 0.5 * (r["rl_true"] + r["rl_mixed"]) - OTHER),
            "timestral": r["timestral"], "trees": r["trees"],
        }
        for k, v in cand.items():
            V.setdefault(k, {})[r["day"]] = (v, r["actual"], r["set"], dl.hour.to_numpy(), wdD)
    rows = {}
    for k, dd in V.items():
        for s in ("out_strict", "out_fresh_strict"):
            ds = [d for d in dd if dd[d][2] == s]
            e = np.array([np.nanmean(np.abs(dd[d][0][:96] - dd[d][1][:96])) for d in ds])
            b = np.array([np.nanmean(np.abs(V["stored"][d][0][:96] - V["stored"][d][1][:96])) for d in ds])
            ev = np.array([np.nanmean(np.abs(dd[d][0][:96] - dd[d][1][:96])[(dd[d][3][:96] >= 17) & (dd[d][3][:96] < 21)]) for d in ds])
            lo, hi = boot(e - b)
            rows.setdefault(k, {})[f"{s[4:]} MAE"] = round(e.mean(), 3)
            rows[k][f"{s[4:]} vs stored"] = f"{(e - b).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
            rows[k][f"{s[4:]} 17-21"] = round(ev.mean(), 2)
    print(pd.DataFrame(rows).T.to_markdown())
    os.makedirs(f"{SP}/P/market", exist_ok=True)


if __name__ == "__main__":
    main()
