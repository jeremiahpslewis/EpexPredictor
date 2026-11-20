"""
Historical window as a reconciliation and prior-predictive check, not a fit (notes 21.157). Zone A is fed the public day-
ahead inputs of DE-LU (ENTSO-E load forecast, wind and solar forecasts; the last THE gas price in the app's store dated before
noon the day before - the app itself lags gas two days, which this does not reproduce; carbon 75 as in
the app's market route). Zone B stays the synthetic neighbour: it is not a real zone, so flows and B's price have no
observed counterpart. For each delivery day, parameters and day shocks are drawn from the prior and the day is cleared;
the observed DE-LU price, day-ahead net position and scheduled generation are compared with the prior-predictive spread.
Usage: hist_window.py DATA_DIR OUT_DIR (DATA_DIR holds entsoe/da/*.parquet and appdata/*.json.gz, the app's stores; never committed).
"""
import gzip, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import vslice as V

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2]); DRAWS = int(os.environ.get("DRAWS", 40)); TZ = "Europe/Berlin"
# ablations (notes 21.161), comma-separated in ABLATE: "flex0" no price-responsive demand; "fleet" zone A's lignite and
# coal cut towards what ran at most in the window (12.6 and 4.5 GW): 13 and 5 GW nominal (coal less the slice's 1 GW planned
# outage = 4 GW) instead of 18 and 12; the output file names carry the tag
ABLATE = [a for a in os.environ.get("ABLATE", "").split(",") if a]; TAG = ("_" + "_".join(ABLATE)) if ABLATE else ""
if "fleet" in ABLATE:
    V.TECH["A"]["lignite"] = (13000,) + V.TECH["A"]["lignite"][1:]; V.TECH["A"]["coal"] = (5000,) + V.TECH["A"]["coal"][1:]
DAYS = pd.date_range(os.environ.get("FIRST", "2026-10-15"), periods=int(os.environ.get("NDAYS", 14)), freq="D", tz=TZ)

def series(name, col):
    d = pd.concat([pd.read_parquet(f) for f in sorted((SP / "entsoe/da").glob(f"{name}_*.parquet"))]); d = d[~d.index.duplicated()]
    return d[col] if isinstance(col, str) else d[col].sum(axis=1)
def store(name, key, scale=1.0):
    with gzip.open(SP / "appdata" / f"{name}.json.gz", "rt") as f: d = json.load(f)[key]
    return pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v * scale for k, v in d.items()}).sort_index().tz_convert(TZ)

load, ws = series("load", "Forecasted Load"), pd.concat([pd.read_parquet(f) for f in sorted((SP / "entsoe/da").glob("ws_*.parquet"))])
ws = ws[~ws.index.duplicated()]; netpos, gen = series("netpos", "netpos"), series("gen", "gen")
price, gas = store("prices_v3_DE_LU", "price", 10.0), store("gasprices_the_v1_DE_LU", "gasprice")      # stored in ct/kWh; THE gas in EUR/MWh

def day_inputs(d):
    idx = pd.date_range(d, (d + pd.DateOffset(days=1)).normalize(), freq="15min", inclusive="left")   # a calendar day: 92 or 100 quarters at a clock change
    if len(idx) != V.T: return None, idx             # clock-change days have 92 or 100 quarters; the slice has 96
    info = V.Info(); info.co2 = 75.0
    g = gas[:d - pd.Timedelta(hours=12)]; info.gas = float(g.iloc[-1])      # last stored value dated before noon the day before
    info.load = {"A": load.reindex(idx).to_numpy(float), "B": V.Info().load["B"]}
    info.wind_cf = {"A": np.clip((ws["Wind Onshore"] + ws["Wind Offshore"]).reindex(idx).to_numpy(float) / V.TECH["A"]["wind"][0], 0, 1), "B": V.Info().wind_cf["B"]}
    info.solar_cf = {"A": np.clip(ws["Solar"].reindex(idx).to_numpy(float) / V.TECH["A"]["solar"][0], 0, 1)}
    return info, idx

def run(args):
    k, d = args; info, idx = day_inputs(d)
    rng = np.random.default_rng(3_000_000 + k); th = V.draw_theta(rng); day = V.draw_day(rng, info)
    if "flex0" in ABLATE: th["flex_share"] = 0.0
    try:
        m, acc, r = V.simulate(th, day, info)
    except Exception as e:
        return k, d, None
    sold = np.zeros(V.T)
    for j, o in enumerate(m.orders):
        if o.zone == "A" and o.side == "sell": sold[o.t] += r.x[j] * o.q
    for b_, blk in enumerate(m.blocks):
        if blk.zone == "A": sold += r.a[b_] * np.array(blk.q)
    gen_a = sold + acc["A"]["netted"]                    # scheduled generation: auction awards plus what portfolios serve themselves
    return k, d, dict(price=r.prices[0], flow=r.f[0], gen=gen_a)

if __name__ == "__main__":
    ok = [d for d in DAYS if day_inputs(d)[0] is not None and load.reindex(day_inputs(d)[1]).notna().all()]
    jobs = [(i * 1000 + k, d) for i, d in enumerate(ok) for k in range(DRAWS)]
    with Pool(int(os.environ.get("WORKERS", 4))) as pool: res = pool.map(run, jobs, chunksize=4)
    rows, fails = [], 0
    for i, d in enumerate(ok):
        idx = day_inputs(d)[1]; sims = [r for (_, dd, r) in res if dd == d and r is not None]; fails += DRAWS - len(sims)
        P = np.array([s["price"] for s in sims]); F = np.array([s["flow"] for s in sims]); G = np.array([s["gen"] for s in sims])
        obs_p = price.reindex(idx).to_numpy(float); obs_np = netpos.reindex(idx).to_numpy(float); obs_g = gen.reindex(idx).to_numpy(float)
        for t in range(V.T):
            rows.append(dict(time=idx[t], obs_price=obs_p[t], p05=np.quantile(P[:, t], 0.05), p50=np.median(P[:, t]), p95=np.quantile(P[:, t], 0.95),
                             obs_netpos=obs_np[t], flow_p50=np.median(-F[:, t]), obs_gen=obs_g[t], gen_p50=np.median(G[:, t]),
                             gen_p05=np.quantile(G[:, t], 0.05), gen_p95=np.quantile(G[:, t], 0.95), load=day_inputs(d)[0].load["A"][t],
                             gas=day_inputs(d)[0].gas))
    R = pd.DataFrame(rows).set_index("time"); R.to_parquet(OUT / f"hist_window{TAG}.parquet")
    inside = (R.obs_price >= R.p05) & (R.obs_price <= R.p95); ginside = (R.obs_gen >= R.gen_p05) & (R.obs_gen <= R.gen_p95)
    h = R.index.hour
    summ = {"days": len(ok), "skipped_days": [str(d.date()) for d in DAYS if d not in ok], "failed_draws": fails, "draws_per_day": DRAWS,
            "price_inside_90": round(float(inside.mean()), 3), "price_mae_median": round(float((R.obs_price - R.p50).abs().mean()), 1),
            "price_bias_median": round(float((R.obs_price - R.p50).mean()), 1),
            "price_inside_by_4h": {f"{a:02d}-{a + 4:02d}": round(float(inside[(h >= a) & (h < a + 4)].mean()), 2) for a in range(0, 24, 4)},
            "price_bias_by_4h": {f"{a:02d}-{a + 4:02d}": round(float((R.obs_price - R.p50)[(h >= a) & (h < a + 4)].mean()), 1) for a in range(0, 24, 4)},
            "obs_netpos_abs_over_1000": round(float((R.obs_netpos.abs() > 1000).mean()), 3), "obs_netpos_mean": round(float(R.obs_netpos.mean())),
            "obs_netpos_range": [round(float(R.obs_netpos.min())), round(float(R.obs_netpos.max()))],
            "gen_inside_90": round(float(ginside.mean()), 3), "gen_bias_median_MW": round(float((R.obs_gen - R.gen_p50).mean())),
            "obs_gen_mean": round(float(R.obs_gen.mean())), "load_mean": round(float(R.load.mean())), "gas_range": [round(float(R.gas.min()), 1), round(float(R.gas.max()), 1)]}
    summ["ablate"] = ABLATE; json.dump(summ, open(OUT / f"hist_window{TAG}.json", "w"), indent=1); print(json.dumps(summ, indent=1))
