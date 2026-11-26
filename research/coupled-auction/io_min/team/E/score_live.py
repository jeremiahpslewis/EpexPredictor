"""Score one live day (team E, neutral scorer): app / champion / matched shadow from SP/live/<D>/forecast_<D>_all.csv.

prices: after D's auction (~11:45 UTC on D-1). The DE price for D is fetched with the app's PriceStore (energy-charts)
        into a scratch copy of the persistent price store (SP/live/score/), never into SP/live/stores.
        MAE over the local day's quarter hours, ct/kWh, per run and member; naive (same local QH, D-7, from the same
        store) and rMAE; differences vs champion. One day: no interval (the scoreboard pools days, then bootstraps).
rl:     after D has passed (TSO wind/solar/load for D complete, D+1): residual-load MAE / bias (GW) of every RL column
        in the CSV (trees RL per run; base vs matched boosters on live weather) against the published figures.
Output: markdown to stdout and SP/live/<D>/score_<part>.md; one-line CSV row SP/live/scores.csv (appended).

usage: tsvenv/bin/python score_live.py --day 2026-11-28 --part prices|rl
"""
import argparse
import asyncio
import os
import shutil
import sys
from datetime import date, datetime, timedelta, timezone

import aiohttp
import numpy as np
import pandas as pd

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_forecast import day_index  # noqa: E402
from naive import naive_for  # noqa: E402

_S = aiohttp.ClientSession


class _P(_S):
    def __init__(self, *a, **k):
        k.setdefault("trust_env", True)
        super().__init__(*a, **k)


aiohttp.ClientSession = _P
SCRATCH = f"{SP}/live/score"


async def fetch(d, part):
    os.makedirs(SCRATCH, exist_ok=True)
    os.environ["EPEXPREDICTOR_DATADIR"] = SCRATCH
    sys.path.insert(0, "/home/user/EpexPredictor")
    from predictor.model.priceregion import PriceRegionName
    de = PriceRegionName.DE.to_region()
    D = day_index(d)
    if part == "prices":
        from predictor.model.pricestore import PriceStore
        fn = "prices_v3_DE_LU.json.gz"
        if not os.path.exists(f"{SCRATCH}/{fn}"):
            shutil.copy2(f"{SP}/live/stores/{fn}", f"{SCRATCH}/{fn}")
        p = await PriceStore(de, SCRATCH).load()
        await p.fetch_missing_data(D[0].to_pydatetime(), D[-1].to_pydatetime())
        return p.data["price"]
    from predictor.model.residualload import GenerationForecastStore
    fn = "generation_v1_DE_LU.json.gz"
    if not os.path.exists(f"{SCRATCH}/{fn}"):
        shutil.copy2(f"{SP}/live/stores/{fn}", f"{SCRATCH}/{fn}")
    g = await GenerationForecastStore(de, SCRATCH).load()
    await g.fetch_missing_data(D[0].to_pydatetime(), D[-1].to_pydatetime())
    return g.data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--part", choices=("prices", "rl"), required=True)
    ap.add_argument("--dir", help="default SP/live/<D>")
    a = ap.parse_args()
    d = date.fromisoformat(a.day)
    out = a.dir or f"{SP}/live/{d}"
    D = day_index(d)
    f = pd.read_csv(f"{out}/forecast_{d}_all.csv", comment="#")
    assert len(f) == len(D) and (pd.to_datetime(f["time_utc"], utc=True).to_numpy() == D.to_numpy()).all()
    got = asyncio.run(fetch(d, a.part))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"# Live score {d} ({a.part}), scored {now} UTC", ""]
    row = {"day": str(d), "part": a.part, "scored_utc": now}
    if a.part == "prices":
        act = got.reindex(D).to_numpy(float)
        if not np.isfinite(act).all():
            sys.exit(f"D's prices not complete yet ({int(np.isfinite(act).sum())}/{len(D)})")
        nv = naive_for(D, got.dropna())
        nmae = float(np.mean(np.abs(nv - act)))
        lines += [f"naive (same local QH, D-7): {nmae:.3f} ct/kWh", "",
                  "| run | trees | Timestral | market | blend | blend rMAE | blend - champion |", "|---|---|---|---|---|---|---|"]
        champ = None
        if "champion_blend" in f:
            champ = float(np.mean(np.abs(f["champion_blend"].to_numpy(float) - act)))
        for run in ("app", "champion", "matched"):
            if f"{run}_blend" not in f:
                lines.append(f"| {run} | missing | | | | | |")
                continue
            m = {c: float(np.mean(np.abs(f[f"{run}_{c}"].to_numpy(float) - act))) for c in ("trees", "timestral", "market", "blend")}
            row |= {f"{run}_{c}": round(v, 4) for c, v in m.items()}
            dif = f"{m['blend'] - champ:+.3f}" if champ is not None else ""
            lines.append(f"| {run}{' (team)' if run == 'champion' else ' (shadow)' if run == 'matched' else ''} | {m['trees']:.3f} | "
                         f"{m['timestral']:.3f} | {m['market']:.3f} | {m['blend']:.3f} | {m['blend'] / nmae:.3f} | {dif} |")
        row["naive"] = round(nmae, 4)
    else:
        g = got.reindex(D)
        if g.isna().any().any():
            sys.exit(f"D's TSO figures not complete yet ({int(g.notna().all(axis=1).sum())}/{len(D)})")
        true = {"rl": (g.tso_load - g.tso_wind - g.tso_solar).to_numpy(float), "tso_wind": g.tso_wind.to_numpy(float),
                "tso_solar": g.tso_solar.to_numpy(float), "tso_load": g.tso_load.to_numpy(float)}
        lines += ["| series | MAE GW | bias GW (forecast - TSO) |", "|---|---|---|"]
        for c in f.columns:
            if not c.endswith("_GW"):
                continue
            tgt = "rl" if ("rl_trees" in c or "rl_mixed" in c) else next((t for t in ("tso_wind", "tso_solar", "tso_load") if t in c), None)
            if tgt is None:
                continue
            e = f[c].to_numpy(float) - true[tgt]
            lines.append(f"| {c} | {np.nanmean(np.abs(e)):.2f} | {np.nanmean(e):+.2f} |")
            row[c] = round(float(np.nanmean(np.abs(e))), 3)
    txt = "\n".join(lines) + "\n"
    print(txt)
    open(f"{out}/score_{a.part}.md", "w").write(txt)
    dry = "DRY RUN" in open(f"{out}/forecast_{d}_all.csv").readline()
    sc = f"{SP}/live/scores{'_dry' if dry else ''}.csv"   # a dry run never reaches the live table
    pd.DataFrame([row]).to_csv(sc, mode="a", header=not os.path.exists(sc), index=False)


if __name__ == "__main__":
    main()
