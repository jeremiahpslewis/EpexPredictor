"""Hindsight bound of the residual-load lever: the faithful harness (faithful_bt.py, imported unchanged) with the TRUE
residual load (published TSO day-ahead load - wind - solar, as stored) in place of the forecast, after the cutoff.

ORACLE_TARGETS (default "tso_load,tso_wind,tso_solar"): which components are replaced by their published figures in
the tree RL (what the DE stage-1 and stage-2 trees read, and the tree part of Timestral's mix).
ORACLE_TS=1 (default): Timestral's own RL forecast is replaced as well, so the mixed RL that Timestral and the market
route read is the truth too (full bound). ORACLE_TS=0: only the tree RL moves (the trees' bound; Timestral/market then
see w*Timestral_own + (1-w)*truth).
Slots without a published figure keep the model's value. This is an ORACLE (uses figures not known at the gate); it
bounds what any RL improvement can buy, it is not a forecast.

usage: SP=... tsvenv/bin/python oracle_rl.py --days 2026-09-10,... --out DIR [--torch-threads 1]
"""
import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor/research/coupled-auction/io_min/faithful")
import faithful_bt as fb  # noqa: E402  (sets up the stores and patches, unchanged)
from predictor.model.timestralload import TimestralLoad  # noqa: E402

TARGETS = [t for t in os.environ.get("ORACLE_TARGETS", "tso_load,tso_wind,tso_solar").split(",") if t]
ORACLE_TS = os.environ.get("ORACLE_TS", "1") == "1"
TRUE: dict = {}   # "gen": the full (uncut) generation figures

_orig_components = fb.rl_components


async def oracle_components(rl, start, end):
    out = await _orig_components(rl, start, end)
    if len(out) == 0:
        return out
    g = TRUE["gen"].reindex(out.index)
    for t in TARGETS:
        tr = g[t].to_numpy(float)
        ok = np.isfinite(tr)
        out[f"sel_{t}"] = np.where(ok, tr, out[f"sel_{t}"].to_numpy())
        out[f"oracle_{t}"] = ok.astype(float)
    return out


fb.rl_components = oracle_components     # _rl_get_data and run_day look it up in faithful_bt's globals

_orig_ts_rl = TimestralLoad.residual_load


async def oracle_ts_rl(self, residual_load):
    s = await _orig_ts_rl(self, residual_load)
    if s is None or not ORACLE_TS:
        return s
    g = TRUE["gen"]
    true = (g.tso_load - g.tso_wind - g.tso_solar).reindex(s.index)
    return true.where(true.notna(), s).rename(s.name)


TimestralLoad.residual_load = oracle_ts_rl


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--torch-threads", type=int, default=1)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.torch_threads)
    a.vintage, a.latency, a.rl_variant = "strict", 6.0, "oracle"
    fb.RLV.variant = "oracle"     # non-base: _rl_get_data recomputes after the cutoff; no extra boosters are trained
    days = [date.fromisoformat(x) for x in a.days.split(",")]
    h = fb.Harness(a)
    await h.setup()
    TRUE["gen"] = h.orig[id(h.rl.generation)]
    fb.log.info(f"oracle targets {TARGETS}, timestral own RL oracle {ORACLE_TS}; {len(days)} days -> {a.out}")
    for d in days:
        t = time.time()
        try:
            await h.run_day(d)
        except Exception as e:
            fb.log.exception(f"{d}: FAILED ({type(e).__name__}: {e})")
        fb.log.info(f"{d}: wall {time.time() - t:.0f}s")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
