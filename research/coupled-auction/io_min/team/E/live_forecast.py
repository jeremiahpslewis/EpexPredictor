"""Live DE day-ahead forecast for delivery day D, issued on the morning of D-1 before the 12:00 Berlin gate (team E).

Steps (all on this machine, ~10-15 min, see team/E/LIVE.md):
 1. live_fetch.py: the app's store classes bring SP/live/stores up to date (prices, gas, TSO figures, latest weather
    forecast from api.open-meteo.com with a logged fallback host) and snapshot them to OUT/stores; plus the grid
    previous-runs d1/d2 the matched variant needs, model-run metadata and what energy-charts serves for D.
 2. Checks that abort: DE prices end exactly at D-1 23:45 local; TSO figures complete to D-1 23:45; weather (DE zone
    and grid) covers D.
 3. Three forecasts in parallel, one thread each, on the same snapshot:
      (a) app       - the deployed app's own path (app_run.py: BaseModelCoordinator + RegionPriceManager);
      (b) champion  - the frozen champion: faithful_bt.py --rl-variant base, FAITHFUL_LIVE=1 (strict code path, live
                      inputs, train window [cutoff - 540 d, cutoff]);
      (c) matched   - faithful_bt.py --rl-variant matched, FAITHFUL_LIVE=1 (d2-trained RL boosters on slots whose
                      grid weather would be d2+ under the strict rule; live weather on those slots is younger than d2).
 4. CSVs in ct/kWh, 96 (92/100) quarter hours of local day D: forecast_<D>_{app,champion,matched}.csv with the members
    trees, timestral, market and the blend; forecast_<D>_all.csv with every column; team_forecast_<D>.csv = matched,
    or champion if matched failed, or app (S's rule of 2026-11-26). input_log_<D>.md/.json. SHA256SUMS. Never
    overwrites an issued file (use a new --out to rerun).

usage:  SP=... $SP/tsvenv/bin/python live_forecast.py --day 2026-11-28
        [--out DIR] [--skip-fetch]  (--skip-fetch: reuse OUT/stores from an earlier fetch, e.g. if Open-Meteo is down)
        [--dry-run]                 (D whose auction is past: output to SP/live/dry/<D>, labelled DRY RUN, not scored)
"""
import argparse
import atexit
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
os.environ["SP"] = SP
HERE = os.path.dirname(os.path.abspath(__file__))
FAITHFUL = os.path.abspath(f"{HERE}/../../faithful/faithful_bt.py")
PY = f"{SP}/tsvenv/bin/python"
TZ = ZoneInfo("Europe/Berlin")
MEMBERS = ("trees", "timestral", "market", "issued")
RUN_TIMEOUT = 15 * 60


def day_index(d: date) -> pd.DatetimeIndex:
    s = pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=TZ))
    e = pd.Timestamp(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TZ))
    return pd.date_range(s, e, freq="15min", inclusive="left").tz_convert("UTC")


def read_store(path):
    df = pd.read_json(path, compression="gzip")
    if pd.api.types.is_integer_dtype(df.index.dtype):
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_inputs(d, out) -> dict:
    D = day_index(d)
    cutoff = D[0] - pd.Timedelta(minutes=15)
    st = f"{out}/stores"
    prices = read_store(f"{st}/prices_v3_DE_LU.json.gz")
    gen = read_store(f"{st}/generation_v1_DE_LU.json.gz")
    w = read_store(f"{st}/weather_v2_DE_LU.json.gz")
    g = read_store(f"{st}/gridweather_v1_DE_LU.json.gz")
    gas = read_store(f"{st}/gasprices_the_v1_DE_LU.json.gz")
    Dm1 = day_index(d - timedelta(days=1))
    chk = {
        "cutoff_utc": str(cutoff),
        "DE_price_last": str(prices.index.max()),
        "DE_price_slots_D-1": int(prices.index.isin(Dm1).sum()),
        "DE_price_slots_after_cutoff": int((prices.index > cutoff).sum()),
        "TSO_last_complete": str(gen.index.max()),
        "TSO_slots_D-1": int(gen.index.isin(Dm1).sum()),
        "gas_last_valid": str(gas.index.max()),
        # THE gas day D (indexed at D 00:00 UTC) already served at fetch time? (the champion cuts it, the app keeps it)
        "gas_day_D_present": bool(pd.Timestamp(d.isoformat(), tz="UTC") in gas.index),
        "weather_DE_slots_on_D": int(w.index.isin(D).sum()),
        "weather_DE_last": str(w.index.max()),
        "grid_hours_on_D": int(g.index.isin(D[D.minute == 0]).sum()),
        "grid_last": str(g.index.max()),
    }
    errors = []
    if chk["DE_price_slots_D-1"] != len(Dm1) or prices.loc[:cutoff].index.max() != cutoff:
        errors.append("DE prices for D-1 incomplete")
    if chk["TSO_slots_D-1"] != len(Dm1):
        errors.append("TSO figures for D-1 incomplete")
    if chk["weather_DE_slots_on_D"] != len(D) or chk["grid_hours_on_D"] != len(D) // 4:
        errors.append("weather does not cover D")
    chk["errors"] = errors
    return chk


ZONE_FILES = {"DE_LU": "DE", "AT": "AT", "BE": "BE", "FR": "FR", "NL": "NL", "SE_4": "SE4", "DK_1": "DK1", "DK_2": "DK2"}


def emergency_weather(d, out, only=None) -> dict:
    """Fill each weather store of the snapshot from its last row to the end of D+2 with the newest local previous-runs
    file (this run's grid_prev_live first for the grid, else SP/fresh_data), preferring d1, then d2, then d3, hourly
    interpolated to 15 min for the zones (radiation shifted as in faithful_bt). Only the snapshot is changed."""
    st, filled = f"{out}/stores", {}
    end = day_index(d + timedelta(days=2))[-1]
    for zone, f in list(ZONE_FILES.items()) + [("GRID", None)]:
        key = f or "GRID"            # the fetch log's store names
        if only is not None and key not in only:
            continue
        fn = f"{st}/gridweather_v1_DE_LU.json.gz" if zone == "GRID" else f"{st}/weather_v2_{zone}.json.gz"
        w = read_store(fn)
        if zone == "GRID":
            srcs = [x for x in (f"{out}/grid_prev_live.parquet", f"{SP}/fresh_data/grid_prev.parquet") if os.path.exists(x)]
            names = {"w120": "wind", "sw": "radiation", "t2": "temp"}
            ren = lambda c: f"{names[c.split('_')[0]]}_{c.split('_')[1]}"   # noqa: E731
        else:
            srcs = [f"{SP}/fresh_data/prev_{f}.parquet"]
            ren = lambda c: c.rsplit("_", 1)[0]   # noqa: E731
        h = None
        for src in srcs:
            p = pd.read_parquet(src)
            for k in (1, 2, 3):
                part = p[[c for c in p.columns if c.endswith(f"_d{k}") and (zone != "GRID" or c.split("_")[0] in ("w120", "sw", "t2"))]].rename(columns=ren)
                part = part.reindex(columns=list(w.columns)).astype(float)
                h = part if h is None else h.combine_first(part)
        step = "1h" if zone == "GRID" else "15min"
        idx = pd.date_range(w.index.max() + pd.Timedelta(step), end, freq=step)
        if len(idx) == 0:
            continue
        hh = h.loc[w.index.max() - pd.Timedelta(hours=3):end + pd.Timedelta(hours=3)]
        if zone != "GRID":
            cols = {}
            for c in hh.columns:
                sc = hh[c].dropna()
                if c.startswith("irradiance_"):
                    sc.index = sc.index - pd.Timedelta(minutes=22.5)
                cols[c] = sc.reindex(sc.index.union(idx)).interpolate("time", limit_area="inside").reindex(idx)
            add = pd.DataFrame(cols, index=idx).ffill(limit=4)
        else:
            add = hh.reindex(idx)
        add = add.dropna()
        if len(add):
            pd.concat([w, add]).to_json(fn, compression="gzip")
            filled[key] = f"{len(add)} rows to {add.index.max()} from {', '.join(os.path.basename(x) for x in srcs)} (d1>d2>d3)"
    return filled


def launch(name, cmd, env, logf):
    return subprocess.Popen(cmd, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--out")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help="torch/OMP threads per forecast process (3 processes)")
    ap.add_argument("--deadline", help="UTC time (YYYY-MM-DDTHH:MM) at which unfinished forecasts are killed; "
                                       "default gate - 55 min (10:05 UTC for an 11:00 UTC gate), dry run: fetch end + 15 min")
    a = ap.parse_args()
    d = date.fromisoformat(a.day)
    out = a.out or (f"{SP}/live/dry/{d}" if a.dry_run else f"{SP}/live/{d}")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/SHA256SUMS"):
        sys.exit(f"{out} already holds an issued forecast (SHA256SUMS); use a new --out")
    try:   # one run per output directory at a time (an armed waiter and a manual run cannot collide)
        os.close(os.open(f"{out}/.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        atexit.register(lambda: os.path.exists(f"{out}/.lock") and os.remove(f"{out}/.lock"))   # released on any exit
    except FileExistsError:
        sys.exit(f"{out}/.lock exists: another run is (or was) working on this directory; check it, or delete the lock "
                 "only if that run is dead")
    t0 = time.time()
    started = datetime.now(timezone.utc)
    gate = pd.Timestamp(datetime.combine(d - timedelta(days=1), datetime.min.time(), tzinfo=TZ).replace(hour=12)).tz_convert("UTC")
    print(f"[{started:%H:%M:%S}Z] live forecast for D={d}, gate {gate} UTC, out {out}", flush=True)
    if not a.dry_run and started > gate.to_pydatetime():
        print("WARNING: started after the gate - this is not a valid live forecast", flush=True)

    # 1) fetch
    if not a.skip_fetch:
        r = subprocess.run([PY, f"{HERE}/live_fetch.py", "--day", str(d), "--outdir", out], cwd=HERE,
                           stdout=open(f"{out}/fetch.log", "w"), stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT)
        if r.returncode != 0:
            sys.exit(f"fetch failed (exit {r.returncode}), see {out}/fetch.log; rerun, or --skip-fetch on an earlier snapshot")
    t_fetch = time.time()
    flog = json.load(open(f"{out}/fetch_log.json"))
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] fetch done in {t_fetch - t0:.0f}s; weather sources: "
          + ", ".join(f"{k}={v.get('source')}" for k, v in flog["weather"].items()), flush=True)

    # 2) checks
    chk = check_inputs(d, out)
    D_last = day_index(d)[-1]
    short = {k for k, v in flog["weather"].items() if "last" in v and pd.Timestamp(v["last"]) < D_last}
    if chk["errors"] == ["weather does not cover D"]:
        short |= {"DE", "GRID"}
    if short and not (set(chk["errors"]) - {"weather does not cover D"}):
        # every Open-Meteo host failed for these stores and no earlier fetch covers D: issue on the newest locally held
        # previous-runs forecasts rather than withhold the forecast (S, 2026-11-26), labelled in every header
        print(f"weather short of D for {sorted(short)}: emergency fill", flush=True)
        filled = emergency_weather(d, out, short)
        flog.setdefault("emergency_weather", filled)
        for k, v in filled.items():
            flog["weather"].setdefault(k, {})["source"] = f"EMERGENCY fill: {v}"
        json.dump(flog, open(f"{out}/fetch_log.json", "w"), indent=1, default=str)
        chk = check_inputs(d, out)
        chk["emergency_weather"] = filled
    print(json.dumps(chk, indent=1), flush=True)
    if chk["errors"]:
        sys.exit("ABORT: " + "; ".join(chk["errors"]))

    # 3) forecasts in parallel
    env = dict(os.environ, SP=SP, FAITHFUL_LIVE="1", FAITHFUL_LIVE_DIR=out, OMP_NUM_THREADS=str(a.threads))
    th = ["--torch-threads", str(a.threads)]
    jobs = {"app": launch("app", [PY, f"{HERE}/app_run.py", "--day", str(d), "--stores", f"{out}/stores", "--out", f"{out}/runs"] + th,
                          dict(os.environ, SP=SP, OMP_NUM_THREADS=str(a.threads)), f"{out}/run_app.log"),
            "champion": launch("champion", [PY, FAITHFUL, "--days", str(d), "--rl-variant", "base", "--out", f"{out}/runs/champion"] + th,
                               env, f"{out}/run_champion.log")}
    # matched needs genuine 2-day-old grid forecasts up to the cutoff: valid time - 48 h <= fetch time of the file
    cutoff = day_index(d)[0] - pd.Timedelta(minutes=15)
    need = cutoff + pd.Timedelta(hours=1) - pd.Timedelta(hours=48)
    if "error" not in flog.get("grid_prev_live", {"error": 1}):
        grid_src, grid_fetch = "grid_prev_live (this run)", pd.Timestamp(flog["grid_prev_live"]["fetched_utc"])
    else:
        fn = f"{SP}/fresh_data/grid_prev.parquet"
        grid_src, grid_fetch = f"{fn} (earlier fetch)", pd.Timestamp(os.path.getmtime(fn), unit="s", tz="UTC")
    grid_ok = need <= grid_fetch
    chk["matched_d2_source"] = {"source": grid_src, "fetched_utc": str(grid_fetch), "needs_issue_by": str(need), "ok": bool(grid_ok)}
    print(f"matched d2 history: {chk['matched_d2_source']}", flush=True)
    if grid_ok:
        jobs["matched"] = launch("matched", [PY, FAITHFUL, "--days", str(d), "--rl-variant", "matched", "--out", f"{out}/runs/matched"] + th,
                                 env, f"{out}/run_matched.log")
    secs = {}
    # hard stops: matched (shadow) at gate - 55 min (10:05 UTC for an 11:00 gate); app and champion may run to
    # gate - 10 min, so a late start still issues; --deadline overrides all; dry run: 15 min
    if a.deadline:
        stops = dict.fromkeys(jobs, pd.Timestamp(a.deadline, tz="UTC").timestamp())
    elif a.dry_run:
        stops = dict.fromkeys(jobs, time.time() + RUN_TIMEOUT)
    else:
        stops = {n: (gate - pd.Timedelta(minutes=55 if n == "matched" else 10)).timestamp() for n in jobs}
    print("hard stops: " + ", ".join(f"{n} {pd.Timestamp(t, unit='s', tz='UTC'):%H:%M:%S}Z" for n, t in stops.items()), flush=True)
    for name, p in sorted(jobs.items(), key=lambda kv: stops[kv[0]]):
        try:
            p.wait(timeout=max(1, stops[name] - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
        secs[name] = round(time.time() - t_fetch, 1)
        print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {name} exit {p.returncode} after {secs[name]}s", flush=True)

    # 4) assemble
    D = day_index(d)
    files = {"app": f"{out}/runs/app_{d}.npz", "champion": f"{out}/runs/champion/{d}.npz", "matched": f"{out}/runs/matched/{d}.npz"}
    base = pd.DataFrame({"time_utc": D.strftime("%Y-%m-%dT%H:%MZ"), "time_local": D.tz_convert(TZ).strftime("%Y-%m-%dT%H:%M%z")})
    allcols, ok, metas, frames = base.copy(), {}, {}, {}
    for name, fn in files.items():
        if not os.path.exists(fn):
            ok[name] = False
            continue
        z = np.load(fn, allow_pickle=False)
        assert np.array_equal(z["idx"], D.asi8), f"{name}: index mismatch"
        df = base.copy()
        for m in MEMBERS:
            df[m if m != "issued" else "blend"] = np.round(z[m], 4)
        ok[name] = bool(np.isfinite(df[["trees", "timestral", "market", "blend"]].to_numpy()).all())
        metas[name] = json.loads(str(z["meta"]))
        frames[name] = df
        for c in ("trees", "timestral", "market", "blend"):
            allcols[f"{name}_{c}"] = df[c]
        # residual load (GW) as the price trees read it, and per model set (to score against tomorrow's TSO figures)
        if "rl_trees" in z.files:
            allcols[f"{name}_rl_trees_GW"] = np.round(z["rl_trees"], 4)
            allcols[f"{name}_rl_mixed_GW"] = np.round(z["rl_mixed"], 4)
            for k in ("base", "matched"):
                for t in ("tso_wind", "tso_solar", "tso_load"):
                    if f"rlc_{k}_{t}" in z.files and np.isfinite(z[f"rlc_{k}_{t}"]).any():
                        allcols[f"{name}_rl_{k}_{t}_GW"] = np.round(z[f"rlc_{k}_{t}"], 4)
            if name == "matched":
                allcols["matched_uses_d2_boosters"] = z["rlc_use_matched"].astype(int)
    # S's rule (2026-11-26 16:00): team forecast = frozen champion config on live inputs, else the app; matched = shadow
    team = next((n for n in ("champion", "app") if ok.get(n)), None)
    if team is None:
        sys.exit("ABORT: no forecast succeeded; see run_*.log")
    wsrc = "; ".join(f"{k} {v.get('source')} fetched {v.get('weather_fetched_utc', v.get('fetched_utc'))}" for k, v in flog["weather"].items())
    label = "DRY RUN (auction already held; not scored)" if a.dry_run else "LIVE"
    roles = {"champion": "TEAM FORECAST candidate: frozen champion config (faithful harness, rl base) on live inputs",
             "app": "deployed app path (app_run.py), the user's baseline",
             "matched": "SHADOW only: rl matched on live weather (no backtest of this configuration); not the team forecast"}

    def header(role):
        return (f"# DE day-ahead forecast for {d} ({label}), ct/kWh, 96 local quarter hours; {role}\n"
                f"# issued {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC, gate {gate} UTC\n"
                f"# weather per store (source: primary = api.open-meteo.com, fallback = previous-runs-api same request, stored = earlier fetch): {wsrc}\n")

    for name, df in frames.items():
        with open(f"{out}/forecast_{d}_{name}.csv", "w") as f:
            f.write(header(roles[name]))
            df.to_csv(f, index=False)
    with open(f"{out}/forecast_{d}_all.csv", "w") as f:
        f.write(header("all members of app / champion / matched(shadow), plus residual load in GW"))
        allcols.to_csv(f, index=False)
    tf = frames[team].copy()
    tf.insert(2, "source", team)
    with open(f"{out}/team_forecast_{d}.csv", "w") as f:
        f.write(header(f"TEAM FORECAST = {team}" + ("" if team == "champion" else " (champion failed or missed the deadline)")))
        tf.to_csv(f, index=False)

    # input log
    log = {"label": label,
           "day": str(d), "gate_utc": str(gate), "started_utc": started.isoformat(timespec="seconds"),
           "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "runtime_s": {"fetch": round(t_fetch - t0, 1), **secs, "total": round(time.time() - t0, 1)},
           "runs_ok": ok, "team_forecast_source": team, "checks": chk,
           "fetch": {k: v for k, v in flog.items() if k != "requests"}, "n_requests": len(flog.get("requests", [])),
           "hosts": sorted({r["host"] for r in flog.get("requests", [])}),
           "run_meta": {n: {k: m.get(k) for k in ("secs", "weather", "train_last_DE", "train_rows_stage1_DE", "rl_extra",
                                                  "rows_beyond_cutoff_dropped", "last_known_price", "blend_is_mean_of_thirds")} for n, m in metas.items()}}
    json.dump(log, open(f"{out}/input_log_{d}.json", "w"), indent=1, default=str)
    pub = flog.get("published_on_D", {})
    md = [f"# Input log, DE day-ahead forecast for {d} ({log['label']})", "",
          f"- gate {gate} UTC; run {log['started_utc']} -> {log['finished_utc']} UTC; runtime {log['runtime_s']}",
          f"- forecasts ok: {ok}; team forecast = {team}",
          f"- DE prices: last {chk['DE_price_last']} UTC (fetched {flog['prices'].get('DE', {}).get('fetched_utc')})",
          f"- TSO wind/solar/load (complete rows): last {chk['TSO_last_complete']} UTC (fetched {flog.get('generation', {}).get('fetched_utc')})",
          f"- energy-charts serving for D at fetch: " + ", ".join(f"{k} {v.get('slots_on_D')} slots ({v.get('fetched_utc')})" for k, v in pub.items()),
          f"- gas (THE, DE store): gas day D present at fetch: {chk['gas_day_D_present']} (champion cuts it, app keeps it); last valid {chk['gas_last_valid']} UTC (fetched {flog['gas'].get('DE', {}).get('fetched_utc')})",
          "- weather (store: source, hosts, last valid, fetched):"]
    md += [f"  - {k}: this run {v.get('source')} via {v.get('hosts')}; forecast in use fetched {v.get('weather_fetched_utc', v.get('fetched_utc'))} "
           f"({v.get('weather_source', v.get('source'))}); last valid {v.get('last')}" for k, v in flog["weather"].items()]
    md += ["- Open-Meteo model runs at fetch start (init / available, UTC):"]
    md += [f"  - {m}: {v.get('last_run_initialisation_time')} / {v.get('last_run_availability_time')}" for m, v in flog["openmeteo_model_runs_at_start"].items()]
    md += [f"- grid previous-runs (matched): {flog.get('grid_prev_live')}"]
    open(f"{out}/input_log_{d}.md", "w").write("\n".join(md) + "\n")
    names = [f for f in sorted(os.listdir(out)) if f.endswith(".csv") or f.startswith("input_log_")]
    sums = {f: sha256(f"{out}/{f}") for f in names}
    open(f"{out}/SHA256SUMS", "w").write("".join(f"{h}  {f}\n" for f, h in sums.items()))
    print(f"\n{log['label']} D={d}: team forecast = {team} ({out}/team_forecast_{d}.csv)")
    for f, h in sums.items():
        print(f"SHA-256 {h}  {f}")
    print(f"total runtime {time.time() - t0:.0f}s")
    # text note for the repo (no data): hashes, absolute paths, input log
    note_dir = f"{HERE}/live"
    os.makedirs(note_dir, exist_ok=True)
    note = f"{note_dir}/{'dry_' if a.dry_run else ''}{d}.md"
    with open(note, "w") as f:
        f.write(f"# {label}: DE day-ahead forecast for {d}\n\nTeam forecast: **{team}** "
                f"(`{out}/team_forecast_{d}.csv`). Runs ok: {ok}. Runtime (s): {log['runtime_s']}.\n\n"
                "Files (in the scratchpad, not in the repo) and SHA-256:\n\n")
        f.write("".join(f"- `{h}`  `{out}/{fn}`\n" for fn, h in sums.items()))
        f.write("\n" + open(f"{out}/input_log_{d}.md").read().split("\n", 1)[1])
    print(f"note: {note}")


if __name__ == "__main__":
    main()
