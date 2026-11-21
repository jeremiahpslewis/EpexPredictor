"""Public ENTSO-E day-ahead data for an aggregate "neighbour" zone of DE_LU (notes/neighbour_data.md).

For each zone coupled to DE_LU (FR NL BE AT CZ PL DK_1 DK_2 SE_4 NO_2, plus CH kept apart because it is
outside SDAC) this fetches, 2025-08-01 .. 2026-11-09 (Europe/Berlin, end exclusive 2026-11-10):
  load   day-ahead load forecast                       query_load_forecast
  ws     day-ahead wind on/offshore + solar forecast   query_wind_and_solar_forecast
  price  day-ahead price                               query_day_ahead_prices
  sched_out / sched_in   DA scheduled exchanges DE_LU->Z / Z->DE_LU   query_scheduled_exchanges(dayahead=True)
  ntc_out / ntc_in       DA NTC DE_LU->Z / Z->DE_LU                   query_net_transfer_capacity_dayahead
  cap    installed capacity per production type, per calendar year (2025, 2026)

Client set-up follows research/market-model/exp/entsoe_da.py: quarterly chunks, resumable, retry with backoff.
Each (zone, series, chunk) ends as one of: <series>_<chunk>.parquet (data), <series>_<chunk>.nodata (the API has
nothing: not retried on rerun), or nothing (failed after retries: picked up on rerun). The API key is never printed.

usage:  neighbour_data.py smoke      one chunk of every series for NL
        neighbour_data.py fetch [Z..] everything, or the zones given (resumable; disjoint zone lists may run in parallel)
        neighbour_data.py process    gaps, native resolution, 15-min resampling, NTC, reconciliation, capacity
Data lives only under $SP/entsoe/neighbours/ (raw/<zone>/, derived/, summary/).
"""
import glob, json, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
OUT = f"{SP}/entsoe/neighbours"
TZ = "Europe/Berlin"
DE = "DE_LU"
SDAC = ["FR", "NL", "BE", "AT", "CZ", "PL", "DK_1", "DK_2", "SE_4", "NO_2"]
ZONES = SDAC + ["CH"]
START, END = pd.Timestamp("2025-08-01", tz=TZ), pd.Timestamp("2026-11-10", tz=TZ)  # end exclusive
QMTU = pd.Timestamp("2025-10-01", tz=TZ)  # SDAC 15-minute MTU go-live
CHUNKS = list(pd.date_range("2025-08-01", "2026-11-01", freq="3MS", tz=TZ)) + [END]  # 08-01, 11-01, 02-01, 05-01, 08-01, 11-01, 11-10
SERIES = ["load", "ws", "price", "sched_out", "sched_in", "ntc_out", "ntc_in", "cap"]


# ----------------------------------------------------------------------------------------------- fetch
def client():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip(), timeout=90)


def queries(c, z):
    return {
        "load": lambda s, e: c.query_load_forecast(z, start=s, end=e),
        "ws": lambda s, e: c.query_wind_and_solar_forecast(z, start=s, end=e),
        "price": lambda s, e: c.query_day_ahead_prices(z, start=s, end=e),
        "sched_out": lambda s, e: c.query_scheduled_exchanges(DE, z, start=s, end=e, dayahead=True),
        "sched_in": lambda s, e: c.query_scheduled_exchanges(z, DE, start=s, end=e, dayahead=True),
        "ntc_out": lambda s, e: c.query_net_transfer_capacity_dayahead(DE, z, start=s, end=e),
        "ntc_in": lambda s, e: c.query_net_transfer_capacity_dayahead(z, DE, start=s, end=e),
        "cap": lambda s, e: c.query_installed_generation_capacity(z, start=s, end=e),
    }


def windows(series, first_only=False):
    if series == "cap":  # yearly product: query the calendar year
        w = [(str(y), pd.Timestamp(f"{y}-01-01", tz=TZ), pd.Timestamp(f"{y + 1}-01-01", tz=TZ)) for y in (2025, 2026)]
    else:
        w = [(s.strftime("%Y-%m"), s, min(e, END)) for s, e in zip(CHUNKS[:-1], CHUNKS[1:]) if s < END]
    return w[:1] if first_only else w


def mask(msg, key):
    return msg.replace(key, "KEY") if key else msg


def log(*a):
    print(*a, flush=True)
    with open(f"{OUT}/fetch.log", "a") as f:
        print(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), *a, file=f)


def fetch(zones, first_only=False):
    from entsoe.exceptions import NoMatchingDataError, InvalidBusinessParameterError, InvalidPSRTypeError
    import requests
    c = client(); key = c.api_key
    os.makedirs(OUT, exist_ok=True)
    for z in zones:
        d = f"{OUT}/raw/{z}"; os.makedirs(d, exist_ok=True)
        for nm, f in queries(c, z).items():
            for tag, s, e in windows(nm, first_only):
                base = f"{d}/{nm}_{tag}"
                if os.path.exists(base + ".parquet") or os.path.exists(base + ".nodata"):
                    continue
                for attempt in range(4):
                    time.sleep(1.0)
                    try:
                        r = f(s, e)
                        r = r.to_frame(nm) if isinstance(r, pd.Series) else r
                        r.columns = [str(x) for x in r.columns]
                        r.to_parquet(base + ".parquet")
                        log("ok", z, nm, tag, r.shape, list(r.columns)[:4]); break
                    except (NoMatchingDataError, InvalidBusinessParameterError, InvalidPSRTypeError) as ex:
                        open(base + ".nodata", "w").write(type(ex).__name__ + "\n")
                        log("nodata", z, nm, tag, type(ex).__name__); break
                    except requests.HTTPError as ex:
                        code = ex.response.status_code if ex.response is not None else None
                        msg = mask(str(ex), key)[:160]
                        if code is not None and 400 <= code < 500 and code != 429:  # request rejected: not retryable
                            open(base + ".nodata", "w").write(f"HTTP {code}: {msg}\n")
                            log("nodata", z, nm, tag, f"HTTP {code}", msg); break
                        log("retry", z, nm, tag, attempt, f"HTTP {code}", msg); time.sleep(10 + 20 * attempt)
                    except Exception as ex:  # connection errors, timeouts, parser surprises
                        log("retry", z, nm, tag, attempt, type(ex).__name__, mask(str(ex), key)[:160]); time.sleep(10 + 20 * attempt)
                else:
                    log("FAILED", z, nm, tag, "(left for rerun)")
    log("finished", zones if first_only else "all")


# ----------------------------------------------------------------------------------------------- process
def load_series(z, nm):
    """Concatenated raw chunks restricted to the period, duplicate stamps (inclusive chunk ends) dropped."""
    fs = sorted(glob.glob(f"{OUT}/raw/{z}/{nm}_*.parquet"))
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs])
    df.index = pd.DatetimeIndex(df.index).tz_convert(TZ)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if nm != "cap":
        df = df[(df.index >= START) & (df.index < END)]
    return df


def day_res(idx):
    """Native resolution per local calendar day: the modal spacing of that day's stamps."""
    s = pd.Series(idx, index=idx).diff().shift(-1)
    s = s[s.notna()]
    return s.groupby(s.index.date).agg(lambda x: x.mode().iloc[0])


def res_text(dr, a, b):
    """Native resolution(s) between a and b as text, e.g. '60min to 2025-12-02, then 15min'."""
    d = dr[(dr.index >= a.date()) & (dr.index < b.date())]
    if d.empty:
        return "-"
    lab = d.map(lambda t: f"{int(t / pd.Timedelta('1min'))}min")
    runs = lab[lab != lab.shift()]
    if len(runs) > 4:  # flip-flopping: give the shares instead
        return ", ".join(f"{k} {v:.0%}" for k, v in lab.value_counts(normalize=True).items())
    parts = [runs.iloc[0]]
    for day, v in list(runs.items())[1:]:
        parts.append(f"from {day} {v}")
    return ", ".join(parts)


def expand(df, a, b):
    """Column by column (a frame can mix resolutions: BE solar went to 15 min while BE wind stayed hourly), see
    _expand_one; the NaN padding a mixed frame carries is dropped first so it is not mistaken for gaps."""
    grid = pd.date_range(a, b, freq="15min", inclusive="left")
    return pd.concat([_expand_one(df[[c]].dropna(), a, b) for c in df.columns], axis=1).reindex(grid)


def _expand_one(df, a, b):
    """df on a 15-minute grid [a, b). Each native value is repeated over the quarter hours of its own interval
    (an hourly MW or price value becomes four equal quarter-hour values; no interpolation, no division by 4).
    The interval is the distance to the next stamp, capped at that day's native resolution (and at 60 min), so a
    missing stamp stays NaN instead of being filled by its predecessor."""
    grid = pd.date_range(a, b, freq="15min", inclusive="left")
    sub = df[(df.index >= a - pd.Timedelta("1h")) & (df.index < b)]
    if sub.empty:
        return pd.DataFrame(index=grid, columns=df.columns, dtype=float)
    dr = day_res(df.index)
    cap = pd.Series(sub.index.date, index=sub.index).map(dr).fillna(pd.Timedelta("15min")).clip(upper=pd.Timedelta("1h"))
    step = pd.Series(sub.index, index=sub.index).diff().shift(-1).fillna(cap.iloc[-1])
    step = np.minimum(step.values, cap.values)
    k = np.maximum((pd.to_timedelta(step) // pd.Timedelta("15min")).astype(int), 1)
    utc = sub.index.tz_convert("UTC")
    new_idx = np.concatenate([utc[i] + pd.to_timedelta(np.arange(k[i]) * 15, "min") for i in range(len(sub))])
    out = pd.DataFrame(np.repeat(sub.values, k, axis=0), index=pd.DatetimeIndex(new_idx, tz="UTC"), columns=sub.columns)
    out = out[~out.index.duplicated(keep="first")].tz_convert(TZ)
    return out.reindex(grid)


def to_15min(df):
    """The derived quarter-hourly series from 2025-10-01 (see expand)."""
    return expand(df, QMTU, END)


def process():
    os.makedirs(f"{OUT}/derived", exist_ok=True); os.makedirs(f"{OUT}/summary", exist_ok=True)
    rows, frames = [], {}
    for z in ZONES:
        for nm in SERIES:
            nod = sorted(glob.glob(f"{OUT}/raw/{z}/{nm}_*.nodata"))
            df = load_series(z, nm)
            n_chunks = len(windows(nm))
            n_data = len(glob.glob(f"{OUT}/raw/{z}/{nm}_*.parquet"))
            base = dict(zone=z, series=nm, chunks=n_chunks, chunks_data=n_data, chunks_nodata=len(nod),
                        chunks_failed=n_chunks - n_data - len(nod))
            if df is None or df.empty:
                rows.append(dict(base, column="", available=False)); continue
            frames[(z, nm)] = df
            if nm == "cap":
                for col in df.columns:
                    rows.append(dict(base, column=col, available=True, first=str(df.index.min().date()), last=str(df.index.max().date())))
                continue
            pre, post = expand(df, START, QMTU), to_15min(df)
            post.to_parquet(f"{OUT}/derived/{z}_{nm}_15min.parquet")
            for col in df.columns:
                nn = df[col].dropna()
                dr = day_res(nn.index)
                rows.append(dict(base, column=col, available=len(nn) > 0,
                                 first=str(nn.index.min()) if len(nn) else "", last=str(nn.index.max()) if len(nn) else "",
                                 res_pre=res_text(dr, START, QMTU), res_post=res_text(dr, QMTU, END),
                                 gap_pre=round(float(pre[col].isna().mean()), 4), gap_post=round(float(post[col].isna().mean()), 4),
                                 gap_all=round(float(pd.concat([pre[col], post[col]]).isna().mean()), 4)))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/summary/availability.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    print(tab.drop(columns=["chunks"]).to_string(index=False))

    # NTC per border and direction
    nt = []
    for z in ZONES:
        for nm, lab in (("ntc_out", f"DE_LU->{z}"), ("ntc_in", f"{z}->DE_LU")):
            df = frames.get((z, nm))
            if df is None:
                nt.append(dict(border=lab, available=False)); continue
            s = df.iloc[:, 0].dropna()
            nt.append(dict(border=lab, available=True, first=str(s.index.min().date()), last=str(s.index.max().date()),
                           n=len(s), p10=s.quantile(.1), median=s.median(), p90=s.quantile(.9), min=s.min(), max=s.max(),
                           share_zero=round(float((s == 0).mean()), 3)))
    nt = pd.DataFrame(nt); nt.to_csv(f"{OUT}/summary/ntc.csv", index=False)
    print("\nNTC\n", nt.to_string(index=False))

    # Reconciliation: net export of DE_LU = sum_Z (DE->Z - Z->DE) vs DA net position (positive = export)
    npf = sorted(glob.glob(f"{SP}/entsoe/da/netpos_*.parquet"))
    netpos = pd.concat([pd.read_parquet(f) for f in npf])["netpos"]
    netpos.index = pd.DatetimeIndex(netpos.index).tz_convert(TZ)
    netpos = netpos[~netpos.index.duplicated()].sort_index()
    netpos = netpos[(netpos.index >= START) & (netpos.index < END)]
    np15 = to_15min(netpos.to_frame())["netpos"]
    net = {}
    for z in ZONES:
        o, i = frames.get((z, "sched_out")), frames.get((z, "sched_in"))
        if o is None and i is None:
            continue
        o15 = to_15min(o).iloc[:, 0] if o is not None else 0.0
        i15 = to_15min(i).iloc[:, 0] if i is not None else 0.0
        net[z] = o15 - i15
        # pre-October on the hourly grid (means of whatever native values exist)
    net15 = pd.DataFrame(net)
    net15.to_parquet(f"{OUT}/derived/DE_LU_sched_net_export_15min.parquet")
    hour_pre = {}
    for z in net:
        o, i = frames.get((z, "sched_out")), frames.get((z, "sched_in"))
        oh = o.iloc[:, 0].resample("h").mean() if o is not None else 0.0
        ih = i.iloc[:, 0].resample("h").mean() if i is not None else 0.0
        hour_pre[z] = (oh - ih)
    hp = pd.DataFrame(hour_pre); hp = hp[hp.index < QMTU]
    nph = netpos.resample("h").mean(); nph = nph[nph.index < QMTU]
    rec = []
    def stats(label, a, b):
        j = pd.concat([a.rename("sched"), b.rename("netpos")], axis=1).dropna()
        d = j["sched"] - j["netpos"]
        rec.append(dict(case=label, n=len(j), bias=round(d.mean(), 1), mae=round(d.abs().mean(), 1),
                        p99_abs=round(d.abs().quantile(.99), 1), max_abs=round(d.abs().max(), 1),
                        share_within_1MW=round(float((d.abs() <= 1).mean()), 3),
                        corr=round(j.corr().iloc[0, 1], 5), mean_netpos=round(j["netpos"].mean(), 1)))
    sdac_cols = [z for z in SDAC if z in net15]
    complete15 = net15[sdac_cols].notna().all(axis=1)
    stats("15min from 2025-10-01, 10 SDAC borders", net15[sdac_cols].sum(axis=1, min_count=len(sdac_cols)), np15)
    if "CH" in net15:
        stats("15min from 2025-10-01, 10 SDAC + CH", net15[sdac_cols + ["CH"]].sum(axis=1, min_count=len(sdac_cols) + 1), np15)
    hcols = [z for z in SDAC if z in hp]
    stats("hourly 2025-08-01..09-30, 10 SDAC borders", hp[hcols].sum(axis=1, min_count=len(hcols)), nph)
    if "CH" in hp:
        stats("hourly 2025-08-01..09-30, 10 SDAC + CH", hp[hcols + ["CH"]].sum(axis=1, min_count=len(hcols) + 1), nph)
    rec = pd.DataFrame(rec); rec.to_csv(f"{OUT}/summary/reconciliation.csv", index=False)
    print("\nReconciliation (sched net export - netpos)\n", rec.to_string(index=False))
    print("15-min stamps with all 10 SDAC borders present:", round(float(complete15.mean()), 4))
    # Per-border mean net export, as context for the reconciliation
    print("\nMean DA scheduled net export DE_LU->Z (MW, from 2025-10-01):\n", net15.mean().round(1).to_string())
    net15.mean().round(1).to_csv(f"{OUT}/summary/mean_net_export.csv")

    # Installed capacity 2026 by production type
    cap = {}
    for z in ZONES:
        df = frames.get((z, "cap"))
        if df is None:
            continue
        y = df[df.index.year == 2026]
        cap[(z, 2026 if len(y) else int(df.index.year.max()))] = (y if len(y) else df[df.index.year == df.index.year.max()]).iloc[-1]
    capt = pd.DataFrame({f"{z}({y})": v for (z, y), v in cap.items()}).fillna(0.0)
    sdac_c = [c for c in capt.columns if c.split("(")[0] in SDAC]
    capt["aggregate_SDAC10"] = capt[sdac_c].sum(axis=1)
    capt = capt.sort_values("aggregate_SDAC10", ascending=False)
    capt.to_csv(f"{OUT}/summary/capacity_2026.csv")
    print("\nInstalled capacity (MW), 2026 where published\n", capt.round(0).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "smoke":
        fetch(["NL"], first_only=True)
    elif cmd == "fetch":
        fetch(sys.argv[2:] or ZONES)
    elif cmd == "process":
        process()
    else:
        sys.exit(__doc__)
