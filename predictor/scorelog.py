"""
Scores the forecast logs against what followed: how good each forecaster was, by time after the last known value.

    python -m predictor.scorelog [data directory] [zone]

Reads forecastlog_<zone>.csv.gz and coreforecastlog_<zone>.csv.gz against the stored prices, and
loadforecastlog_<zone>.csv.gz against the stored published residual load. Each target time counts once per issue it
was forecast in. The comparison with the tree models is paired, and its range comes from resampling whole days of issue (5 to 95 %).

These are the forecasts as they were issued, with the weather forecast of the time. They are a different thing from
the rolling backtest in performance_testing.py, which uses historical weather. The price log is also scored against
a same-slot-last-week baseline (the actual price 7 x 24 hours before the target time, known when the forecast was
issued), and each lead reports how many quarter hours were due, how many have an actual price and how many each
forecaster covered. With fewer than MIN_DAYS days of issue the figures are printed but no conclusion should be drawn.
"""

import os
import sys

import numpy as np
import pandas as pd

from .model.forecastlog import ForecastLog

# quarter hours after the last known value: name -> (from, to)
LEADS = {"first 6 hours": (0, 24), "first day": (0, 96), "second day": (96, 192)}
# days of issue below which the figures are reported with the conclusion left open
MIN_DAYS = 28
SLOT = pd.Timedelta(minutes=15)


def with_last_week(log: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    """Adds the same-slot-last-week baseline: the actual price 7 x 24 hours before each target time."""
    return log.assign(last_week=actual.reindex(log["time"] - pd.Timedelta(days=7)).to_numpy())


def coverage(log: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    """
    For each lead: the issues, the quarter hours due (issues x quarter hours of the lead), those with an actual
    price, and for each forecaster those with an actual price and its forecast.
    """
    names = [c for c in log.columns if c not in ("time", "issued", "last_known")]
    issues = log[["issued", "last_known"]].drop_duplicates()
    lead = ((log["time"] - log["last_known"]) / SLOT).round().astype(int) - 1
    has_actual = actual.reindex(log["time"]).notna().to_numpy()
    rows = []
    for name, (first, last) in LEADS.items():
        due = [last_known + (k + 1) * SLOT for last_known in issues["last_known"] for k in range(first, last)]
        row = {"lead": name, "issues": len(issues), "days": issues["issued"].dt.floor("D").nunique(),
               "expected": len(due), "actual": int(actual.reindex(pd.DatetimeIndex(due)).notna().sum()) if due else 0}
        part = ((lead >= first) & (lead < last)).to_numpy() & has_actual
        for forecast in names:
            row[forecast] = int((part & log[forecast].notna().to_numpy()).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def read_store(path: str) -> pd.DataFrame:
    data = pd.read_json(path, compression="gzip")
    if pd.api.types.is_integer_dtype(data.index.dtype):
        data.index = pd.to_datetime(data.index, unit="ms", utc=True)
    elif data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    return data


def score(log: pd.DataFrame, actual: pd.Series, reference: str = "trees", seed: int = 0) -> pd.DataFrame:
    """
    Mean absolute error of each forecast column of the log by lead, on the rows where the actual value and all
    forecasts of the issue are there, and its change against the reference column.
    """
    names = [c for c in log.columns if c not in ("time", "issued", "last_known")]
    log = log.assign(actual=actual.reindex(log["time"]).to_numpy(),
                     lead=((log["time"] - log["last_known"]) / pd.Timedelta(minutes=15)).round().astype(int) - 1,
                     day=log["issued"].dt.floor("D"))
    rows = []
    rng = np.random.default_rng(seed)
    for lead, (first, last) in LEADS.items():
        part = log[(log["lead"] >= first) & (log["lead"] < last) & log["actual"].notna()]
        for name in names:
            both = part[part[name].notna() & part[reference].notna()] if reference in part else part[part[name].notna()]
            if len(both) == 0:
                continue
            error = (both[name] - both["actual"]).abs()
            row = {"lead": lead, "forecast": name, "mae": error.mean(), "values": len(both), "days": both["day"].nunique()}
            if reference in both and name != reference:
                paired = (error - (both[reference] - both["actual"]).abs()).groupby(both["day"]).agg(["sum", "count"])
                row["change"] = paired["sum"].sum() / paired["count"].sum()
                if len(paired) >= 5:
                    picks = rng.integers(0, len(paired), (2000, len(paired)))
                    sums, counts = paired["sum"].to_numpy()[picks].sum(axis=1), paired["count"].to_numpy()[picks].sum(axis=1)
                    row["low"], row["high"] = np.percentile(sums / counts, [5, 95])
            rows.append(row)
    return pd.DataFrame(rows)


def report(title: str, unit: str, table: pd.DataFrame, covered: pd.DataFrame | None = None) -> str:
    lines = [title]
    if covered is not None and len(covered):
        first = covered.iloc[0]
        lines.append(f"  {int(first['issues'])} issues on {int(first['days'])} days of issue logged")
        names = [c for c in covered.columns if c not in ("lead", "issues", "days", "expected", "actual")]
        for _, row in covered.iterrows():
            lines.append(f"  {row['lead']}: {int(row['expected'])} quarter hours due, {int(row['actual'])} with an actual price; covered "
                         + ", ".join(f"{n} {int(row[n])}" for n in names))
    if len(table) == 0:
        lines.append("  nothing to score yet")
        return "\n".join(lines) if covered is not None else f"{title}: nothing to score yet"
    for lead in LEADS:
        part = table[table["lead"] == lead]
        if len(part) == 0:
            continue
        days = int(part['days'].max())
        caveat = f"; fewer than {MIN_DAYS}, too few to conclude" if days < MIN_DAYS else ""
        lines.append(f"  {lead} ({days} days of issue{caveat})")
        for _, row in part.iterrows():
            line = f"    {row['forecast']:10s} {row['mae']:6.3f} {unit}"
            if pd.notna(row.get("change")):
                line += f"   against the tree models {row['change']:+.3f}"
                if pd.notna(row.get("low")):
                    line += f" ({row['low']:+.3f} .. {row['high']:+.3f})"
            lines.append(line)
    return "\n".join(lines)


def main(datadir: str, zone: str = "DE_LU") -> str:
    out = []
    path = os.path.join(datadir, f"forecastlog_{zone}.csv.gz")
    if os.path.exists(path):
        prices = read_store(os.path.join(datadir, f"prices_v3_{zone}.json.gz"))["price"]
        log = with_last_week(ForecastLog.read(path), prices)
        out.append(report("Price forecast as issued (price = the issued blend; last_week = same slot a week before), mean absolute error",
                          "ct/kWh", score(log, prices), coverage(log, prices)))
    else:
        out.append(f"no forecast log at {path}")
    path = os.path.join(datadir, f"loadforecastlog_{zone}.csv.gz")
    if os.path.exists(path):
        published = read_store(os.path.join(datadir, f"generation_v1_{zone}.json.gz"))
        residual = published["tso_load"] - published["tso_wind"] - published["tso_solar"]
        log = ForecastLog.read(path)
        log["average"] = 0.5 * (log["trees"] + log["timestral"])
        # what the application uses (see timestralload.py), counted from each issue's last published figure
        from .model.timestralload import TimestralLoad
        ahead = ((log["time"] - log["last_known"]) / pd.Timedelta(minutes=15)).round().astype(int) - 1
        weight = TimestralLoad.weights(int(ahead.max()) + 1)[ahead.to_numpy()]
        log["used"] = weight * log["timestral"] + (1.0 - weight) * log["trees"]
        out.append(report("Residual load forecast, mean absolute error", "GW", score(log, residual)))
    path = os.path.join(datadir, f"coreforecastlog_{zone}.csv.gz")
    if os.path.exists(path):
        prices = read_store(os.path.join(datadir, f"prices_v3_{zone}.json.gz"))["price"]
        out.append(report("Core of the market model (curves and one level, no Timestral), mean absolute error", "ct/kWh", score(ForecastLog.read(path), prices)))
    return "\n\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else os.getenv("EPEXPREDICTOR_DATADIR", "."), sys.argv[2] if len(sys.argv) > 2 else "DE_LU"))
