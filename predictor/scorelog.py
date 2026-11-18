"""
Scores the forecast logs against what followed: how good each forecaster was, by time after the last known value.

    python -m predictor.scorelog [data directory] [zone]

Reads forecastlog_<zone>.csv.gz against the stored prices, and loadforecastlog_<zone>.csv.gz against the stored
published residual load. Each target time counts once per issue it was forecast in. The comparison with the tree
models is paired, and its range comes from resampling whole days of issue (5 to 95 %).
"""

import os
import sys

import numpy as np
import pandas as pd

from .model.forecastlog import ForecastLog

# quarter hours after the last known value: name -> (from, to)
LEADS = {"first 6 hours": (0, 24), "first day": (0, 96), "second day": (96, 192)}


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


def report(title: str, unit: str, table: pd.DataFrame) -> str:
    if len(table) == 0:
        return f"{title}: nothing to score yet"
    lines = [title]
    for lead in LEADS:
        part = table[table["lead"] == lead]
        if len(part) == 0:
            continue
        lines.append(f"  {lead} ({int(part['days'].max())} days of issue)")
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
        out.append(report("Price forecast, mean absolute error", "ct/kWh", score(ForecastLog.read(path), prices)))
    else:
        out.append(f"no forecast log at {path}")
    path = os.path.join(datadir, f"loadforecastlog_{zone}.csv.gz")
    if os.path.exists(path):
        published = read_store(os.path.join(datadir, f"generation_v1_{zone}.json.gz"))
        residual = published["tso_load"] - published["tso_wind"] - published["tso_solar"]
        log = ForecastLog.read(path)
        log["average"] = 0.5 * (log["trees"] + log["timestral"])
        out.append(report("Residual load forecast, mean absolute error", "GW", score(log, residual)))
    return "\n\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else os.getenv("EPEXPREDICTOR_DATADIR", "."), sys.argv[2] if len(sys.argv) > 2 else "DE_LU"))
