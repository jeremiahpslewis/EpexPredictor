"""
A log of the forecasts as they were issued, to check later how good they were.

Every time the forecast changes, the prices forecast for the quarter hours after the last known price are appended:
what the tree models said, what each further forecaster said, and the mix that was issued. One issue per hour at
most. The file is a gzipped CSV in the data directory and is never rewritten, only added to.
"""

import gzip
import logging
import os
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


class ForecastLog:
    # quarter hours after the last known price that are logged (two days)
    HORIZON = 192
    # the columns of the file, always all of them: a forecaster that gave nothing in an issue leaves its column empty
    FORECASTS = ("trees", "timestral", "market", "price")

    path: str | None
    last_issue: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def __init__(self, storage_dir: str | None, zone: str, name: str = "forecastlog", forecasts: tuple[str, ...] | None = None):
        """name: start of the file name. forecasts: the columns of the file, if not the ones of the price forecast"""
        self.path = os.path.join(storage_dir, f"{name}_{zone}.csv.gz") if storage_dir else None
        if forecasts is not None:
            self.FORECASTS = forecasts

    def append(self, last_known: datetime, columns: dict[str, pd.Series], issued: datetime | None = None) -> bool:
        """
        Add one issue. columns: forecasts by name, indexed by time. last_known: the last time with a known value;
        what comes after it is logged. Returns whether it was written.
        A failure is logged and never stops the forecast.
        """
        if self.path is None:
            return False
        try:
            issued = pd.Timestamp(issued or datetime.now(timezone.utc)).tz_convert("UTC")
            last = pd.Timestamp(last_known).tz_convert("UTC")
            key = (issued.floor("h"), last)
            if key == self.last_issue:
                return False

            frame = pd.DataFrame({name: series for name, series in columns.items() if series is not None})
            frame = frame.reindex(columns=list(self.FORECASTS))
            frame = frame.loc[frame.index > last].iloc[:self.HORIZON].dropna(how="all")
            if len(frame) == 0:
                return False
            frame.index = frame.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
            frame.insert(0, "last_known", last.strftime("%Y-%m-%dT%H:%MZ"))
            frame.insert(0, "issued", issued.strftime("%Y-%m-%dT%H:%MZ"))
            frame.index.name = "time"

            new_file = not os.path.exists(self.path)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with gzip.open(self.path, "at", encoding="utf-8") as file:
                frame.round(4).to_csv(file, header=new_file)
            self.last_issue = key
            return True
        except Exception:
            log.exception("failed to write the forecast log - continuing without it")
            return False

    @staticmethod
    def read(path: str) -> pd.DataFrame:
        """The whole log. A forecaster that gave nothing in an issue has NaN there."""
        with gzip.open(path, "rt", encoding="utf-8") as file:
            lines = file.read().splitlines()
        header = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:] if line and not line.startswith("time,")]
        frame = pd.DataFrame([row + [""] * (len(header) - len(row)) for row in rows], columns=header)
        for column in ("time", "issued", "last_known"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        for column in header[3:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame
