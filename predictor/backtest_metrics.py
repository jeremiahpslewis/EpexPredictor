"""
Coverage and accuracy of a rolling backtest, by forecast day.

Each backtest window starts at d0 and covers three forecast days, [d0, d1), [d1, d2) and [d2, d3). The intervals are
half-open, so every quarter hour belongs to exactly one forecast day. For each day the metrics count:

- expected: the quarter hours of the day;
- actual: those with an actual price;
- covered: for each model, those with an actual price and a forecast of that model;
- common: those with an actual price and a forecast of every model.

The errors of all models are taken on the common quarter hours only, so the models are compared on the same
timestamps. A model that gives no forecast for a quarter hour shows up as lower coverage, not as a lower error.
Errors are pooled over all common quarter hours of all windows (each quarter hour counts once).
"""

import math

import pandas as pd

SLOT = pd.Timedelta(minutes=15)
DAYS = (1, 2, 3)


def last_week(actual: pd.Series, start, end) -> pd.Series:
    """
    The same-slot-last-week baseline for [start, end): the actual price 7 x 24 hours earlier (UTC), indexed by the
    target time. It only uses prices before start as long as end - start is at most seven days.
    """
    index = pd.date_range(start, end, freq=SLOT, inclusive="left")
    return pd.Series(actual.reindex(index - pd.Timedelta(days=7)).to_numpy(), index=index)


class Metrics:
    """Collects coverage and errors by forecast day over a rolling backtest, for several models at once."""

    def __init__(self, models: tuple[str, ...] = ("stage1", "stage2", "last_week")):
        self.models = models
        self.windows = 0
        self.expected = {d: 0 for d in DAYS}
        self.actual = {d: 0 for d in DAYS}
        self.covered = {d: {m: 0 for m in models} for d in DAYS}
        self.common = {d: 0 for d in DAYS}
        self.abs_sum = {d: {m: 0.0 for m in models} for d in DAYS}
        self.sq_sum = {d: {m: 0.0 for m in models} for d in DAYS}

    def add(self, actual: pd.Series, predictions: dict[str, pd.Series], d0, d1, d2, d3):
        """
        actual: actual prices indexed by time. predictions: a forecast by model name, indexed by time (a model left
        out counts as giving no forecast). d0..d3: the bounds of the three forecast days.
        """
        self.windows += 1
        for day, (start, end) in zip(DAYS, ((d0, d1), (d1, d2), (d2, d3))):
            index = pd.date_range(start, end, freq=SLOT, inclusive="left")
            truth = actual.reindex(index)
            has_actual = truth.notna()
            self.expected[day] += len(index)
            self.actual[day] += int(has_actual.sum())
            forecasts = {m: (predictions[m].reindex(index) if predictions.get(m) is not None else pd.Series(float("nan"), index=index))
                         for m in self.models}
            common = has_actual.copy()
            for m, forecast in forecasts.items():
                self.covered[day][m] += int((has_actual & forecast.notna()).sum())
                common &= forecast.notna()
            self.common[day] += int(common.sum())
            for m, forecast in forecasts.items():
                error = (forecast[common] - truth[common])
                self.abs_sum[day][m] += float(error.abs().sum())
                self.sq_sum[day][m] += float(error.pow(2).sum())

    def mae(self, day: int, model: str) -> float:
        return self.abs_sum[day][model] / self.common[day] if self.common[day] else float("nan")

    def rmse(self, day: int, model: str) -> float:
        return math.sqrt(self.sq_sum[day][model] / self.common[day]) if self.common[day] else float("nan")

    def coverage(self, day: int, model: str) -> float:
        """Share of the quarter hours with an actual price for which the model gave a forecast."""
        return self.covered[day][model] / self.actual[day] if self.actual[day] else float("nan")


def table(metrics: dict[str, Metrics], baseline: str = "last_week") -> str:
    """A Markdown table: one row per region and forecast day."""
    some = next(iter(metrics.values()))
    models = some.models
    header = ["Region", "Day", "Expected", "Actual"] + [f"Covered {m}" for m in models] + ["Common"]
    header += [f"MAE {m}" for m in models] + [f"RMSE {m}" for m in models]
    header += [f"MAE {m} / {baseline}" for m in models if m != baseline]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for region, mt in metrics.items():
        for day in DAYS:
            row = [str(region), str(day), str(mt.expected[day]), str(mt.actual[day])]
            row += [f"{mt.covered[day][m]} ({mt.coverage(day, m):.1%})" for m in models]
            row += [str(mt.common[day])]
            row += [f"{mt.mae(day, m):.2f}" for m in models]
            row += [f"{mt.rmse(day, m):.2f}" for m in models]
            base = mt.mae(day, baseline) if baseline in models else float("nan")
            row += [f"{mt.mae(day, m) / base:.2f}" if base and not math.isnan(base) else "–" for m in models if m != baseline]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
