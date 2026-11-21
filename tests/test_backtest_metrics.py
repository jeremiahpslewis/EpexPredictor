"""Tests for the coverage and accuracy metrics of the rolling backtest."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predictor.backtest_metrics import Metrics, last_week, table

D0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
D1, D2, D3 = (D0 + timedelta(days=k) for k in (1, 2, 3))
INDEX = pd.date_range(D0 - timedelta(days=7), D3 + timedelta(hours=1), freq="15min")


def flat(value: float) -> pd.Series:
    return pd.Series(value, index=INDEX)


def test_each_quarter_hour_belongs_to_one_forecast_day():
    # a large error exactly at d1 is the first quarter hour of day 2 and must not count for day 1
    actual = flat(10.0)
    forecast = flat(10.0)
    forecast[pd.Timestamp(D1)] = 1010.0
    m = Metrics(("stage1",))
    m.add(actual, {"stage1": forecast}, D0, D1, D2, D3)
    assert [m.expected[d] for d in (1, 2, 3)] == [96, 96, 96]
    assert m.mae(1, "stage1") == 0.0
    assert m.mae(2, "stage1") == pytest.approx(1000.0 / 96)
    assert m.mae(3, "stage1") == 0.0
    # nor does the error at d3 (outside the window) count anywhere
    forecast[pd.Timestamp(D3)] = 1010.0
    m = Metrics(("stage1",))
    m.add(actual, {"stage1": forecast}, D0, D1, D2, D3)
    assert m.mae(3, "stage1") == 0.0


def test_a_missed_spike_shows_as_lower_coverage_not_as_a_lower_error():
    actual = flat(10.0)
    spike = pd.Timestamp(D0 + timedelta(hours=18))
    actual[spike] = 100.0
    actual[pd.Timestamp(D0 + timedelta(hours=3))] = float("nan")   # one quarter hour without an actual price
    stage1 = flat(10.0)                                             # forecasts the spike as a normal hour
    stage2 = flat(10.0)
    stage2[spike] = float("nan")                                     # gives no forecast at the spike
    m = Metrics(("stage1", "stage2", "last_week"))
    m.add(actual, {"stage1": stage1, "stage2": stage2, "last_week": last_week(actual, D0, D3)}, D0, D1, D2, D3)
    assert (m.expected[1], m.actual[1]) == (96, 95)
    assert m.covered[1] == {"stage1": 95, "stage2": 94, "last_week": 95}
    assert m.common[1] == 94
    # on the common quarter hours the spike is left out for every model alike
    assert m.mae(1, "stage1") == m.mae(1, "stage2") == 0.0
    assert m.coverage(1, "stage2") == pytest.approx(94 / 95)
    # had stage 1 been scored on all of its own quarter hours, it would carry the spike
    assert (stage1[actual.index[(actual.index >= D0) & (actual.index < D1)]] - actual).abs().max() == 90.0
    text = table({"DE": m})
    assert "| DE | 1 | 96 | 95 | 95 (100.0%) | 94 (98.9%) | 95 (100.0%) | 94 |" in text


def test_last_week_uses_the_price_seven_days_before():
    actual = pd.Series(range(len(INDEX)), index=INDEX, dtype=float)
    baseline = last_week(actual, D0, D3)
    assert len(baseline) == 3 * 96
    assert baseline[pd.Timestamp(D1)] == actual[pd.Timestamp(D1 - timedelta(days=7))]
    # it only reads prices known before d0
    assert baseline.index.min() == pd.Timestamp(D0) and (baseline.index - pd.Timedelta(days=7)).max() < pd.Timestamp(D0)


def test_a_model_left_out_counts_as_uncovered():
    m = Metrics(("stage1", "stage2"))
    m.add(flat(10.0), {"stage1": flat(11.0)}, D0, D1, D2, D3)
    assert m.covered[1] == {"stage1": 96, "stage2": 0}
    assert m.common[1] == 0
