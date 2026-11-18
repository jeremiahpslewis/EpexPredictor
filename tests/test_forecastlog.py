"""Tests for the log of issued forecasts."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from predictor.model.forecastlog import ForecastLog

LAST_KNOWN = datetime(2025, 11, 1, 22, 45, tzinfo=timezone.utc)
ISSUED = datetime(2025, 11, 1, 12, 20, tzinfo=timezone.utc)


def series(value: float, slots: int = 300) -> pd.Series:
    index = pd.date_range(LAST_KNOWN - timedelta(hours=5), periods=slots, freq="15min")
    return pd.Series(value, index=index)


def test_nothing_is_written_without_a_data_directory():
    assert ForecastLog(None, "DE_LU").append(LAST_KNOWN, {"price": series(10.0)}, ISSUED) is False


def test_an_issue_holds_the_time_after_the_last_known_price(tmp_path):
    forecastlog = ForecastLog(str(tmp_path), "DE_LU")
    assert forecastlog.append(LAST_KNOWN, {"trees": series(10.0), "timestral": series(20.0), "price": series(15.0)}, ISSUED) is True

    frame = ForecastLog.read(forecastlog.path)
    assert len(frame) == ForecastLog.HORIZON
    assert frame["time"].min() == pd.Timestamp(LAST_KNOWN) + timedelta(minutes=15)
    assert (frame["issued"] == pd.Timestamp(ISSUED).floor("min")).all()
    assert (frame["last_known"] == pd.Timestamp(LAST_KNOWN)).all()
    assert (frame["trees"] == 10.0).all() and (frame["timestral"] == 20.0).all() and (frame["price"] == 15.0).all()


def test_one_issue_per_hour_and_last_known_price(tmp_path):
    forecastlog = ForecastLog(str(tmp_path), "DE_LU")
    columns = {"trees": series(10.0), "price": series(15.0)}
    assert forecastlog.append(LAST_KNOWN, columns, ISSUED) is True
    assert forecastlog.append(LAST_KNOWN, columns, ISSUED + timedelta(minutes=10)) is False
    assert forecastlog.append(LAST_KNOWN, columns, ISSUED + timedelta(hours=1)) is True
    assert forecastlog.append(LAST_KNOWN + timedelta(days=1), columns, ISSUED + timedelta(hours=1, minutes=5)) is True

    frame = ForecastLog.read(forecastlog.path)
    assert frame["issued"].nunique() == 3
    # the series ends 183 quarter hours after the later last known price
    assert len(frame) == 2 * ForecastLog.HORIZON + 183


def test_the_log_survives_a_restart(tmp_path):
    columns = {"trees": series(10.0), "price": series(15.0)}
    assert ForecastLog(str(tmp_path), "DE_LU").append(LAST_KNOWN, columns, ISSUED) is True
    again = ForecastLog(str(tmp_path), "DE_LU")
    assert again.append(LAST_KNOWN, columns, ISSUED + timedelta(hours=2)) is True
    assert len(ForecastLog.read(again.path)) == 2 * ForecastLog.HORIZON


def test_a_failure_does_not_raise(tmp_path):
    forecastlog = ForecastLog(str(tmp_path), "DE_LU")
    forecastlog.path = str(tmp_path / "missing" / "\0bad")
    assert forecastlog.append(LAST_KNOWN, {"price": series(15.0)}, ISSUED) is False


def test_issues_with_and_without_a_forecaster_share_one_file(tmp_path):
    forecastlog = ForecastLog(str(tmp_path), "DE_LU")
    assert forecastlog.append(LAST_KNOWN, {"trees": series(10.0), "timestral": series(20.0), "price": series(15.0)}, ISSUED) is True
    with_market = {"trees": series(10.0), "timestral": series(20.0), "market": series(30.0), "price": series(20.0)}
    assert forecastlog.append(LAST_KNOWN, with_market, ISSUED + timedelta(hours=1)) is True
    assert forecastlog.append(LAST_KNOWN, {"trees": series(10.0), "price": series(10.0)}, ISSUED + timedelta(hours=2)) is True

    frame = ForecastLog.read(forecastlog.path)
    assert list(frame.columns) == ["time", "issued", "last_known", *ForecastLog.FORECASTS]
    by_issue = frame.groupby("issued")
    assert by_issue["market"].count().tolist() == [0, ForecastLog.HORIZON, 0]
    assert by_issue["timestral"].count().tolist() == [ForecastLog.HORIZON, ForecastLog.HORIZON, 0]
    assert by_issue["price"].first().tolist() == [15.0, 20.0, 10.0]
