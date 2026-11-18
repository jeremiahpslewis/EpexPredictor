"""Tests for the scoring of the forecast logs."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predictor import scorelog
from predictor.model.forecastlog import ForecastLog

START = datetime(2025, 11, 1, 22, 45, tzinfo=timezone.utc)


def write_logs(tmp_path, days: int = 8):
    index = pd.date_range(START - timedelta(days=1), START + timedelta(days=days + 3), freq="15min", name="time")
    prices = pd.DataFrame({"price": 10.0}, index=index)
    prices.to_json(tmp_path / "prices_v3_DE_LU.json.gz", compression="gzip")
    pd.DataFrame({"tso_load": 50.0, "tso_wind": 10.0, "tso_solar": 5.0}, index=index).to_json(tmp_path / "generation_v1_DE_LU.json.gz", compression="gzip")
    forecastlog = ForecastLog(str(tmp_path), "DE_LU")
    loadlog = ForecastLog(str(tmp_path), "DE_LU", "loadforecastlog", ("trees", "timestral"))
    for day in range(days):
        last = START + timedelta(days=day)
        columns = {"trees": pd.Series(12.0, index=index), "timestral": pd.Series(11.0, index=index), "price": pd.Series(11.5, index=index)}
        if day % 2 == 0:
            columns["market"] = pd.Series(10.5, index=index)
        assert forecastlog.append(last, columns, last + timedelta(hours=1))
        assert loadlog.append(last, {"trees": pd.Series(38.0, index=index), "timestral": pd.Series(34.0, index=index)}, last + timedelta(hours=1))


def test_scores_each_forecaster_against_the_tree_models(tmp_path):
    write_logs(tmp_path)
    table = scorelog.score(ForecastLog.read(str(tmp_path / "forecastlog_DE_LU.csv.gz")), scorelog.read_store(str(tmp_path / "prices_v3_DE_LU.json.gz"))["price"])
    first_day = table[table["lead"] == "first day"].set_index("forecast")
    assert first_day.loc["trees", "mae"] == pytest.approx(2.0)
    assert first_day.loc["timestral", "mae"] == pytest.approx(1.0)
    assert first_day.loc["timestral", "change"] == pytest.approx(-1.0)
    assert first_day.loc["timestral", "low"] == pytest.approx(-1.0) and first_day.loc["timestral", "high"] == pytest.approx(-1.0)
    assert first_day.loc["timestral", "values"] == 8 * 96 and first_day.loc["timestral", "days"] == 8
    # the market model gave a forecast on every other day only
    assert first_day.loc["market", "days"] == 4 and first_day.loc["market", "mae"] == pytest.approx(0.5)
    assert set(table["lead"]) == set(scorelog.LEADS)


def test_report_covers_both_logs(tmp_path):
    write_logs(tmp_path)
    text = scorelog.main(str(tmp_path))
    assert "Price forecast" in text and "Residual load forecast" in text
    # trees 38, Timestral 34, their average 36 against a published 35
    assert "average     1.000 GW   against the tree models -2.000" in text


def test_report_without_logs(tmp_path):
    assert "no forecast log" in scorelog.main(str(tmp_path))
