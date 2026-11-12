"""Tests for predictor.model.europeweather module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from predictor.model.europeweather import EuropeWeather
from predictor.model.priceregion import PriceRegionName
from predictor.model.weatherstore import WeatherStore

START = datetime(2025, 11, 1, tzinfo=timezone.utc)
END = datetime(2025, 11, 2, tzinfo=timezone.utc)


def weather_store(name: PriceRegionName, wind: list[float], irradiance: float, temp: float) -> WeatherStore:
    """A weather store answering with constant weather for the given wind speeds (one per location)."""
    dates = pd.date_range(start=START, end=END, freq="15min", tz="UTC")
    data = {}
    for i, w in enumerate(wind):
        data[f"wind_{i}"] = [w] * len(dates)
        data[f"irradiance_{i}"] = [irradiance] * len(dates)
        data[f"temp_{i}"] = [temp] * len(dates)
        data[f"pressure_{i}"] = [1013.0] * len(dates)
    store = WeatherStore(name.to_region())
    store.get_data = AsyncMock(return_value=pd.DataFrame(data, index=dates))
    return store


class TestEuropeWeather:
    """Tests for the large-scale weather features."""

    @pytest.mark.asyncio
    async def test_averages_over_zones(self):
        """Test the averages over zones, each zone counting once regardless of its number of locations."""
        europe = EuropeWeather([
            weather_store(PriceRegionName.DE, [45.0, 90.0], 100.0, 10.0),  # at/above rated wind speed
            weather_store(PriceRegionName.FR, [0.0, 22.5], 300.0, 20.0),
        ])

        features = await europe.get_data(START, END)

        assert features is not None
        assert list(features.columns) == ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]
        # DE: both locations at rated power -> 1.0. FR: (0 + 0.5^3) / 2
        assert features["eu_wind"].iloc[0] == pytest.approx((1.0 + 0.0625) / 2)
        assert features["eu_wind_northsea"].iloc[0] == pytest.approx(1.0)
        assert features["eu_irradiance"].iloc[0] == pytest.approx(200.0)
        assert features["eu_temperature"].iloc[0] == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_ignores_zones_outside_the_averaged_area(self):
        """Test that only the configured zones contribute."""
        spain = weather_store(PriceRegionName.ES, [45.0], 900.0, 40.0)
        europe = EuropeWeather([weather_store(PriceRegionName.FR, [22.5], 300.0, 20.0), spain])

        features = await europe.get_data(START, END)

        assert features is not None
        assert features["eu_irradiance"].iloc[0] == pytest.approx(300.0)
        assert "eu_wind_northsea" not in features.columns
        assert not spain.get_data.called

    @pytest.mark.asyncio
    async def test_no_features_without_zones(self):
        """Test that nothing is returned when no zone of the averaged area is available."""
        europe = EuropeWeather([weather_store(PriceRegionName.ES, [10.0], 100.0, 10.0)])

        assert await europe.get_data(START, END) is None

    @pytest.mark.asyncio
    async def test_recomputes_only_when_weather_changes(self):
        """Test that repeated requests for the same range reuse the result until a store is updated."""
        store = weather_store(PriceRegionName.FR, [22.5], 300.0, 20.0)
        europe = EuropeWeather([store])

        first = await europe.get_data(START, END)
        assert await europe.get_data(START, END) is first

        store.last_updated = datetime(2025, 11, 3, tzinfo=timezone.utc)
        assert await europe.get_data(START, END) is not first
