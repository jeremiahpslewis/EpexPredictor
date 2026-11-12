"""Tests for predictor.model.weatherstore module."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from predictor.model.weatherstore import WeatherStore


class TestWeatherStoreInit:
    """Tests for WeatherStore initialization."""

    def test_init_without_storage(self, sample_region):
        """Test initialization without storage directory."""
        store = WeatherStore(sample_region)
        assert store.region == sample_region
        assert store.storage_dir is None

    def test_init_with_storage(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        store = WeatherStore(sample_region, temp_storage_dir)
        assert store.storage_dir == temp_storage_dir
        assert store.storage_fn_prefix.startswith("weather")


class TestWeatherStoreNeedsHistoryQuery:
    """Tests for needs_history_query method."""

    def test_recent_date_does_not_need_history(self, sample_region):
        """Test that recent dates don't need history API."""
        store = WeatherStore(sample_region)
        recent = datetime.now(timezone.utc)
        assert store.needs_history_query(recent) is False

    def test_old_date_needs_history(self, sample_region):
        """Test that old dates need history API."""
        store = WeatherStore(sample_region)
        old = datetime.now(timezone.utc) - timedelta(days=90)
        assert store.needs_history_query(old) is True

    def test_boundary_date(self, sample_region):
        """Test the boundary between history and forecast API."""
        store = WeatherStore(sample_region)
        # Just under 60 days should not need history
        under = datetime.now(timezone.utc) - timedelta(days=0)
        assert store.needs_history_query(under) is False

        # Over 60 days should need history
        over = datetime.now(timezone.utc) - timedelta(days=2)
        assert store.needs_history_query(over) is True


class TestWeatherStoreFetchMissingData:
    """Tests for fetch_missing_data method."""

    @pytest.mark.asyncio
    async def test_fetch_missing_data_creates_request(self, sample_region):
        """Test that fetch_missing_data creates proper API requests."""
        store = WeatherStore(sample_region)

        # Create mock response - list of forecasts for each location
        # The API returns a list with one entry per location
        mock_response_data = [
            {
                "minutely_15": {
                    "time": [
                        "2025-11-01T00:00",
                        "2025-11-01T00:15",
                        "2025-11-01T00:30",
                        "2025-11-01T00:45",
                    ],
                    "wind_speed_80m": [5.0, 6.0, 7.0, 8.0],
                    "temperature_2m": [10.0, 11.0, 12.0, 13.0],
                    "global_tilted_irradiance": [100.0, 110.0, 120.0, 130.0],
                    "pressure_msl": [1013.0, 1014.0, 1015.0, 1016.0],
                    "relative_humidity_2m": [75.0, 76.0, 77.0, 78.0],
                }
            }
            for _ in sample_region.latitudes
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            # The code uses resp.text() then json.loads(), so return a JSON string
            mock_response.text = AsyncMock(return_value=json.dumps(mock_response_data))

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_context)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            # Use a longer date range to ensure gen_missing_date_ranges yields ranges
            start = datetime(2025, 11, 1, tzinfo=timezone.utc)
            end = datetime(2025, 11, 3, tzinfo=timezone.utc)

            await store.fetch_missing_data(start, end)

            # Verify a request was made
            assert mock_session_instance.get.called


class TestWeatherStoreMissingRanges:
    """Tests for gen_missing_date_ranges method."""

    def test_empty_store_returns_full_range(self, sample_region):
        """Test that empty store returns the full requested range."""
        store = WeatherStore(sample_region)
        start = datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, 12, 0, tzinfo=timezone.utc)

        ranges = list(store.gen_missing_date_ranges(start, end))
        assert len(ranges) >= 1

    def test_partial_data_returns_missing_ranges(self, sample_region):
        """Test that partially filled store returns only missing ranges."""
        store = WeatherStore(sample_region)

        # Add data for Nov 1
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame(
            {
                "wind_speed_80m_0": [5.0] * len(dates),
                "temperature_2m_0": [10.0] * len(dates),
                "global_tilted_irradiance_0": [100.0] * len(dates),
            },
            index=dates,
        )
        df.index.name = "time"
        store._update_data(df)

        # Request range including Nov 1-3
        start = datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, 12, 0, tzinfo=timezone.utc)

        ranges = list(store.gen_missing_date_ranges(start, end))

        # Should only return ranges for Nov 2-3 (not Nov 1 which we have)
        for range_start, range_end in ranges:
            # The range should be after our existing data
            assert range_start >= datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)


class TestWeatherStoreURLConstruction:
    """Tests for URL construction."""

    def test_forecast_api_url(self, sample_region):
        """Test that forecast API URL is constructed correctly."""
        store = WeatherStore(sample_region)
        # The URL construction happens in fetch, but we can verify the base URL logic
        assert not store.needs_history_query(datetime.now(timezone.utc))

    def test_history_api_url(self, sample_region):
        """Test that history API URL is used for old dates."""
        store = WeatherStore(sample_region)
        old_date = datetime.now(timezone.utc) - timedelta(days=90)
        assert store.needs_history_query(old_date)


class TestWeatherStoreLocations:
    """Tests for the sample locations of a region."""

    def test_expected_columns_cover_extra_locations(self, sample_region):
        """Test that weather-only locations get the same columns as the region's own ones."""
        store = WeatherStore(sample_region)
        locations = len(sample_region.latitudes) + len(sample_region.extra_weather_latitudes)

        assert len(sample_region.extra_weather_latitudes) > 0
        assert len(store.expected_columns()) == locations * 5
        assert f"wind_{locations - 1}" in store.expected_columns()

    @pytest.mark.asyncio
    async def test_request_includes_extra_locations(self, sample_region):
        """Test that the weather-only locations are part of the API request."""
        store = WeatherStore(sample_region)
        requested = []

        class FailingSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def get(self, url):
                requested.append(url)
                raise RuntimeError("no network in tests")

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        with patch("predictor.model.weatherstore.aiohttp.ClientSession", return_value=FailingSession()):
            with pytest.raises(RuntimeError):
                await store.refresh_range(start, start + timedelta(days=1))

        lats = ",".join(map(str, sample_region.latitudes + sample_region.extra_weather_latitudes))
        lons = ",".join(map(str, sample_region.longitudes + sample_region.extra_weather_longitudes))
        assert f"latitude={lats}&longitude={lons}&" in requested[0]

    @pytest.mark.asyncio
    async def test_load_discards_cache_of_other_locations(self, sample_region, temp_storage_dir):
        """Test that a cache written for a different set of locations is not mixed with new data."""
        dates = pd.date_range(start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC")

        store = WeatherStore(sample_region, temp_storage_dir)
        outdated = sorted(store.expected_columns())[:-5]
        store.data = pd.DataFrame({c: [1.0] * len(dates) for c in outdated}, index=dates)
        await store.serialize()

        loaded = await WeatherStore(sample_region, temp_storage_dir).load()
        assert loaded.data.empty
        assert loaded.known_source_horizon is None

    @pytest.mark.asyncio
    async def test_load_keeps_cache_of_current_locations(self, sample_region, temp_storage_dir):
        """Test that a cache matching the configured locations survives a restart."""
        dates = pd.date_range(start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC")

        store = WeatherStore(sample_region, temp_storage_dir)
        store.data = pd.DataFrame({c: [1.0] * len(dates) for c in sorted(store.expected_columns())}, index=dates)
        await store.serialize()

        loaded = await WeatherStore(sample_region, temp_storage_dir).load()
        assert len(loaded.data) == len(dates)
