"""Tests for predictor.model.gaspricestore module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from predictor.model.gaspricestore import GasPriceStore, parse_imbalance_prices, parse_preliminary_prices
from predictor.model.priceregion import PriceRegionName

# shaped like the answers of https://api.tradinghub.eu/api/dataexport/xmlexport (incl. the leading BOM)
IMBALANCE_XML = b"""\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>
<PricesEnergyImbalance xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <PricesEnergyImbalance>
    <Gasday>2026-11-11</Gasday>
    <PositiveEnergyImbalancePrice>84.324</PositiveEnergyImbalancePrice>
    <NegativeEnergyImbalancePrice>89.2</NegativeEnergyImbalancePrice>
    <VTPDailyAveragePrice>86.045</VTPDailyAveragePrice>
    <Unit>EUR/MWh</Unit>
  </PricesEnergyImbalance>
  <PricesEnergyImbalance>
    <Gasday>2026-11-10</Gasday>
    <PositiveEnergyImbalancePrice>81.225</PositiveEnergyImbalancePrice>
    <NegativeEnergyImbalancePrice>88.88</NegativeEnergyImbalancePrice>
    <VTPDailyAveragePrice>85.595</VTPDailyAveragePrice>
    <Unit>EUR/MWh</Unit>
  </PricesEnergyImbalance>
</PricesEnergyImbalance>"""

PRELIMINARY_XML = b"""\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>
<PreliminaryImbalancePrices xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <PreliminaryImbalancePrices>
    <PreisSystemBuyTomorrowDate>1900-01-01T00:00:00</PreisSystemBuyTomorrowDate>
    <PreisSystemBuyTomorrow>-1</PreisSystemBuyTomorrow>
    <GasdayIndexPriceEEXYesterdayDate>2026-11-12T05:00:00</GasdayIndexPriceEEXYesterdayDate>
    <GasdayIndexPriceEEXYesterday>86.5</GasdayIndexPriceEEXYesterday>
    <GasdayIndexPriceEEXTodayDate>2026-11-12T16:00:00</GasdayIndexPriceEEXTodayDate>
    <GasdayIndexPriceEEXToday>88.43</GasdayIndexPriceEEXToday>
    <GasdayIndexPriceEEXTomorrowDate>2026-11-13T06:00:00</GasdayIndexPriceEEXTomorrowDate>
    <GasdayIndexPriceEEXTomorrow>92.05</GasdayIndexPriceEEXTomorrow>
  </PreliminaryImbalancePrices>
</PreliminaryImbalancePrices>"""


def day(date: str) -> pd.Timestamp:
    return pd.Timestamp(date, tz="UTC")


class TestParseImbalancePrices:
    """Tests for parse_imbalance_prices."""

    def test_parses_daily_average_per_gasday(self):
        """Test that the average price is returned per gas day, sorted by date."""
        prices = parse_imbalance_prices(IMBALANCE_XML)

        assert list(prices.index) == [day("2026-11-10"), day("2026-11-11")]
        assert list(prices) == [85.595, 86.045]

    def test_empty_export(self):
        """Test a period without entries, which is answered with an empty root element."""
        prices = parse_imbalance_prices(b'<?xml version="1.0" encoding="utf-8"?><PricesEnergyImbalance />')

        assert len(prices) == 0
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_rejects_documents_with_dtd(self):
        """Test that entity definitions are refused instead of expanded."""
        with pytest.raises(ValueError):
            parse_imbalance_prices(b'<!DOCTYPE x [<!ENTITY a "b">]><PricesEnergyImbalance />')


class TestParsePreliminaryPrices:
    """Tests for parse_preliminary_prices."""

    def test_maps_timestamps_to_gasdays(self):
        """Test that yesterday (which ends at 06:00), today and tomorrow end up on the right gas day."""
        prices = parse_preliminary_prices(PRELIMINARY_XML)

        assert list(prices.index) == [day("2026-11-11"), day("2026-11-12"), day("2026-11-13")]
        assert list(prices) == [86.5, 88.43, 92.05]

    def test_skips_values_not_available_yet(self):
        """Test that the placeholder for a missing value is ignored."""
        xml = PRELIMINARY_XML.replace(b"2026-11-13T06:00:00", b"1900-01-01T00:00:00").replace(b"92.05", b"-1")
        prices = parse_preliminary_prices(xml)

        assert list(prices.index) == [day("2026-11-11"), day("2026-11-12")]


class TestGasPriceStoreFetchMissingData:
    """Tests for fetch_missing_data method."""

    @staticmethod
    def _store(region=PriceRegionName.DE):
        store = GasPriceStore(region.to_region())
        store._fetch = AsyncMock(side_effect=lambda session, report, params=None: (
            PRELIMINARY_XML if report == "PricesEnergyImbalancePreliminary" else IMBALANCE_XML
        ))
        return store

    @pytest.mark.asyncio
    async def test_combines_final_and_preliminary_prices(self):
        """Test that final prices win and preliminary ones extend the series."""
        store = self._store()

        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        end = datetime(2026, 11, 14, tzinfo=timezone.utc)
        assert await store.fetch_missing_data(start, end) is True

        # 15 minute resolution, forward filled within the known days
        assert store.data.index[0] == day("2026-11-10")
        assert store.data.index[-1] == day("2026-11-13")
        assert store.data.loc[day("2026-11-10") + pd.Timedelta(hours=12), "gasprice"] == 85.595
        assert store.data.loc[day("2026-11-11"), "gasprice"] == 86.045  # final, not the preliminary 86.5
        assert store.data.loc[day("2026-11-12"), "gasprice"] == 88.43
        assert store.data.loc[day("2026-11-13"), "gasprice"] == 92.05

    @pytest.mark.asyncio
    async def test_requests_missing_range_plus_refresh_margin(self):
        """Test that already known days are re-requested, so that preliminary values get replaced."""
        store = self._store()

        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        end = datetime(2026, 11, 14, tzinfo=timezone.utc)
        await store.fetch_missing_data(start, end)

        report, params = store._fetch.call_args_list[0].args[1:]
        assert report == "PricesEnergyImbalance"
        assert params == {"startDate": "2026-11-03", "endDate": "2026-11-14"}

    @pytest.mark.asyncio
    async def test_nothing_missing_does_not_fetch(self):
        """Test that a covered range causes no request."""
        store = self._store()
        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        await store.fetch_missing_data(start, datetime(2026, 11, 14, tzinfo=timezone.utc))
        store._fetch.reset_mock()

        assert await store.fetch_missing_data(start, datetime(2026, 11, 12, tzinfo=timezone.utc)) is False
        assert not store._fetch.called

    @pytest.mark.asyncio
    async def test_failure_keeps_existing_data(self):
        """Test that an unreachable API is not fatal."""
        store = self._store()
        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        await store.fetch_missing_data(start, datetime(2026, 11, 13, tzinfo=timezone.utc))
        known = store.data.copy()

        store._fetch = AsyncMock(side_effect=RuntimeError("503"))
        assert await store.fetch_missing_data(start, datetime(2026, 11, 20, tzinfo=timezone.utc)) is False
        pd.testing.assert_frame_equal(store.data, known)

    @pytest.mark.asyncio
    async def test_region_without_gas_price_does_not_fetch(self):
        """Test that regions which do not use the gas price never query the API."""
        store = self._store(PriceRegionName.FR)

        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        end = datetime(2026, 11, 14, tzinfo=timezone.utc)
        assert await store.fetch_missing_data(start, end) is False
        assert not store._fetch.called
