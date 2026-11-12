import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import override

import aiohttp
from aiohttp import ClientTimeout
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

THE_API_URL = "https://api.tradinghub.eu/api/dataexport/xmlexport"
# THE asks every automated client to identify itself
THE_USER_AGENT = "EpexPredictor (+https://github.com/b3nn0/EpexPredictor)"
# A gas day runs from 06:00 to 06:00 German local time
GASDAY_START_HOUR = 6


def _parse_xml(xml: bytes) -> ET.Element:
    # the export never contains a DTD - refuse anything that does, instead of letting the parser expand entities
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ValueError("unexpected DTD in THE data export")
    return ET.fromstring(xml)


def _to_series(prices: dict[pd.Timestamp, float]) -> pd.Series:
    index = pd.DatetimeIndex(list(prices.keys()), tz="UTC")
    return pd.Series(list(prices.values()), index=index, dtype=float).sort_index()


def parse_imbalance_prices(xml: bytes) -> pd.Series:
    """
    Parses a "PricesEnergyImbalance" export.
    Returns the volume-weighted average price at the THE virtual trading point (EUR/MWh), indexed by gas day
    """
    prices = {}
    for entry in _parse_xml(xml):
        gasday = entry.findtext("Gasday")
        price = entry.findtext("VTPDailyAveragePrice")
        if gasday and price:
            prices[pd.Timestamp(gasday[:10])] = float(price)
    return _to_series(prices)


def parse_preliminary_prices(xml: bytes) -> pd.Series:
    """
    Parses a "PricesEnergyImbalancePreliminary" export, which carries the gas day index of yesterday, of the running
    gas day and (once trading for it started) of tomorrow. Same unit and index as parse_imbalance_prices.
    """
    prices = {}
    for entry in _parse_xml(xml):
        for day in ("Yesterday", "Today", "Tomorrow"):
            price = entry.findtext(f"GasdayIndexPriceEEX{day}")
            date = entry.findtext(f"GasdayIndexPriceEEX{day}Date")
            if not price or not date:
                continue
            # timestamps are German local time. Values that are not available yet come as -1 / year 1900
            ts = pd.Timestamp(date)
            if float(price) <= 0 or ts.year < 2000:
                continue
            gasday = (ts - pd.Timedelta(hours=GASDAY_START_HOUR)).normalize()
            prices[gasday] = float(price)
    return _to_series(prices)


class GasPriceStore(DataStore):
    """
    Fetches and caches natural gas prices from the open data export of Trading Hub Europe, the German gas market
    area manager: the average price at the virtual trading point per gas day.
    Only German gas prices supported for now, but should serve as a rough indication for other markets, too.
    https://api.tradinghub.eu/api/dataexport/manual/en
    """

    # Preliminary values are replaced by final ones a day or two later, so always re-request the last few days
    REFRESH_DAYS = 7

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None

    update_lock: asyncio.Lock


    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "gasprices_the_v1")
        self.update_lock = asyncio.Lock()


    async def _fetch(self, session: aiohttp.ClientSession, report: str, params: dict[str, str] | None = None) -> bytes:
        url = f"{THE_API_URL}/{report}"
        log.info(f"{self.region.bidding_zone_entsoe}: Fetching natural gas price data: {url} {params or ''}")
        async with session.get(url, params=params, headers={"User-Agent": THE_USER_AGENT}, timeout=ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            return await resp.read()


    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            if not self.region.use_de_nat_gas_price:
                return False

            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            missing = self.gen_missing_date_ranges(start, end)
            if len(missing) == 0:
                return False

            qstart = min(rstart for rstart, _ in missing) - timedelta(days=self.REFRESH_DAYS)
            qend = max(rend for _, rend in missing)

            updated = False
            try:
                async with aiohttp.ClientSession() as session:
                    final = parse_imbalance_prices(await self._fetch(session, "PricesEnergyImbalance", {
                        "startDate": qstart.strftime("%Y-%m-%d"),
                        "endDate": qend.strftime("%Y-%m-%d"),
                    }))
                    preliminary = parse_preliminary_prices(await self._fetch(session, "PricesEnergyImbalancePreliminary"))

                # final prices win, preliminary ones only cover the newest gas days
                prices = final.combine_first(preliminary)
                if len(prices) > 0:
                    df = prices.to_frame("gasprice")
                    df.index.name = "time"
                    df = df.resample('15min').ffill()
                    updated = self._update_data(df)
            except Exception as e:
                # A few days old gas prices are still good enough, so this is not fatal
                log.warning(f"{self.region.bidding_zone_entsoe}: failed to update gas prices: {e}")

            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: gas price data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()

            return updated


    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=12)
