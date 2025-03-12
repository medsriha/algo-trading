from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
import os
import dotenv
import datetime
import pandas as pd
from typing import Optional, Union, List
import logging
from pathlib import Path
import json

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class AlpacaDataProvider:
    """A class to fetch historical stock data from Alpaca API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the StockDataFetcher with API credentials.

        :param api_key: Alpaca API key. If None, will look for API_KEY in environment
        :param secret_key: Alpaca secret key. If None, will look for SECRET_KEY in environment
        :param cache_dir: Directory to store cached data. If None, defaults to ./data/cache
        """
        logger.info("Initializing StockDataFetcher")

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            logger.error("API credentials missing")
            raise ValueError("API credentials not found. Please provide them or set in environment.")

        logger.debug("Creating Alpaca client")
        self.client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)

        # Convert cache_dir to Path object if provided as string
        if cache_dir is None:
            self.cache_dir = Path.cwd() / "data" / "cache"
        else:
            self.cache_dir = Path.cwd() / "data" / cache_dir if isinstance(cache_dir, str) else cache_dir

        logger.info(f"Setting up cache directory at {self.cache_dir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        """Get the cache directory path for a symbol."""
        symbol_dir = self.cache_dir / symbol.lower()
        logger.debug(f"Creating symbol directory at {symbol_dir}")
        symbol_dir.mkdir(exist_ok=True)
        cache_path = symbol_dir / "historical_data.parquet"
        logger.debug(f"Cache path: {cache_path}")
        return cache_path

    def _get_metadata_path(self, symbol: str) -> Path:
        """Get the metadata file path for a symbol."""
        symbol_dir = self.cache_dir / symbol.lower()
        metadata_path = symbol_dir / "metadata.json"
        logger.debug(f"Metadata path: {metadata_path}")
        return metadata_path

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> None:
        """Save data and metadata to cache."""
        logger.info(f"Saving data to cache for {symbol}")
        cache_path = self._get_cache_path(symbol)
        metadata_path = self._get_metadata_path(symbol)

        try:
            logger.debug(f"Saving DataFrame with shape {df.shape} to {cache_path}")
            df.to_parquet(cache_path)

            metadata = {
                "start_date": start_date,
                "end_date": end_date,
                "last_updated": datetime.datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns),
            }

            logger.debug(f"Saving metadata to {metadata_path}: {metadata}")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully cached data for {symbol}")
        except Exception as e:
            logger.error(f"Error saving cache for {symbol}: {str(e)}")
            raise

    def _check_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Check if valid cached data exists."""
        logger.info(f"Checking cache for {symbol}")
        cache_path = self._get_cache_path(symbol)
        metadata_path = self._get_metadata_path(symbol)

        if not cache_path.exists():
            logger.debug(f"Cache file not found at {cache_path}")
            return None
        if not metadata_path.exists():
            logger.debug(f"Metadata file not found at {metadata_path}")
            return None

        try:
            logger.debug(f"Reading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            logger.debug(f"Cached metadata: {metadata}")
            logger.debug(f"Requested date range: {start_date} to {end_date}")

            # Check if cached data matches requested date range
            if metadata["start_date"] == start_date and metadata["end_date"] == end_date:
                logger.info(f"Cache hit for {symbol}! Loading data from {cache_path}")
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded DataFrame with shape {df.shape}")
                return df

            logger.debug("Cache miss: date range mismatch")
            return None
        except Exception as e:
            logger.warning(f"Error reading cache for {symbol}: {e}", exc_info=True)
            return None

    def get_stock_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = "1D",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for given symbol(s).

        :param symbols: Single stock symbol or list of symbols
        :param timeframe: Time interval ('1H' for hourly, '1D' for daily)
        :param start_date: Start date in 'YYYY-MM-DD' format. Defaults to start of current year
        :param end_date: End date in 'YYYY-MM-DD' format. Defaults to today

        Returns:
            pandas DataFrame with historical data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            logger.debug(f"Converted single symbol to list: {symbols}")

        # Set default dates if not provided
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            logger.debug(f"Using default start_date: {start_date}")

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            logger.debug(f"Using default end_date: {end_date}")

        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")

        timeframe_map = {"1H": TimeFrame.Hour, "1D": TimeFrame.Day}
        if timeframe not in timeframe_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            raise ValueError(f"Invalid timeframe. Must be one of: {list(timeframe_map.keys())}")

        try:
            # For single symbol, check cache first
            if len(symbols) == 1:
                logger.debug("Single symbol request, checking cache")
                cached_data = self._check_cache(symbols[0], start_date, end_date)
                if cached_data is not None:
                    return cached_data
                logger.debug("No valid cache found, fetching from API")

            logger.debug("Preparing API request parameters")
            params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_map[timeframe],
                start=start_date,
                end=end_date,
                adjustment="all",
                columns=columns,
            )

            logger.info("Sending request to Alpaca API")
            bars = self.client.get_stock_bars(params)
            df = bars.df.reset_index()
            logger.info(f"Received data with shape {df.shape}")

            # Cache the data for single symbol requests
            if len(symbols) == 1:
                logger.debug("Single symbol request, caching results")
                self._save_to_cache(df, symbols[0], start_date, end_date)

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}", exc_info=True)
            raise Exception(f"Error fetching data: {str(e)}")
