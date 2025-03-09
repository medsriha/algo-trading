from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import logging
from typing import Union, List, Optional
import pandas as pd
from algo_trading.data_providers import AlpacaDataProvider
from algo_trading.models import CrossoverConfig
import talib
from pathlib import Path
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class CrossoverEntry:
    """Class to manage market data operations and calculations."""

    DATA_DIR = Path("data")

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        crossover_config: Optional[CrossoverConfig] = CrossoverConfig(),
        cache_dir: Optional[str] = "market_data_cache",
    ):
        """Initialize the MarketDataManager.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca secret key
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.crossover_config = crossover_config
        self.api_key = api_key
        self.api_secret = api_secret

        self.fetcher = AlpacaDataProvider(api_key=api_key, secret_key=api_secret, cache_dir=self.cache_dir)

    def prepare_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate the data."""
        logger.info(f"Preparing {frame.shape[0]} rows of data")

        frame = frame.copy()
        # Convert timestamp to date only and remove timezone
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None).dt.normalize()

        # Validate data
        for col in ["timestamp", "close"]:
            if frame[col].isnull().any():
                raise ValueError(f"Dataset contains null {col} values")

        frame = frame.sort_values("timestamp").reset_index(drop=True)
        reduced = frame[["symbol", "timestamp", "close"]]
        return reduced

    @staticmethod
    def _identify_crossover_points(df: pd.DataFrame) -> pd.DataFrame:
        """Identify points where the lower SMA crosses under the upper SMA and add to DataFrame.
        Maintains True values throughout the bearish period until the upper SMA crosses back.
        Only marks periods as bearish if we've seen the actual crossover point.

        Args:
            df (pd.DataFrame): DataFrame with calculated SMAs

        Returns:
            pd.DataFrame: DataFrame with added crossunder_point column
        """
        logger.info("Identifying SMA crossover points")

        # Initialize crossunder column with False
        df["crossunder_point"] = False
        in_bearish_period = False
        saw_crossover = False

        # Handle edge case: if we start in bearish period without seeing crossover,
        # we keep everything False until we see an actual crossover
        if df["SMA_lower_below_upper"].iloc[0]:
            logger.debug("Started with bearish period but no crossover seen - waiting for first valid crossover")

        df["crossunder_point"] = False

        in_bearish_period = False
        saw_crossover = False

        for i in range(1, len(df)):
            prev_below = df["SMA_lower_below_upper"].iloc[i - 1]
            curr_below = df["SMA_lower_below_upper"].iloc[i]
            # Start of bearish period (False -> True means lower SMA crossed under upper SMA)
            if not prev_below and curr_below:
                in_bearish_period = True
                saw_crossover = True
                logger.debug(f"Started bearish period at {df['timestamp'].iloc[i]}")

            # End of bearish period (True -> False means upper SMA crossed back over)
            elif prev_below and not curr_below:
                in_bearish_period = False
                logger.debug(f"Ended bearish period at {df['timestamp'].iloc[i]}")
            # Set crossunder_point based on whether we're in a bearish period AND we've seen a crossover
            df.loc[df.index[i], "crossunder_point"] = in_bearish_period and saw_crossover

        bearish_periods = df["crossunder_point"].sum()
        logger.info(f"Found {bearish_periods} days in valid bearish periods")
        return df

    @staticmethod
    def _calculate_trend_from_crossover(frame: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend percentage change from crossover points until exit.
        Trend is calculated continuously from each crossunder point.

        Args:
            frame (pd.DataFrame): DataFrame with calculated SMAs and crossover points

        Returns:
            pd.DataFrame: DataFrame with added trend_from_crossover column
        """
        logger.info("Calculating trends from crossover points")

        # Initialize trend column with zeros
        frame["trend_from_crossover"] = 0.0
        current_crossover_price = None

        for i in range(1, len(frame)):  # Start from 1 to check previous values
            # If we hit a crossunder point, start a new trend calculation
            if frame["crossunder_point"].iloc[i] and not frame["SMA_lower_below_upper"].iloc[i - 1]:
                current_crossover_price = frame["close"].iloc[i]
                logger.debug(f"New crossunder point at index {i}, price: {current_crossover_price}")

            # We are inside a bearish period
            if current_crossover_price is not None:
                current_price = frame["close"].iloc[i]
                trend = ((current_price - current_crossover_price) / current_crossover_price) * 100
                frame.loc[frame.index[i], "trend_from_crossover"] = trend

        return frame

    def _calculate_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        logger.info("Calculating technical indicators")

        # Calculate SMAs
        frame["SMA_lower"] = frame["close"].rolling(window=self.crossover_config.lower_sma).mean()
        frame["SMA_upper"] = frame["close"].rolling(window=self.crossover_config.upper_sma).mean()
        frame["SMA_lower_below_upper"] = frame["SMA_lower"] < frame["SMA_upper"]

        # Calculate RSI
        frame["RSI"] = talib.RSI(frame["close"].values, timeperiod=self.crossover_config.rsi_period)
        frame["RSI_oversold"] = frame["RSI"] < self.crossover_config.rsi_oversold
        frame["RSI_overbought"] = frame["RSI"] > self.crossover_config.rsi_overbought
        frame["RSI_under_bought"] = frame["RSI"] < self.crossover_config.rsi_underbought

        # Add crossunder points
        frame = self._identify_crossover_points(frame)

        # Calculate trends from crossover points
        frame = self._calculate_trend_from_crossover(frame)

        frame = frame[frame["SMA_upper"].notna()].reset_index(drop=True)
        return frame

    def get_todays_close(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        """Get today's closing prices for given tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]

        try:
            joined_tickers = ",".join(tickers)
            url = f"https://data.alpaca.markets/v2/stocks/snapshots?symbols={joined_tickers}"

            headers = {
                "accept": "application/json",
                "APCA-API-KEY-ID": self.api_key or os.getenv("ALPACA_API_KEY"),
                "APCA-API-SECRET-KEY": self.api_secret or os.getenv("ALPACA_SECRET_KEY"),
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            rows = []
            for ticker, ticker_data in data.items():
                try:
                    close = ticker_data["dailyBar"]["c"]
                    # Convert to date only and remove timezone
                    timestamp = pd.to_datetime(ticker_data["dailyBar"]["t"]).date()

                    rows.append({"symbol": ticker, "timestamp": timestamp, "close": close})
                    logger.debug(f"Closing price for {ticker}: {close}")
                except (KeyError, TypeError) as e:
                    logger.error(f"Error processing data for {ticker}: {e}")

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[["symbol", "timestamp", "close"]]
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_historical_close(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        """Get historical close prices for SMA calculation."""
        try:
            # Only fetch 55 days (50 for max SMA + 5 extra days for better indicator calculation)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=self.crossover_config.upper_sma * 2)).strftime("%Y-%m-%d")

            df = self.fetcher.get_stock_data(
                symbols=tickers,
                timeframe="1D",
                start_date=start_date,
                end_date=end_date,
                columns=["symbol", "timestamp", "close"],  # Only request needed columns
            )

            return df[["symbol", "timestamp", "close"]]

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise

    def get_market_data(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        """Get both historical and current market data for given tickers."""
        try:
            hist_df = self.get_historical_close(tickers)

            if hist_df.empty:
                raise Exception("Historical data is empty")

            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.date

            current_df = self.get_todays_close(tickers)

            combined_df = pd.concat([hist_df, current_df], ignore_index=True)
            combined_df = self.prepare_data(combined_df)

            combined_df = combined_df.sort_values(["symbol", "timestamp"])
            combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

            combined_df = self._calculate_indicators(combined_df)

            return combined_df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def is_today_an_entry(self, frame: pd.DataFrame) -> dict[str, bool]:
        """Return true if today meets entry criteria based on the crossover strategy.

        The entry criteria are:
        1. Must be in a valid crossunder period (SMA_lower below SMA_upper)
        2. Must have seen the actual crossover point
        3. Must be in an uptrend from the crossover point
        4. RSI must be below the underbought threshold
        5. Must not be in an overbought RSI condition
        """
        results = {}
        for ticker, group in frame.groupby("symbol"):
            if len(group) < 2:  # Need at least 2 days of data
                return False

            group["entry"] = 0
            # Get last two rows
            yesterday = group.iloc[-2]
            today = group.iloc[-1]

            # Check all conditions for entry
            conditions = {
                "in_valid_crossunder": today["crossunder_point"],  # Must be in valid crossunder period
                "upward_trend": today["trend_from_crossover"] > 0,  # Must be in uptrend from crossover
                "rsi_below_threshold": today["RSI"] < self.crossover_config.rsi_underbought,  # RSI below threshold
                "not_overbought": not today["RSI_overbought"],  # Not in overbought condition
                "bearish_continuation": yesterday["SMA_lower_below_upper"]
                and today["SMA_lower_below_upper"],  # Still in bearish period
            }

            # Log conditions for debugging
            logger.debug(f"Entry conditions: {conditions}")

            is_entry = all(conditions.values())
            # Return True only if all conditions are met
            results[ticker] = is_entry
            group.iloc[-2]["entry"] = 1 if is_entry else 0
            os.makedirs(self.DATA_DIR / ticker / "today", exist_ok=True)

            group.to_csv(self.DATA_DIR / ticker / "today" / "crossover.csv")

        return results


if __name__ == "__main__":
    entry = CrossoverEntry()
    frame = entry.get_market_data(["AAPL"])
    print(entry.is_today_an_entry(frame))
