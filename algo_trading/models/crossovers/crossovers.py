from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import os
import sqlite3
from datetime import datetime
import pandas as pd
import talib

from algo_trading.data_providers import AlpacaDataProvider
from algo_trading.scripts.crossovers import CrossoversPlotter, CrossoversAnalyst
from algo_trading.database.crossovers.configs import DatabaseCrossoversConfig
from algo_trading.database.crossovers.connection import check_database_exists, init_db, get_db_connection
from algo_trading.models.crossovers import CrossoverConfig


logger = logging.getLogger(__name__)


class Crossover:
    # Class constants
    DATA_DIR = Path("data")
    PATTERN_TYPES = ("gain", "loss")

    def __init__(
        self, tickers: List[str], crossover_config: CrossoverConfig, database_config: DatabaseCrossoversConfig
    ) -> None:
        if not isinstance(tickers, list):
            raise TypeError("Tickers must be a list")

        self.tickers = tickers
        self.crossover_config = crossover_config
        self.output_path = self.DATA_DIR
        self.database_config = database_config
        logger.info(f"Initializing database for {database_config.table_name}")
        logger.info(f"Checking if database exists: {check_database_exists(database_config.table_name)}")
        logger.info(f"Plotter and analysis will be saved in the output path {self.output_path}")
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        if not check_database_exists(database_config.table_name):
            logger.info(f"Initializing database for {database_config.table_name}")
            init_db(database_config)
        else:
            logger.info(f"Database for {database_config.table_name} already exists")

        self._download_data()

    def _download_data(self) -> None:
        """Download and save data for all tickers."""
        fetcher = AlpacaDataProvider()
        for ticker in self.tickers:
            df = fetcher.get_stock_data(
                ticker, start_date=self.crossover_config.start_date, end_date=self.crossover_config.end_date
            )
            output_dir = self.output_path / ticker
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_dir / "dataframe.csv", index=False)

    def _prepare_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate the data."""
        logger.info(f"Preparing {frame.shape[0]} rows of data")

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])

        # Validate data
        for col in ["timestamp", "close"]:
            if frame[col].isnull().any():
                raise ValueError(f"Dataset contains null {col} values")

        frame = frame.sort_values("timestamp").reset_index(drop=True)

        frame = self._calculate_indicators(frame)

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

        frame = frame[frame["SMA_upper"].notna()].reset_index(drop=True)
        # Add crossunder points
        frame = self.identify_crossover_points(frame)

        # Calculate trends from crossover points
        frame = self.calculate_trend_from_crossover(frame)
        return frame

    def find(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Find and group crossover patterns."""
        logger.info("Finding crossover patterns")

        # Initialize pattern columns
        frame["gain"] = 0
        frame["loss"] = 0

        pattern_state = {"in_pattern": False, "start_index": None, "initial_price": None, "gain": 1, "loss": 1}

        for i in range(1, len(frame) - 1):
            current_price = frame.iloc[i]["close"]

            # Log current conditions
            conditions = {
                "previous_bearish": frame.iloc[i - 1]["SMA_lower_below_upper"],
                "current_bearish": frame.iloc[i]["SMA_lower_below_upper"],
                "crossunder": frame.iloc[i]["crossunder_point"],
                "uptrend": frame.iloc[i]["trend_from_crossover"] > 0,
                "rsi_overbought": frame.iloc[i]["RSI_under_bought"],
            }
            logger.debug(f"Index {i} conditions: {conditions}")

            # Check pattern entry conditions
            if (
                not pattern_state["in_pattern"]
                and conditions["previous_bearish"]
                and conditions["current_bearish"]
                and conditions["crossunder"]
                and conditions["uptrend"]
                and conditions["rsi_overbought"]
            ):
                logger.info(f"New pattern started at index {i} with price {current_price}")
                pattern_state.update({"in_pattern": True, "start_index": i, "initial_price": current_price})

                # Initialize pattern in both columns
                pattern_slice = slice(pattern_state["start_index"], i + 1)
                frame.loc[frame.index[pattern_slice], "gain"] = pattern_state["gain"]
                frame.loc[frame.index[pattern_slice], "loss"] = pattern_state["loss"]
                continue

            elif pattern_state["in_pattern"]:
                # Check pattern exit conditions
                pattern_days = i - pattern_state["start_index"] + 1
                price_change = (current_price - pattern_state["initial_price"]) / pattern_state["initial_price"]

                # Update current pattern in both columns
                pattern_slice = slice(pattern_state["start_index"], i + 1)
                frame.loc[frame.index[pattern_slice], "gain"] = pattern_state["gain"]
                frame.loc[frame.index[pattern_slice], "loss"] = pattern_state["loss"]

                logger.debug(f"Pattern day {pattern_days}, price change: {price_change:.2%}")

                # Sell if we have reached the crossover length, RSI is overbought, or price change is greater than the take profit or stop loss
                if (
                    pattern_days == self.crossover_config.crossover_length
                    or frame.iloc[i]["RSI_overbought"]
                    or abs(price_change) >= max(self.crossover_config.take_profit, self.crossover_config.stop_loss)
                ):
                    # Determine final pattern type and increment counter
                    pattern_type = "gain" if price_change > 0 else "loss"
                    pattern_state[pattern_type] += 1

                    logger.info(
                        f"Pattern completed at index {i}: type={pattern_type}, "
                        f"days={pattern_days}, price_change={price_change:.2%}"
                    )

                    # Clear the non-winning pattern type
                    non_winning_type = "loss" if pattern_type == "gain" else "gain"
                    frame.loc[frame.index[pattern_slice], non_winning_type] = 0

                    pattern_state.update({"in_pattern": False, "start_index": None, "initial_price": None})
            else:
                pass

        return frame

    def store_results(self, metrics: Dict[str, Any]) -> bool:
        """Store analysis results in the database.

        Args:
            ticker: Stock ticker symbol
            metrics: Dictionary containing analysis metrics
            all_uptrend: Boolean indicating if all bearish periods are in uptrend

        Returns:
            bool: True if storage successful, False otherwise
        """
        try:
            with get_db_connection(self.database_config) as conn:
                cursor = conn.cursor()

                # Prepare the SQL statement
                sql = f"""
                    INSERT INTO {self.database_config.table_name}
                    (data_creation_date, timestamp_date_start, timestamp_date_end, 
                    data_source, ticker, total_trades, total_gains, total_losses, 
                    all_bearish_uptrend, combined_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Prepare the data tuple
                data = (
                    metrics["data_creation_date"],
                    metrics["timestamp_date_start"],
                    metrics["timestamp_date_end"],
                    metrics["data_source"],
                    metrics["ticker"],
                    metrics["total_trades"],
                    metrics["total_gains"],
                    metrics["total_losses"],
                    metrics["all_bearish_uptrend"],
                    metrics["combined_return"],
                )

                cursor.execute(sql, data)
                conn.commit()

                logger.info(f"Successfully stored results for {metrics['ticker']}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Database error while storing results for {metrics['ticker']}: {e}")
            return False

    def run(self) -> bool:
        """Run the crossover pattern analysis for all tickers."""
        try:
            for ticker in self.tickers:
                logger.info(f"Processing {ticker}")

                # Load and process data
                df = pd.read_csv(self.output_path / ticker / "dataframe.csv")
                df = self._prepare_data(df)
                df = self.find(df)

                # Ensure the ticker output directory exists
                ticker_output_path = self.output_path / ticker
                os.makedirs(ticker_output_path, exist_ok=True)

                # Save results
                df.to_csv(ticker_output_path / "crossovers.csv", index=False)
                logger.info(f"Saved crossovers data to {ticker_output_path / 'crossovers.csv'}")

                total_gains = self.total_gains(df)
                total_losses = self.total_losses(df)
                # Calculate metrics
                bearish_periods, all_uptrend = self.calculate_bearish_periods(df)
                results = {
                    "data_creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_date_start": df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_date_end": df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    "data_source": "Alpaca",
                    "ticker": ticker,
                    "total_trades": total_gains + total_losses,
                    "total_gains": total_gains,
                    "total_losses": total_losses,
                    "all_bearish_uptrend": all_uptrend,
                    "combined_return": self.combined_return(df),
                }
                # Store results in database
                if not self.store_results(results):
                    logger.error(f"Failed to store results for {ticker}")

                # Generate visualizations and report
                self._generate_outputs(df, ticker, bearish_periods, results)

            return True

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return False

    def _start_bearish_period(self, start_idx: int, df: pd.DataFrame) -> Dict[str, Any]:
        """Initialize a new bearish period at the given index."""
        logger.debug(f"Starting new bearish period at index {start_idx}")

        return {
            "start_idx": start_idx,
            "start_price": float(df["close"].iloc[start_idx]),
            "start_date": df["timestamp"].iloc[start_idx],
            "lowest_price": float(df["close"].iloc[start_idx]),
            "lowest_idx": start_idx,
        }

    def _end_bearish_period(self, df: pd.DataFrame, period: Dict[str, Any], end_idx: int) -> Dict[str, Any]:
        """Calculate metrics for a completed bearish period."""
        logger.debug(f"Ending bearish period from index {period['start_idx']} to {end_idx}")

        # Ensure we're using the correct range of data
        period_slice = slice(period["start_idx"], end_idx + 1)
        period_data = df.iloc[period_slice]

        # Find the lowest price in the period
        lowest_idx = period_data["close"].idxmin()
        lowest_price = float(df["close"].iloc[lowest_idx])

        end_price = float(df["close"].iloc[end_idx])
        start_price = period["start_price"]

        # Calculate period metrics
        period_length = end_idx - period["start_idx"] + 1  # Include both start and end days
        price_change = ((end_price - start_price) / start_price) * 100
        max_drawdown = ((lowest_price - start_price) / start_price) * 100
        recovery = ((end_price - lowest_price) / lowest_price) * 100 if lowest_price != end_price else 0

        return {
            **period,
            "end_idx": end_idx,
            "end_date": df["timestamp"].iloc[end_idx],
            "end_price": end_price,
            "lowest_price": lowest_price,
            "lowest_idx": lowest_idx,
            "length": period_length,
            "price_change_pct": price_change,
            "max_drawdown_pct": max_drawdown,
            "recovery_pct": recovery,
            "is_uptrend": end_price > start_price,
        }

    def _handle_final_period(self, df: pd.DataFrame, period: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a bearish period that extends to the end of the data."""
        logger.debug("Handling final unclosed bearish period")
        return self._end_bearish_period(df, period, len(df) - 1)

    def calculate_bearish_periods(self, frame: pd.DataFrame) -> Tuple[List[Dict[str, Any]], bool]:
        """Calculate the length and details of each bearish period where patterns were found.
        A bearish period is when SMA_lower_below_upper is True and contains at least one pattern."""
        logger.info("Calculating bearish periods with patterns")
        bearish_periods = []
        current_period = None
        all_uptrend = True

        for i in range(len(frame)):
            is_bearish = frame["SMA_lower_below_upper"].iloc[i]
            has_pattern = frame.iloc[i]["gain"] > 0 or frame.iloc[i]["loss"] > 0

            # Start a new bearish period
            if is_bearish and current_period is None and has_pattern:
                current_period = self._start_bearish_period(i, frame)
                logger.debug(f"Started new bearish period with pattern at index {i}")

            # End current bearish period
            elif (not is_bearish or not has_pattern) and current_period is not None:
                period_data = self._end_bearish_period(frame, current_period, i - 1)  # End at previous index
                all_uptrend &= period_data["is_uptrend"]
                bearish_periods.append(period_data)
                current_period = None
                logger.debug(f"Ended bearish period at index {i - 1}")

            # Update lowest price if in current period
            elif current_period is not None:
                current_price = float(frame["close"].iloc[i])
                if current_price < current_period["lowest_price"]:
                    current_period["lowest_price"] = current_price
                    current_period["lowest_idx"] = i

        # Handle any ongoing bearish period at the end of the data
        if current_period is not None:
            period_data = self._handle_final_period(frame, current_period)
            all_uptrend &= period_data["is_uptrend"]
            bearish_periods.append(period_data)

        logger.info(f"Found {len(bearish_periods)} bearish periods containing patterns")

        # Add pattern information to each period
        for period in bearish_periods:
            period_slice = slice(period["start_idx"], period["end_idx"] + 1)
            period_data = frame.iloc[period_slice]
            period["gain_patterns"] = period_data["gain"].nunique() - 1  # Subtract 1 to exclude 0
            period["loss_patterns"] = period_data["loss"].nunique() - 1  # Subtract 1 to exclude 0
            period["total_patterns"] = period["gain_patterns"] + period["loss_patterns"]
            logger.debug(
                f"Period from {period['start_date']} to {period['end_date']}: "
                f"{period['total_patterns']} patterns ({period['gain_patterns']} gains, "
                f"{period['loss_patterns']} losses)"
            )

        return bearish_periods, all_uptrend

    def combined_return(self, df: pd.DataFrame) -> float:
        """Calculate the combined return from both gain and loss patterns."""
        gain_returns = self._calculate_pattern_returns(df, "gain")
        loss_returns = self._calculate_pattern_returns(df, "loss")

        total_gain_return = sum(gain_returns) if gain_returns else 0
        total_loss_return = sum(loss_returns) if loss_returns else 0

        return total_gain_return + total_loss_return

    def _calculate_pattern_returns(self, df: pd.DataFrame, pattern_type: str) -> List[float]:
        """Calculate returns for a specific pattern type.

        Args:
            pattern_type (str): Type of pattern ('gain' or 'loss')

        Returns:
            List[float]: List of percentage returns for each pattern
        """
        returns = []
        # Get unique pattern IDs excluding 0
        pattern_ids = sorted(df[df[pattern_type] > 0][pattern_type].unique())

        for pattern_id in pattern_ids:
            pattern_data = df[df[pattern_type] == pattern_id]
            start_price = pattern_data["close"].iloc[0]
            end_price = pattern_data["close"].iloc[-1]
            return_pct = ((end_price - start_price) / start_price) * 100
            returns.append(return_pct)

        return returns

    @staticmethod
    def total_gains(df: pd.DataFrame) -> int:
        return df["gain"].nunique() - 1  # Subtract 1 to exclude 0

    @staticmethod
    def total_losses(df: pd.DataFrame) -> int:
        return df["loss"].nunique() - 1  # Subtract 1 to exclude 0

    def _calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMAs on daily data."""
        logger.info(f"[crossover] Calculating SMAs")
        df["SMA_lower"] = df["close"].rolling(window=self.crossover_config.lower_sma).mean()
        df["SMA_upper"] = df["close"].rolling(window=self.crossover_config.upper_sma).mean()
        df["SMA_lower_below_upper"] = df["SMA_lower"] < df["SMA_upper"]

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI) using ta-lib.

        :param period: The period over which to calculate RSI (default: 14)
        """

        logger.info(f"[crossover] Calculating RSI with period {self.crossover_config.rsi_period}")

        try:
            # Calculate RSI using ta-lib
            df["RSI"] = talib.RSI(df["close"].values, timeperiod=self.crossover_config.rsi_period)

            # Add RSI conditions
            df["RSI_oversold"] = df["RSI"] < self.crossover_config.rsi_oversold
            df["RSI_overbought"] = df["RSI"] > self.crossover_config.rsi_overbought
            df["RSI_under_bought"] = df["RSI"] < self.crossover_config.rsi_underbought

            logger.debug(f"[crossover] RSI calculation completed")

            return df

        except Exception as e:
            logger.error(f"[crossover] Error calculating RSI: {str(e)}")
            raise

    def _generate_outputs(
        self, df: pd.DataFrame, ticker: str, bearish_periods: List[Dict[str, Any]], metrics: Dict[str, float]
    ) -> None:
        """Generate visualizations and report for the given ticker."""
        # Ensure the output directory exists
        ticker_output_path = self.output_path / ticker
        os.makedirs(ticker_output_path, exist_ok=True)

        plotter = CrossoversPlotter(
            df=df,
            ticker=ticker,
            crossover_config=self.crossover_config,
            output_path=ticker_output_path,
        )
        plotter.save_plot(bearish_periods=bearish_periods)

        report_writer = CrossoversAnalyst(
            frame=df,
            ticker=ticker,
            crossover_config=self.crossover_config,
            output_path=ticker_output_path,
        )
        report_writer.write(
            total_gains=metrics["total_gains"],
            total_losses=metrics["total_losses"],
            bearish_periods=bearish_periods,
        )

    def identify_crossover_points(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate_trend_from_crossover(self, frame: pd.DataFrame) -> pd.DataFrame:
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
