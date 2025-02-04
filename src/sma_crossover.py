import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
from get_data import StockDataFetcher
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class CrossoverConfig:
    """Configuration for SMA Crossover Analysis."""

    upper_sma: int
    lower_sma: int
    stop_loss: float
    take_profit: float
    pattern_length: int
    output_dir: str = "../data"


class SMACrossoverAnalyzer:
    """Analyzer for SMA crossover patterns in stock data.

    This class analyzes stock price data to identify crossover patterns between
    two Simple Moving Averages (SMA) of different periods.
    """

    def __init__(
        self,
        ticker: str,
        dataframe: pd.DataFrame,
        config: Optional[CrossoverConfig] = None,
    ):
        """Initialize the analyzer with data and configuration.

        Args:
            ticker: Stock ticker symbol
            dataframe: DataFrame containing stock price data
            config: Configuration object for analysis parameters
            logger: Optional logger instance

        Raises:
            ValueError: If required data columns are missing
            TypeError: If input types are incorrect
        """
        if not isinstance(ticker, str):
            raise TypeError("Ticker must be a string")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dataframe must be a pandas DataFrame")

        required_columns = {"timestamp", "close"}
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"Dataframe must contain columns: {required_columns}")

        self.ticker = ticker
        self.config = config or CrossoverConfig(
            upper_sma=50,
            lower_sma=20,
            stop_loss=0.05,
            take_profit=0.1,
            pattern_length=10,
        )

        # Setup output path
        self.output_path = Path(self.config.output_dir) / self.ticker
        os.makedirs(self.output_path, exist_ok=True)

        # Read and validate data
        self._read_data(dataframe)

    def remove_directory(self):
        """Remove the output directory."""
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

    def _read_data(self, dataframe: pd.DataFrame) -> None:
        """Load and prepare the daily data.

        Args:
            dataframe: DataFrame containing stock price data

        Raises:
            ValueError: If data validation fails
        """
        logger.info(
            f"[crossover][{self.ticker}] Reading {dataframe.shape[0]} rows of data"
        )
        try:
            # Make a copy to avoid modifying the original
            self.df = dataframe.copy()
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

            # Validate data
            if self.df["timestamp"].isnull().any():
                raise ValueError("Dataset contains null timestamps")
            if self.df["close"].isnull().any():
                raise ValueError("Dataset contains null close prices")

            # Sort by timestamp
            self.df.sort_values("timestamp", inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.cleanup()
            raise

    def get_dataframe(self):
        return self.df

    def _calculate_sma(self):
        """Calculate SMAs on daily data."""
        logger.info(f"[crossover][{self.ticker}] Calculating SMAs")
        self.df["SMA_lower"] = (
            self.df["close"].rolling(window=self.config.lower_sma).mean()
        )
        self.df["SMA_upper"] = (
            self.df["close"].rolling(window=self.config.upper_sma).mean()
        )
        self.df["SMA_lower_below_upper"] = self.df["SMA_lower"] < self.df["SMA_upper"]

    def _calculate_price_trends(
        self, initial_window: int = 5, lookback_period: int = 20, trend_threshold: float = 0.001
    ) -> None:
        """Calculate the trend direction for price movement with dynamic window size based on historical data.

        For each point, calculates the window size based on the average length of upward trends
        in the previous lookback_period days.

        Args:
            initial_window: Initial number of periods to use for trend calculation (default: 5)
            lookback_period: Number of previous days to analyze for trend lengths (default: 20)
        """
        logger.info(
            f"[crossover][{self.ticker}] Calculating price trends with dynamic windows"
        )

        # Initialize columns
        self.df["price_trend"] = 0.0
        self.df["price_trend_direction"] = 0
        self.df["window_size"] = initial_window
        self.df["upward_trend_days"] = 0  # New column for consecutive upward days

        # First pass with initial window to get basic trend direction
        initial_trends = (
            self.df["close"]
            .rolling(window=initial_window)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            .fillna(0)
        )

        initial_directions = np.where(
            initial_trends > trend_threshold,
            1,
            np.where(initial_trends < -trend_threshold, -1, 0),
        )

        # For each point, calculate the dynamic window based on historical data
        for i in range(lookback_period, self.df.shape[0]):
            # Get historical slice
            historical_slice = initial_directions[max(0, i - lookback_period) : i]

            # Calculate lengths of upward trends in historical data
            trend_lengths = []
            current_length = 0

            for direction in historical_slice:
                if direction == 1:  # Upward trend
                    current_length += 1
                elif current_length > 0:  # End of upward trend
                    trend_lengths.append(current_length)
                    current_length = 0

            # Add the last trend length if we ended in an upward trend
            if current_length > 0:
                trend_lengths.append(current_length)

            # Calculate dynamic window size based on average upward trend length
            if trend_lengths:
                dynamic_window = int(np.mean(trend_lengths))
                # Ensure window is at least the initial size and not too large
                dynamic_window = max(initial_window, min(dynamic_window, 20))
            else:
                dynamic_window = initial_window

            # Store the window size
            self.df.loc[i, "window_size"] = dynamic_window

            # Calculate trend using the dynamic window
            if i >= dynamic_window:
                try:
                    window_data = (
                        self.df["close"].iloc[i - dynamic_window + 1 : i + 1].values
                    )
                    x = np.arange(len(window_data), dtype=float)

                    # Check if we have enough unique values
                    if len(np.unique(window_data)) > 1:
                        # Normalize x and y to prevent SVD convergence issues
                        x = (x - x.mean()) / x.std()
                        y = (window_data - window_data.mean()) / window_data.std()

                        slope = np.polyfit(x, y, 1)[0]
                        self.df.loc[i, "price_trend"] = slope
                        trend_direction = (
                            1
                            if slope > trend_threshold
                            else -1 if slope < -trend_threshold else 0
                        )
                        self.df.loc[i, "price_trend_direction"] = trend_direction

                        # Update consecutive upward trend days
                        if trend_direction == 1:  # Upward trend
                            if i > 0:
                                self.df.loc[i, "upward_trend_days"] = (
                                    self.df.loc[i - 1, "upward_trend_days"] + 1
                                )
                            else:
                                self.df.loc[i, "upward_trend_days"] = 1
                        else:  # Not an upward trend
                            self.df.loc[i, "upward_trend_days"] = 0

                    else:
                        # If all values are the same, trend is flat
                        self.df.loc[i, "price_trend"] = 0
                        self.df.loc[i, "price_trend_direction"] = 0
                        self.df.loc[i, "upward_trend_days"] = 0

                except (np.linalg.LinAlgError, ValueError) as e:
                    # If there's an error, use the previous value
                    if i > 0:
                        self.df.loc[i, "price_trend"] = self.df.loc[
                            i - 1, "price_trend"
                        ]
                        self.df.loc[i, "price_trend_direction"] = self.df.loc[
                            i - 1, "price_trend_direction"
                        ]
                        self.df.loc[i, "upward_trend_days"] = (
                            0  # Reset upward trend days on error
                        )
                    else:
                        self.df.loc[i, "price_trend"] = 0
                        self.df.loc[i, "price_trend_direction"] = 0
                        self.df.loc[i, "upward_trend_days"] = 0

        # Fill initial values
        self.df.loc[:lookback_period, "window_size"] = initial_window
        self.df.loc[:lookback_period, "upward_trend_days"] = (
            0  # Initialize early values
        )

        # Log statistics about window sizes and upward trends
        avg_window = self.df["window_size"].mean()
        min_window = self.df["window_size"].min()
        max_window = self.df["window_size"].max()
        max_upward_days = self.df["upward_trend_days"].max()
        avg_upward_days = self.df["upward_trend_days"].mean()

        logger.info(
            f"[crossover][{self.ticker}] Window size statistics - "
            f"Average: {avg_window:.1f}, Min: {min_window}, Max: {max_window}"
        )
        logger.info(
            f"[crossover][{self.ticker}] Upward trend statistics - "
            f"Max consecutive days: {max_upward_days}, Average: {avg_upward_days:.1f}"
        )

    def find_crossover_patterns(self):
        """Find and group crossover patterns in the data."""
        logger.info(f"[crossover][{self.ticker}] Finding crossover patterns")

        # Calculate SMAs first
        self._calculate_sma()

        # Calculate price trends
        self._calculate_price_trends()

        # Initialize pattern columns
        self.df["pattern_group_buy"] = 0
        self.df["pattern_group_ditch"] = 0

        buy_pattern_count = 1
        ditch_pattern_count = 1
        in_pattern = False
        current_pattern_start = None
        initial_price = None

        # Iterate through the data to identify patterns
        for i in range(1, len(self.df) - 1):  # Start from 1 to check previous day
            current_price = self.df.iloc[i]["close"]
            prev_sma_condition = self.df.iloc[i - 1]["SMA_lower_below_upper"]
            upward_trend_days = self.df.iloc[i]["upward_trend_days"]
            avg_upward_days = self.df.iloc[i]["window_size"]
            trend_direction = self.df.iloc[i]["price_trend_direction"]
            # Start pattern only when previous day have SMA_lower below upper
            # and it's been trending upward for N days as long as the number of days trending
            # is less than the average of upper trend length from historical data
            if (
                not in_pattern
                and prev_sma_condition
                and upward_trend_days <= avg_upward_days
                and trend_direction == 1
                and upward_trend_days > 1
            ):
                in_pattern = True
                current_pattern_start = i
                initial_price = current_price
                pattern_days = 1

            elif in_pattern:
                pattern_days = i - current_pattern_start + 1
                price_change = (current_price - initial_price) / initial_price

                # Check pattern conditions
                if pattern_days == self.config.pattern_length:
                    # Pattern reached max length - check price direction
                    if current_price > initial_price:
                        self.df.loc[
                            self.df.index[slice(current_pattern_start, i + 1)],
                            "pattern_group_buy",
                        ] = buy_pattern_count
                        buy_pattern_count += 1
                    else:
                        self.df.loc[
                            self.df.index[slice(current_pattern_start, i + 1)],
                            "pattern_group_ditch",
                        ] = ditch_pattern_count
                        ditch_pattern_count += 1

                    # Reset pattern tracking
                    in_pattern = False
                    current_pattern_start = None
                    initial_price = None
                    continue

                # Check take profit/loss conditions
                if price_change >= self.config.take_profit:
                    self.df.loc[
                        self.df.index[slice(current_pattern_start, i + 1)],
                        "pattern_group_buy",
                    ] = buy_pattern_count
                    buy_pattern_count += 1

                    # Reset pattern tracking
                    in_pattern = False
                    current_pattern_start = None
                    initial_price = None

                elif price_change <= -self.config.stop_loss:
                    self.df.loc[
                        self.df.index[slice(current_pattern_start, i + 1)],
                        "pattern_group_ditch",
                    ] = ditch_pattern_count
                    ditch_pattern_count += 1

                    # Reset pattern tracking
                    in_pattern = False
                    current_pattern_start = None
                    initial_price = None

    def save_dataframe(self, output_file="dataframe.csv"):
        """Save the dataframe to a csv file."""
        self.df.to_csv(self.output_path / output_file, index=False)
        logger.info(f"[crossover][{self.ticker}] Dataframe saved to {output_file}")

    @property
    def total_buy_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_buy"].nunique() - 1  # Subtract 1 to exclude 0

    @property
    def total_ditch_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_ditch"].nunique() - 1  # Subtract 1 to exclude 0

    def write_pattern_statistics(self, output_file="report.txt"):
        """Write pattern statistics to a file."""
        logger.info(
            f"[crossover][{self.ticker}] Writing pattern statistics to {output_file}"
        )
        if (
            "pattern_group_buy" not in self.df.columns
            or "pattern_group_ditch" not in self.df.columns
        ):
            logger.error(
                f"[crossover][{self.ticker}] Pattern group columns not found. Please run find_crossover_patterns first."
            )
            return

        with open(self.output_path / output_file, "w") as f:
            # Configuration information
            f.write(f"=== Configuration ===\n")
            f.write(f"Take Profit: {self.config.take_profit * 100:.1f}%\n")
            f.write(f"Stop Loss: {self.config.stop_loss * 100:.1f}%\n")
            f.write(f"Pattern Length: {self.config.pattern_length} days\n")
            f.write(f"SMAs: {self.config.lower_sma}/{self.config.upper_sma}\n\n")

            # Add Price Level Analysis section after the Configuration section
            f.write("=== Price Level Analysis ===\n")

            # Buy Patterns Price Analysis
            f.write("\nBuy Pattern Price Levels:\n")
            for pattern_id in range(1, self.total_buy_patterns + 1):
                pattern_data = self.df[self.df["pattern_group_buy"] == pattern_id]
                if not pattern_data.empty:
                    entry_price = pattern_data["close"].iloc[0]
                    exit_price = pattern_data["close"].iloc[-1]
                    take_profit_price = entry_price * (1 + self.config.take_profit)
                    stop_loss_price = entry_price * (1 - self.config.stop_loss)
                    actual_return = ((exit_price - entry_price) / entry_price) * 100

                    f.write(f"\nPattern {pattern_id}:\n")
                    f.write(f"  Entry Price: ${entry_price:.2f}\n")
                    f.write(f"  Exit Price: ${exit_price:.2f}\n")
                    f.write(
                        f"  Take Profit Level (+"
                        f"{self.config.take_profit*100:.1f}%): ${take_profit_price:.2f}\n"
                    )
                    f.write(
                        f"  Stop Loss Level (-"
                        f"{self.config.stop_loss*100:.1f}%): ${stop_loss_price:.2f}\n"
                    )
                    f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                    f.write(f"  Pattern Length: {len(pattern_data)} days\n")
                    f.write(
                        f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                        f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n"
                    )

            # Ditch Patterns Price Analysis
            f.write("\nDitch Pattern Price Levels:\n")
            for pattern_id in range(1, self.total_ditch_patterns + 1):
                pattern_data = self.df[self.df["pattern_group_ditch"] == pattern_id]
                if not pattern_data.empty:
                    entry_price = pattern_data["close"].iloc[0]
                    exit_price = pattern_data["close"].iloc[-1]
                    take_profit_price = entry_price * (1 + self.config.take_profit)
                    stop_loss_price = entry_price * (1 - self.config.stop_loss)
                    actual_return = ((exit_price - entry_price) / entry_price) * 100

                    f.write(f"\nPattern {pattern_id}:\n")
                    f.write(f"  Entry Price: ${entry_price:.2f}\n")
                    f.write(f"  Exit Price: ${exit_price:.2f}\n")
                    f.write(
                        f"  Take Profit Level (+"
                        f"{self.config.take_profit*100:.1f}%): ${take_profit_price:.2f}\n"
                    )
                    f.write(
                        f"  Stop Loss Level (-"
                        f"{self.config.stop_loss*100:.1f}%): ${stop_loss_price:.2f}\n"
                    )
                    f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                    f.write(f"  Pattern Length: {len(pattern_data)} days\n")
                    f.write(
                        f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                        f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n"
                    )

            f.write("\n")  # Add spacing before next section

            # Basic pattern statistics
            total_buy_days = len(self.df[self.df["pattern_group_buy"] > 0])
            total_ditch_days = len(self.df[self.df["pattern_group_ditch"] > 0])
            unique_buy_patterns = self.df["pattern_group_buy"].nunique() - 1
            unique_ditch_patterns = self.df["pattern_group_ditch"].nunique() - 1

            f.write(f"=== Pattern Statistics for {self.ticker} ===\n")
            f.write(
                f"Analysis Period: {self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}\n"
            )
            f.write(f"Total days with buy patterns: {total_buy_days}\n")
            f.write(f"Total days with ditch patterns: {total_ditch_days}\n")
            f.write(f"Number of unique buy patterns: {unique_buy_patterns}\n")
            f.write(f"Number of unique ditch patterns: {unique_ditch_patterns}\n\n")

            # Pattern length statistics for buy patterns
            f.write("=== Buy Pattern Length Analysis ===\n")
            buy_lengths = {}
            buy_profits = []
            for pattern_id in range(1, unique_buy_patterns + 1):
                pattern_data = self.df[self.df["pattern_group_buy"] == pattern_id]
                if not pattern_data.empty:
                    length = len(pattern_data)
                    buy_lengths[pattern_id] = length
                    start_price = pattern_data["close"].iloc[0]
                    end_price = pattern_data["close"].iloc[-1]
                    price_change = ((end_price - start_price) / start_price) * 100
                    buy_profits.append(price_change)
                    f.write(
                        f"Pattern {pattern_id}: {length} days, {price_change:+.2f}% return "
                        f"({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to "
                        f"{pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n"
                    )

            if buy_lengths:
                avg_length = sum(buy_lengths.values()) / len(buy_lengths)
                max_length = max(buy_lengths.values())
                min_length = min(buy_lengths.values())
                avg_profit = sum(buy_profits) / len(buy_profits) if buy_profits else 0
                f.write(f"\nBuy Pattern Summary:\n")
                f.write(f"  Average Length: {avg_length:.2f} days\n")
                f.write(f"  Maximum Length: {max_length} days\n")
                f.write(f"  Minimum Length: {min_length} days\n")
                f.write(f"  Average Return: {avg_profit:+.2f}%\n")
                f.write(
                    f"  Take Profit Hits: {sum(1 for x in buy_profits if x >= self.config.take_profit * 100)}\n"
                )
                f.write(
                    f"  Stop Loss Hits: {sum(1 for x in buy_profits if x <= -self.config.stop_loss * 100)}\n"
                )
                f.write(
                    f"  Pattern Length Hits: {sum(1 for x in buy_lengths.values() if x >= self.config.pattern_length)}\n\n"
                )

            # Pattern length statistics for ditch patterns
            f.write("=== Ditch Pattern Length Analysis ===\n")
            ditch_lengths = {}
            ditch_losses = []
            for pattern_id in range(1, unique_ditch_patterns + 1):
                pattern_data = self.df[self.df["pattern_group_ditch"] == pattern_id]
                if not pattern_data.empty:
                    length = len(pattern_data)
                    ditch_lengths[pattern_id] = length
                    start_price = pattern_data["close"].iloc[0]
                    end_price = pattern_data["close"].iloc[-1]
                    price_change = ((end_price - start_price) / start_price) * 100
                    ditch_losses.append(price_change)
                    f.write(
                        f"Pattern {pattern_id}: {length} days, {price_change:+.2f}% return "
                        f"({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to "
                        f"{pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n"
                    )

            if ditch_lengths:
                avg_length = sum(ditch_lengths.values()) / len(ditch_lengths)
                max_length = max(ditch_lengths.values())
                min_length = min(ditch_lengths.values())
                avg_loss = sum(ditch_losses) / len(ditch_losses) if ditch_losses else 0
                f.write(f"\nDitch Pattern Summary:\n")
                f.write(f"  Average Length: {avg_length:.2f} days\n")
                f.write(f"  Maximum Length: {max_length} days\n")
                f.write(f"  Minimum Length: {min_length} days\n")
                f.write(f"  Average Return: {avg_loss:+.2f}%\n")
                f.write(
                    f"  Take Profit Hits: {sum(1 for x in ditch_losses if x >= self.config.take_profit * 100)}\n"
                )
                f.write(
                    f"  Stop Loss Hits: {sum(1 for x in ditch_losses if x <= -self.config.stop_loss * 100)}\n"
                )
                f.write(
                    f"  Pattern Length Hits: {sum(1 for x in ditch_lengths.values() if x >= self.config.pattern_length)}\n\n"
                )

            # Monthly statistics
            f.write("=== Monthly Analysis ===\n")
            self.df["month"] = self.df["timestamp"].dt.to_period("M")
            monthly_buy = (
                self.df[self.df["pattern_group_buy"] > 0]
                .groupby("month")["pattern_group_buy"]
                .nunique()
            )
            monthly_ditch = (
                self.df[self.df["pattern_group_ditch"] > 0]
                .groupby("month")["pattern_group_ditch"]
                .nunique()
            )

            f.write("Buy Patterns:\n")
            f.write(f"  Average per month: {monthly_buy.mean():.2f}\n")
            f.write("Most active months:\n")
            for month, count in monthly_buy.nlargest(3).items():
                f.write(f"  {month}: {count} patterns\n")

            f.write("\nDitch Patterns:\n")
            f.write(f"  Average per month: {monthly_ditch.mean():.2f}\n")
            f.write("Most active months:\n")
            for month, count in monthly_ditch.nlargest(3).items():
                f.write(f"  {month}: {count} patterns\n")
            f.write("\n")

            # Seasonal analysis
            f.write("=== Seasonal Distribution ===\n")
            self.df["season"] = self.df["timestamp"].dt.month % 12 // 3
            season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
            self.df["season"] = self.df["season"].map(season_names)

            seasonal_buy = (
                self.df[self.df["pattern_group_buy"] > 0]
                .groupby("season")["pattern_group_buy"]
                .nunique()
            )
            seasonal_ditch = (
                self.df[self.df["pattern_group_ditch"] > 0]
                .groupby("season")["pattern_group_ditch"]
                .nunique()
            )

            f.write("Buy Patterns:\n")
            for season, count in seasonal_buy.items():
                f.write(f"  {season}: {count} patterns\n")

            f.write("\nDitch Patterns:\n")
            for season, count in seasonal_ditch.items():
                f.write(f"  {season}: {count} patterns\n")
            f.write("\n")

            # Add Combined Performance Analysis
            f.write("=== Combined Performance Analysis ===\n")
            total_buy_return = sum(buy_profits) if buy_profits else 0
            total_ditch_return = sum(ditch_losses) if ditch_losses else 0
            combined_return = total_buy_return + total_ditch_return

            total_patterns = len(buy_profits) + len(ditch_losses)
            if total_patterns > 0:
                avg_pattern_return = combined_return / total_patterns
            else:
                avg_pattern_return = 0

            f.write(f"Total Buy Pattern Returns: {total_buy_return:+.2f}%\n")
            f.write(f"Total Ditch Pattern Returns: {total_ditch_return:+.2f}%\n")
            f.write(f"Combined Return: {self.calculate_combined_return():+.2f}%\n")
            f.write(f"Average Return per Pattern: {avg_pattern_return:+.2f}%\n")
            f.write(f"Total Number of Patterns: {total_patterns}\n\n")

            # Add timestamp of analysis
            f.write(
                f"\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        logger.info(f"[crossover][{self.ticker}] Statistics written to {output_file}")

    def plot_analysis(self, filename="plot.png"):
        """Create and save the analysis plot."""
        logger.info(f"[crossover][{self.ticker}] Creating analysis plot")

        # Set the backend to Agg explicitly
        import matplotlib

        matplotlib.use("Agg")

        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(15, 12),
                height_ratios=[3, 1],
                gridspec_kw={"hspace": 0.3},
            )

            # Upper subplot - Price and SMAs
            ax1.plot(
                self.df["timestamp"],
                self.df["close"],
                label="Daily Close Price",
                alpha=0.6,
                marker="o",
                markersize=3,
            )
            ax1.plot(
                self.df["timestamp"],
                self.df["SMA_lower"],
                label=f"{self.config.lower_sma} SMA",
                color="orange",
                linewidth=2,
            )
            ax1.plot(
                self.df["timestamp"],
                self.df["SMA_upper"],
                label=f"{self.config.upper_sma} SMA",
                color="green",
                linewidth=2,
            )

            # Plot buy patterns
            buy_groups = sorted(
                self.df[self.df["pattern_group_buy"] > 0]["pattern_group_buy"].unique()
            )
            for group in buy_groups:
                group_data = self.df[self.df["pattern_group_buy"] == group]
                ax1.scatter(
                    group_data["timestamp"],
                    group_data["close"],
                    color="green",
                    marker="^",
                    label=f"Buy Pattern {group}" if group == buy_groups[0] else "",
                    zorder=5,
                    s=100,
                )

                # Add annotation for entry point
                entry_point = group_data.iloc[0]
                ax1.annotate(
                    "Entry",
                    xy=(entry_point["timestamp"], entry_point["close"]),
                    xytext=(10, 30),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

                # Add annotation for exit point
                exit_point = group_data.iloc[-1]
                ax1.annotate(
                    "Exit",
                    xy=(exit_point["timestamp"], exit_point["close"]),
                    xytext=(10, -30),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            # Plot ditch patterns
            ditch_groups = sorted(
                self.df[self.df["pattern_group_ditch"] > 0][
                    "pattern_group_ditch"
                ].unique()
            )
            for group in ditch_groups:
                group_data = self.df[self.df["pattern_group_ditch"] == group]
                ax1.scatter(
                    group_data["timestamp"],
                    group_data["close"],
                    color="red",
                    marker="v",
                    label=f"Ditch Pattern {group}" if group == ditch_groups[0] else "",
                    zorder=5,
                    s=100,
                )

                # Add annotation for entry point
                entry_point = group_data.iloc[0]
                ax1.annotate(
                    "Entry",
                    xy=(entry_point["timestamp"], entry_point["close"]),
                    xytext=(-10, 30),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

                # Add annotation for exit point
                exit_point = group_data.iloc[-1]
                ax1.annotate(
                    "Exit",
                    xy=(exit_point["timestamp"], exit_point["close"]),
                    xytext=(-10, -30),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            ax1.set_title(
                f"{self.ticker} Daily Price with {self.config.lower_sma} and {self.config.upper_sma} Period SMAs"
            )
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            ax1.tick_params(axis="x", rotation=45)

            # Lower subplot - Price trend direction
            ax2.plot(
                self.df["timestamp"],
                self.df["price_trend_direction"],
                label="Price Trend",
                color="purple",
                linestyle="-",
                linewidth=2,
            )

            # Set y-axis limits and ticks for trend subplot
            ax2.set_ylim([-1.5, 1.5])
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(["Downward", "Flat", "Upward"])
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price Trend Direction")
            ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

            # Generate filename with timestamp
            filename = f"{self.ticker}_sma_analysis.png"
            filepath = self.output_path / filename

            # Save the plot with extra space for legend
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"[crossover][{self.ticker}] Plot saved to {filepath}")

            # Clear the current figure and close to free memory
            plt.clf()
            plt.close(fig)

        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error creating plot: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources and remove output directory if exists."""
        try:
            if hasattr(self, "output_path") and self.output_path.exists():
                shutil.rmtree(self.output_path)
                logger.info(f"[crossover][{self.ticker}] Cleaned up output directory")
        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error during cleanup: {str(e)}")

    def calculate_combined_return(self) -> float:
        """Calculate the combined return from both buy and ditch patterns.

        Returns:
            float: The combined percentage return from all patterns
        """
        buy_returns = []
        ditch_returns = []

        # Calculate returns for buy patterns
        for pattern_id in range(1, self.total_buy_patterns + 1):
            pattern_data = self.df[self.df["pattern_group_buy"] == pattern_id]
            if not pattern_data.empty:
                start_price = pattern_data["close"].iloc[0]
                end_price = pattern_data["close"].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                buy_returns.append(return_pct)

        # Calculate returns for ditch patterns
        for pattern_id in range(1, self.total_ditch_patterns + 1):
            pattern_data = self.df[self.df["pattern_group_ditch"] == pattern_id]
            if not pattern_data.empty:
                start_price = pattern_data["close"].iloc[0]
                end_price = pattern_data["close"].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                ditch_returns.append(return_pct)

        # Calculate combined return
        total_buy_return = sum(buy_returns) if buy_returns else 0
        total_ditch_return = sum(ditch_returns) if ditch_returns else 0
        combined_return = total_buy_return + total_ditch_return

        return combined_return


def process_ticker(
    ticker: str, min_return: float, min_buy_patterns: int
) -> tuple[str, float]:
    """Process a single ticker."""
    fetcher = StockDataFetcher()
    df = fetcher.get_stock_data(ticker, timeframe="1D")

    analyzer = SMACrossoverAnalyzer(
        ticker=ticker,
        dataframe=df,
        config=CrossoverConfig(
            upper_sma=200,
            lower_sma=50,
            stop_loss=0.05,
            take_profit=0.1,
            pattern_length=10,
        ),
    )
    analyzer.find_crossover_patterns()

    if analyzer.total_buy_patterns >= min_buy_patterns:
        combined_return = int(analyzer.calculate_combined_return())
        if combined_return >= min_return:
            analyzer.write_pattern_statistics()
            analyzer.plot_analysis()
            analyzer.save_dataframe()
            return ticker, combined_return
    analyzer.cleanup()
    return ticker, 0


def process_tickers(
    tickers: List[str],
    min_return: float,
    min_buy_patterns: int,
) -> Dict[str, float]:
    """Process multiple tickers and return results above threshold.

    Args:
        tickers: List of stock ticker symbols
        min_return: Minimum absolute combined return percentage to keep

    Returns:
        dict: Dictionary of tickers and their combined returns that meet the threshold
    """
    results = {}

    # Clean the tickers and remove any whitespace/newlines
    clean_tickers = [t.strip() for t in tickers if t.strip()]

    # Process tickers sequentially
    for ticker in clean_tickers:
        ticker, combined_return = process_ticker(ticker, min_return, min_buy_patterns)
        if combined_return != 0:  # Only keep results that met the threshold
            results[ticker] = combined_return

    # Sort results by absolute return value
    sorted_results = dict(
        sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Log the results
    logger.info(f"Found {len(sorted_results)} tickers with significant patterns:")
    for ticker, return_value in sorted_results.items():
        logger.info(f"{ticker}: {return_value:+.2f}%")

    return sorted_results


if __name__ == "__main__":
    with open("../tickers/large_cap.txt", "r") as f:
        tickers = f.readlines()

    # Process tickers and get results with minimum 5% absolute combined return
    results = process_tickers(tickers, min_return=10.0, min_buy_patterns=2)
