from ..get_data import StockDataFetcher
import datetime
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


class SMACrossoverAnalyzerLive:
    def __init__(self, ticker: str, config: CrossoverConfig):

        self.ticker = ticker
        self.config = config or CrossoverConfig(
            upper_sma=50,
            lower_sma=20,
            stop_loss=0.05,
            take_profit=0.1,
            pattern_length=10,
        )
        self.df = self._get_previous_quotes()

        # Calculate SMAs first
        self._calculate_sma()

        # Calculate price trends
        self._calculate_price_trends()

        self._print_today_data()

    def _print_today_data(self):
        """Print today's data and crossover pattern status."""
        today_data = self.df.iloc[-1]
        today_date = today_data["timestamp"].strftime("%Y-%m-%d")

        # Check for crossover pattern
        prev_sma_condition = self.df.iloc[-2]["SMA_lower_below_upper"]
        upward_trend_days = today_data["upward_trend_days"]
        avg_upward_days = today_data["window_size"]
        trend_direction = today_data["price_trend_direction"]
        prev_sma_condition = self.df.iloc[-2]["SMA_lower_below_upper"]

        has_pattern = (
            prev_sma_condition
            and upward_trend_days <= avg_upward_days
            and trend_direction == 1
            and upward_trend_days > 1
        )

        pattern_status = "PATTERN DETECTED" if has_pattern else "No pattern"

        logger.info(
            f"[crossover][live][{self.ticker}] Today's data ({today_date}):\n"
            f"  Close: ${today_data['close']:.2f}\n"
            f"  Is yesterday SMA Lower below Upper: {prev_sma_condition}\n"
            f"  Are we in a upward trend? {trend_direction == 1}\n"
            f"  Upward trend days: {upward_trend_days}\n"
            f"  Average upward trend days: {avg_upward_days}\n"
            f"  Pattern Status: {pattern_status}"
        )

    def _read_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
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
            df = dataframe.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Validate data
            if df["timestamp"].isnull().any():
                raise ValueError("Dataset contains null timestamps")
            if df["close"].isnull().any():
                raise ValueError("Dataset contains null close prices")

            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.info(f"[crossover][live][{self.ticker}] Data loaded successfully")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.cleanup()
            raise

    def _get_previous_quotes(self, timeframe="1D"):
        fetcher = StockDataFetcher()
        today = datetime.datetime.now()
        # get today data from alpaca all the way back to the upper_sma (e.g. 200 days)
        df = fetcher.get_stock_data(
            self.ticker,
            timeframe,
            start_date=(today - datetime.timedelta(days=self.config.upper_sma * 2)).strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )

        return self._read_data(df)

    def _calculate_sma(self):
        """Calculate SMAs on daily data."""
        logger.info(f"[crossover][live][{self.ticker}] Calculating SMAs")
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
            f"[crossover][live][{self.ticker}] Calculating price trends with dynamic windows"
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


    def is_sma_crossover(self):
        """Find and group crossover patterns in the data."""
        logger.info(
            f"[crossover][live][{self.ticker}] Finding crossover patterns in today's data"
        )

        # Initialize pattern columns
        self.df["pattern_group_buy"] = 0
        self.df["pattern_group_ditch"] = 0

        prev_sma_condition = self.df.iloc[-2]["SMA_lower_below_upper"]
        upward_trend_days = self.df.iloc[-1]["upward_trend_days"]
        avg_upward_days = self.df.iloc[-1]["window_size"]
        trend_direction = self.df.iloc[-1]["price_trend_direction"]

        today_date = self.df.iloc[-1]["timestamp"].strftime("%Y-%m-%d")
        logger.info(f"[crossover][live][{self.ticker}] Today's date: {today_date}")
        # Start pattern only when previous day have SMA_lower below upper
        # and it's been trending upward for N days as long as the number of days trending
        # is less than the average of upper trend length from historical data
        if (
            prev_sma_condition
            and upward_trend_days <= avg_upward_days
            and trend_direction == 1
            and upward_trend_days > 1
        ):
            return True
        else:
            return False

    def buy(self):
        """Buy the stock"""
        logger.info(f"[crossover][live][{self.ticker}] Buying the stock")

    def sell(self):
        """Sell the stock"""
        pass

    def plot_analysis(self, filename="plot.png"):
        """Create and save the analysis plot."""
        logger.info(f"[crossover][live][{self.ticker}] Creating analysis plot")

        # Set the backend to Agg explicitly
        import matplotlib

        matplotlib.use("Agg")

        try:
            # Create figure with two subplots
            fig, ax1 = plt.subplots(
                1, 1, figsize=(15, 12), height_ratios=[3], gridspec_kw={"hspace": 0.3}
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

            ax1.set_title(
                f"{self.ticker} Daily Price with {self.config.lower_sma} and {self.config.upper_sma} Period SMAs"
            )
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            ax1.tick_params(axis="x", rotation=45)
            # Save the plot with extra space for legend
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            logger.info(f"[crossover][live][{self.ticker}] Plot saved to {filename}")

            # Clear the current figure and close to free memory
            plt.clf()
            plt.close(fig)

        except Exception as e:
            logger.error(
                f"[crossover][live][{self.ticker}] Error creating plot: {str(e)}"
            )
            raise


if __name__ == "__main__":
    analyzer = SMACrossoverAnalyzerLive(
        "AAPL",
        CrossoverConfig(
            upper_sma=50,
            lower_sma=20,
            stop_loss=0.05,
            take_profit=0.1,
            pattern_length=10,
        ),
    )
    if analyzer.is_sma_crossover():
        analyzer.buy()

    analyzer.df.to_csv("./AAPL_sma_crossover.csv", index=False)
    analyzer.plot_analysis()
