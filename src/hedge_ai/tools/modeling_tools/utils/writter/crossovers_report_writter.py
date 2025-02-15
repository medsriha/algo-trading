from datetime import datetime
import logging
import pandas as pd
from pathlib import Path
from hedge_ai.tools.modeling_tools.configs.configs import CrossoverConfig
logger = logging.getLogger(__name__)


class CrossoversReportWriter:
    def __init__(self, df: pd.DataFrame, ticker: str, crossover_config: CrossoverConfig, output_path: Path):
        """Initialize the CrossoverReportWriter.

        Args:
            df: DataFrame containing the trading data
            ticker: Stock ticker symbol
            crossover_config: Configuration object containing trading parameters
            output_path: Path where reports should be saved
            crossovers: Dictionary containing crossover statistics
            total_buy_patterns: Total number of buy patterns
            total_ditch_patterns: Total number of loss patterns
        """
        self.df = df
        self.ticker = ticker
        self.crossover_config = crossover_config
        self.output_path = output_path

    def _write_config_section(self, f):
        """Helper method to write configuration section."""
        f.write(f"=== Configuration ===\n")
        f.write(f"Take Profit: {self.crossover_config.take_profit * 100:.1f}%\n")
        f.write(f"Stop Loss: {self.crossover_config.stop_loss * 100:.1f}%\n")
        f.write(f"Pattern Length: {self.crossover_config.crossover_length} days\n")
        f.write(f"SMAs: {self.crossover_config.lower_sma}/{self.crossover_config.upper_sma}\n\n")

    def _write_pattern_analysis(self, f, pattern_type, total_patterns, pattern_column):
        """Helper method to write pattern price analysis."""
        f.write(f"\n{pattern_type} Pattern Price Levels:\n")
        if total_patterns == 0:
            f.write(f"No {pattern_type.lower()} patterns found in the data.\n")
            return

        for pattern_id in range(1, total_patterns + 1):
            pattern_data = self.df[self.df[pattern_column] == pattern_id]
            if not pattern_data.empty:
                entry_price = pattern_data["close"].iloc[0]
                exit_price = pattern_data["close"].iloc[-1]
                take_profit_price = entry_price * (1 + self.crossover_config.take_profit)
                stop_loss_price = entry_price * (1 - self.crossover_config.stop_loss)
                actual_return = ((exit_price - entry_price) / entry_price) * 100

                f.write(f"\nPattern {pattern_id}:\n")
                f.write(f"  Entry Price: ${entry_price:.2f}\n")
                f.write(f"  Exit Price: ${exit_price:.2f}\n")
                f.write(f"  Take Profit Level (+{self.crossover_config.take_profit * 100:.1f}%): ${take_profit_price:.2f}\n")
                f.write(f"  Stop Loss Level (-{self.crossover_config.stop_loss * 100:.1f}%): ${stop_loss_price:.2f}\n")
                f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                f.write(f"  Pattern Length: {len(pattern_data)} days\n")
                f.write(
                    f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                    f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n"
                )

    def _analyze_pattern_lengths(self, f, pattern_type, unique_patterns, pattern_column):
        """Helper method to analyze pattern lengths."""
        f.write(f"=== {pattern_type} Pattern Length Analysis ===\n")
        pattern_lengths = {}
        pattern_profits = []
        
        for pattern_id in range(1, unique_patterns + 1):
            pattern_data = self.df[self.df[pattern_column] == pattern_id]
            if not pattern_data.empty:
                length = len(pattern_data)
                pattern_lengths[pattern_id] = length
                start_price = pattern_data["close"].iloc[0]
                end_price = pattern_data["close"].iloc[-1]
                price_change = ((end_price - start_price) / start_price) * 100
                pattern_profits.append(price_change)
                f.write(
                    f"Pattern {pattern_id}: {length} days, {price_change:+.2f}% return "
                    f"({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to "
                    f"{pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n"
                )

        if pattern_lengths:
            avg_length = sum(pattern_lengths.values()) / len(pattern_lengths)
            max_length = max(pattern_lengths.values())
            min_length = min(pattern_lengths.values())
            avg_return = sum(pattern_profits) / len(pattern_profits) if pattern_profits else 0
            
            f.write(f"\n{pattern_type} Pattern Summary:\n")
            f.write(f"  Average Length: {avg_length:.2f} days\n")
            f.write(f"  Maximum Length: {max_length} days\n")
            f.write(f"  Minimum Length: {min_length} days\n")
            f.write(f"  Average Return: {avg_return:+.2f}%\n")
            f.write(f"  Take Profit Hits: {sum(1 for x in pattern_profits if x >= self.crossover_config.take_profit * 100)}\n")
            f.write(f"  Stop Loss Hits: {sum(1 for x in pattern_profits if x <= -self.crossover_config.stop_loss * 100)}\n")
            f.write(
                f"  Pattern Length Hits: {sum(1 for x in pattern_lengths.values() if x >= self.crossover_config.crossover_length)}\n\n"
            )
        else:
            f.write(f"No {pattern_type.lower()} patterns found in the data.\n\n") 
            
        return pattern_profits

    def _write_basic_stats(self, f):
        """Helper method to write basic pattern statistics."""
        total_gains_days = len(self.df[self.df["gain"] > 0])
        total_losses_days = len(self.df[self.df["loss"] > 0])
        unique_gains = self.df["gain"].nunique() - 1
        unique_losses = self.df["loss"].nunique() - 1

        f.write(f"=== Pattern Statistics for {self.ticker} ===\n")
        f.write(
            f"Analysis Period: {self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}\n"
        )
        f.write(f"Total days with gains: {total_gains_days}\n")
        f.write(f"Total days with losses: {total_losses_days}\n")
        f.write(f"Number of unique gains: {unique_gains}\n")
        f.write(f"Number of unique losses: {unique_losses}\n\n")

    def _write_monthly_analysis(self, f):
        """Helper method to write monthly analysis."""
        f.write("=== Monthly Analysis ===\n")
        self.df["month"] = self.df["timestamp"].dt.to_period("M")
        monthly_gains = self.df[self.df["gain"] > 0].groupby("month")["gain"].nunique()
        monthly_losses = self.df[self.df["loss"] > 0].groupby("month")["loss"].nunique()

        f.write("Buy Patterns:\n")
        f.write(f"  Average per month: {monthly_gains.mean():.2f}\n")
        f.write("Most active months:\n")
        for month, count in monthly_gains.nlargest(3).items():
            f.write(f"  {month}: {count} patterns\n")

        f.write("\nLoss Patterns:\n")
        f.write(f"  Average per month: {monthly_losses.mean():.2f}\n")
        f.write("Most active months:\n")
        for month, count in monthly_losses.nlargest(3).items():
            f.write(f"  {month}: {count} patterns\n")
        f.write("\n")

    def _write_seasonal_analysis(self, f):
        """Helper method to write seasonal analysis."""
        f.write("=== Seasonal Distribution ===\n")
        self.df["season"] = self.df["timestamp"].dt.month % 12 // 3
        season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
        self.df["season"] = self.df["season"].map(season_names)

        seasonal_buy = self.df[self.df["gain"] > 0].groupby("season")["gain"].nunique()
        seasonal_loss = self.df[self.df["loss"] > 0].groupby("season")["loss"].nunique()

        f.write("Buy Patterns:\n")
        for season, count in seasonal_buy.items():
            f.write(f"  {season}: {count} patterns\n")

        f.write("\nLoss Patterns:\n")
        for season, count in seasonal_loss.items():
            f.write(f"  {season}: {count} patterns\n")
        f.write("\n")

    def _write_crossover_analysis(self, f, count_crossovers):
        """Helper method to write crossover analysis."""
        f.write("=== Crossover Analysis ===\n")
        f.write(f"Total Crossovers: {count_crossovers['total']}\n")
        f.write(f"Bullish Crossovers (Lower SMA crosses above Upper): {count_crossovers['bullish']}\n")
        f.write(f"Bearish Crossovers (Lower SMA crosses below Upper): {count_crossovers['bearish']}\n\n")

    def _write_bearish_periods_analysis(self, f, bearish_periods):
        """Helper method to write bearish periods analysis."""
        f.write("=== Bearish Periods Analysis ===\n")
        if bearish_periods:
            total_days = sum(period["length"] for period in bearish_periods)
            avg_length = total_days / len(bearish_periods)
            max_length = max(period["length"] for period in bearish_periods)
            min_length = min(period["length"] for period in bearish_periods)
            avg_price_change = sum(period["price_change_pct"] for period in bearish_periods) / len(bearish_periods)

            f.write(f"Total Bearish Periods: {len(bearish_periods)}\n")
            f.write(f"Average Length: {avg_length:.2f} days\n")
            f.write(f"Longest Period: {max_length} days\n")
            f.write(f"Shortest Period: {min_length} days\n")
            f.write(f"Average Price Change: {avg_price_change:.2f}%\n\n")

            f.write("Individual Bearish Periods:\n")
            for i, period in enumerate(bearish_periods, 1):
                f.write(f"\nPeriod {i}:\n")
                f.write(f"  Start Date: {period['start_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"  End Date: {period['end_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"  Length: {period['length']} days\n")
                f.write(f"  Start Price: ${period['start_price']:.2f}\n")
                f.write(f"  End Price: ${period['end_price']:.2f}\n")
                f.write(f"  Price Change: {period['price_change_pct']:.2f}%\n")
            f.write("\n")
        else:
            f.write("No bearish periods found in the data.\n\n")

    def _write_combined_performance(self, f, gains_profits, losses_profits):
        """Helper method to write combined performance analysis."""
        total_gains_return = sum(gains_profits) if gains_profits else 0
        total_losses_return = sum(losses_profits) if losses_profits else 0
        combined_return = total_gains_return + total_losses_return

        total_patterns = len(gains_profits) + len(losses_profits)
        if total_patterns > 0:
            avg_pattern_return = combined_return / total_patterns
        else:
            avg_pattern_return = 0

        f.write(f"=== Combined Performance Analysis ===\n")
        f.write(f"Total Gain trade Returns: {total_gains_return:+.2f}%\n")
        f.write(f"Total Loss trade Returns: {total_losses_return:+.2f}%\n")
        f.write(f"Combined Return: {combined_return:+.2f}%\n")
        f.write(f"Average Return per trade: {avg_pattern_return:+.2f}%\n")
        f.write(f"Total Number of trades: {total_patterns}\n\n")

    def write(self, total_gains, total_losses, count_crossovers, bearish_periods, output_file="report.txt"):
        """Write pattern statistics to a file."""
        logger.info(f"[crossover][{self.ticker}] Writing pattern statistics to {output_file}")
        if "gain" not in self.df.columns or "loss" not in self.df.columns:
            logger.error(
                f"[crossover][{self.ticker}] Pattern group columns not found. Please run find_crossover_patterns first."
            )
            return

        with open(self.output_path / output_file, "w") as f:
            # Write configuration
            self._write_config_section(f)
            
            # Write price level analysis
            f.write("=== Price Level Analysis ===\n")
            self._write_pattern_analysis(f, "Gain", total_gains, "gain")
            self._write_pattern_analysis(f, "Loss", total_losses, "loss")
            
            # Write basic pattern statistics
            self._write_basic_stats(f)
            
            # Analyze pattern lengths and get profits
            gains_profits = self._analyze_pattern_lengths(f, "Gain", self.df["gain"].nunique() - 1, "gain")
            losses_profits = self._analyze_pattern_lengths(f, "Loss", self.df["loss"].nunique() - 1, "loss")
            
            # Write remaining sections
            self._write_monthly_analysis(f)
            self._write_seasonal_analysis(f)
            self._write_crossover_analysis(f, count_crossovers)
            self._write_bearish_periods_analysis(f, bearish_periods)
            self._write_combined_performance(f, gains_profits, losses_profits)
            
            # Add timestamp
            f.write(f"\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"[crossover][{self.ticker}] Statistics written to {output_file}")
