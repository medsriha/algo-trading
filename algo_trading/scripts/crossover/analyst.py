from datetime import datetime
import logging
import pandas as pd
from pathlib import Path
from algo_trading.models.crossover import CrossoverConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class CrossoverAnalyst:
    def __init__(self, frame: pd.DataFrame, ticker: str, crossover_config: CrossoverConfig, output_path: Path):
        """Initialize the CrossoverReportWriter.

        Args:
            frame: DataFrame containing the trading data
            ticker: Stock ticker symbol
            crossover_config: Configuration object containing trading parameters
            output_path: Path where reports should be saved
        """
        self.frame = frame
        self.ticker = ticker
        self.crossover_config = crossover_config
        self.output_path = output_path

    def _write_config_section(self, f):
        """Write strategy configuration and parameters."""
        f.write("=== Strategy Configuration ===\n")
        f.write(f"Strategy: SMA Crossover with RSI Filter\n")
        f.write(f"Fast SMA Period: {self.crossover_config.lower_sma} days\n")
        f.write(f"Slow SMA Period: {self.crossover_config.upper_sma} days\n")
        f.write(f"RSI Period: {self.crossover_config.rsi_period} days\n")
        f.write(f"RSI Oversold Level: {self.crossover_config.rsi_oversold}\n")
        f.write(f"RSI Overbought Level: {self.crossover_config.rsi_overbought}\n")
        f.write(f"RSI Entry Level: {self.crossover_config.rsi_underbought}\n")
        f.write(f"Maximum Trade Duration: {self.crossover_config.crossover_length} days\n")
        f.write(f"Take Profit: {self.crossover_config.take_profit * 100:.1f}%\n")
        f.write(f"Stop Loss: {self.crossover_config.stop_loss * 100:.1f}%\n\n")

    def _write_executive_summary(self, f, total_gains, total_losses, bearish_periods):
        """Write an executive summary of the strategy performance."""
        total_trades = total_gains + total_losses
        win_rate = (total_gains / total_trades * 100) if total_trades > 0 else 0

        # Calculate overall return
        total_return = 0
        for pattern_type in ["gain", "loss"]:
            pattern_data = self.frame[self.frame[pattern_type] > 0]
            for pattern_id in pattern_data[pattern_type].unique():
                data = pattern_data[pattern_data[pattern_type] == pattern_id]
                start_price = data["close"].iloc[0]
                end_price = data["close"].iloc[-1]
                total_return += ((end_price - start_price) / start_price) * 100

        f.write("=== Executive Summary ===\n")
        f.write(f"Ticker: {self.ticker}\n")
        f.write(
            f"Analysis Period: {self.frame['timestamp'].min().strftime('%Y-%m-%d')} to {self.frame['timestamp'].max().strftime('%Y-%m-%d')}\n"
        )
        f.write(f"Total Trading Days: {len(self.frame)}\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Winning Trades: {total_gains}\n")
        f.write(f"Losing Trades: {total_losses}\n")
        f.write(f"Win Rate: {win_rate:.1f}%\n")
        f.write(f"Total Return: {total_return:+.2f}%\n")
        f.write(f"Average Return per Trade: {(total_return / total_trades if total_trades > 0 else 0):+.2f}%\n")
        f.write(f"Number of Bearish Periods: {len(bearish_periods)}\n\n")

    def _write_risk_metrics(self, f):
        """Write risk-related metrics for the strategy."""
        f.write("=== Risk Metrics ===\n")

        # Calculate drawdowns for all patterns
        max_drawdown = 0
        avg_drawdown = 0
        drawdown_count = 0

        for pattern_type in ["gain", "loss"]:
            pattern_data = self.frame[self.frame[pattern_type] > 0]
            for pattern_id in pattern_data[pattern_type].unique():
                data = pattern_data[pattern_data[pattern_type] == pattern_id]
                start_price = data["close"].iloc[0]
                lowest_price = data["close"].min()
                drawdown = ((lowest_price - start_price) / start_price) * 100
                max_drawdown = min(max_drawdown, drawdown)
                avg_drawdown += drawdown
                drawdown_count += 1

        avg_drawdown = avg_drawdown / drawdown_count if drawdown_count > 0 else 0

        f.write(f"Maximum Drawdown: {max_drawdown:.2f}%\n")
        f.write(f"Average Drawdown: {avg_drawdown:.2f}%\n")
        f.write(
            f"Risk/Reward Ratio: {abs(self.crossover_config.take_profit / self.crossover_config.stop_loss):.2f}\n\n"
        )

    def _write_trade_analysis(self, f, pattern_type, total_patterns, pattern_column):
        """Helper method to write pattern price analysis."""
        f.write(f"\n{pattern_type} Trade Price Levels:\n")
        if total_patterns == 0:
            f.write(f"No {pattern_type.lower()} trades found in the data.\n")
            return

        for pattern_id in range(1, total_patterns + 1):
            pattern_data = self.frame[self.frame[pattern_column] == pattern_id]
            if not pattern_data.empty:
                entry_price = pattern_data["close"].iloc[0]
                exit_price = pattern_data["close"].iloc[-1]
                take_profit_price = entry_price * (1 + self.crossover_config.take_profit)
                stop_loss_price = entry_price * (1 - self.crossover_config.stop_loss)
                actual_return = ((exit_price - entry_price) / entry_price) * 100

                f.write(f"\nPattern {pattern_id}:\n")
                f.write(f"  Entry Price: ${entry_price:.2f}\n")
                f.write(f"  Exit Price: ${exit_price:.2f}\n")
                f.write(
                    f"  Take Profit Level (+{self.crossover_config.take_profit * 100:.1f}%): ${take_profit_price:.2f}\n"
                )
                f.write(f"  Stop Loss Level (-{self.crossover_config.stop_loss * 100:.1f}%): ${stop_loss_price:.2f}\n")
                f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                f.write(f"  Trade Length: {len(pattern_data)} days\n")
                f.write(
                    f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                    f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n"
                )

    def _write_basic_stats(self, f):
        """Helper method to write basic pattern statistics."""
        total_gains_days = len(self.frame[self.frame["gain"] > 0])
        total_losses_days = len(self.frame[self.frame["loss"] > 0])
        unique_gains = self.frame["gain"].nunique() - 1  # Subtract 1 to exclude 0
        unique_losses = self.frame["loss"].nunique() - 1  # Subtract 1 to exclude 0

        f.write(f"\n\n=== Trade Statistics for {self.ticker} ===\n")
        f.write(
            f"Analysis Period: {self.frame['timestamp'].min().strftime('%Y-%m-%d')} to {self.frame['timestamp'].max().strftime('%Y-%m-%d')}\n"
        )
        f.write(f"Total days with patterns: {total_gains_days + total_losses_days}\n")
        f.write(f"Days with gain patterns: {total_gains_days}\n")
        f.write(f"Days with loss patterns: {total_losses_days}\n")
        f.write(f"Number of unique gain patterns: {unique_gains}\n")
        f.write(f"Number of unique loss patterns: {unique_losses}\n\n")

    def _write_bearish_periods_analysis(self, f, bearish_periods):
        """Helper method to write bearish periods analysis."""
        f.write("=== Bearish Periods Analysis ===\n")
        if not bearish_periods:
            f.write("No bearish periods with patterns found in the data.\n\n")
            return

        # Calculate aggregate statistics
        total_days = sum(period["length"] for period in bearish_periods)
        avg_length = total_days / len(bearish_periods)
        max_length = max(period["length"] for period in bearish_periods)
        min_length = min(period["length"] for period in bearish_periods)
        avg_price_change = sum(period["price_change_pct"] for period in bearish_periods) / len(bearish_periods)

        # Write summary statistics
        f.write(f"Total Bearish Periods with Patterns: {len(bearish_periods)}\n")
        f.write(f"Average Length: {avg_length:.2f} days\n")
        f.write(f"Longest Period: {max_length} days\n")
        f.write(f"Shortest Period: {min_length} days\n")
        f.write(f"Average Price Change: {avg_price_change:.2f}%\n\n")

        f.write("\n")

    def _write_pattern_performance_summary(self, f):
        """Write summary of pattern performance metrics."""
        f.write("\n=== Pattern Performance Summary ===\n")

        # Calculate gain pattern metrics
        gain_patterns = self.frame[self.frame["gain"] > 0]
        if not gain_patterns.empty:
            gain_returns = []
            for pattern_id in gain_patterns["gain"].unique():
                pattern_data = gain_patterns[gain_patterns["gain"] == pattern_id]
                start_price = pattern_data["close"].iloc[0]
                end_price = pattern_data["close"].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                gain_returns.append(return_pct)

            f.write("\nGain Patterns:\n")
            f.write(f"  Average Return: {sum(gain_returns) / len(gain_returns):+.2f}%\n")
            f.write(f"  Best Return: {max(gain_returns):+.2f}%\n")
            f.write(f"  Worst Return: {min(gain_returns):+.2f}%\n")
            f.write(f"  Success Rate: {len([r for r in gain_returns if r > 0]) / len(gain_returns) * 100:.1f}%\n")

        # Calculate loss pattern metrics
        loss_patterns = self.frame[self.frame["loss"] > 0]
        if not loss_patterns.empty:
            loss_returns = []
            for pattern_id in loss_patterns["loss"].unique():
                pattern_data = loss_patterns[loss_patterns["loss"] == pattern_id]
                start_price = pattern_data["close"].iloc[0]
                end_price = pattern_data["close"].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                loss_returns.append(return_pct)

            f.write("\nLoss Patterns:\n")
            f.write(f"  Average Return: {sum(loss_returns) / len(loss_returns):+.2f}%\n")
            f.write(f"  Best Return: {max(loss_returns):+.2f}%\n")
            f.write(f"  Worst Return: {min(loss_returns):+.2f}%\n")
            f.write(f"  Recovery Rate: {len([r for r in loss_returns if r > 0]) / len(loss_returns) * 100:.1f}%\n")

    def _write_rsi_analysis(self, f):
        """Write RSI analysis for patterns."""
        f.write("\n=== RSI Analysis ===\n")

        for pattern_type in ["gain", "loss"]:
            pattern_data = self.frame[self.frame[pattern_type] > 0]
            if not pattern_data.empty:
                entry_rsi = pattern_data.groupby(pattern_type)["RSI"].first()
                exit_rsi = pattern_data.groupby(pattern_type)["RSI"].last()

                f.write(f"\n{pattern_type.capitalize()} Patterns RSI:\n")
                f.write(f"  Average Entry RSI: {entry_rsi.mean():.1f}\n")
                f.write(f"  Average Exit RSI: {exit_rsi.mean():.1f}\n")
                f.write(f"  Highest Entry RSI: {entry_rsi.max():.1f}\n")
                f.write(f"  Lowest Entry RSI: {entry_rsi.min():.1f}\n")

    def _write_bearish_period_details(self, f, bearish_periods):
        """Write detailed analysis of each bearish period."""
        f.write("\n=== Detailed Bearish Period Analysis ===\n")

        for i, period in enumerate(bearish_periods, 1):
            f.write(f"\nPeriod {i}:\n")
            f.write(
                f"  Duration: {period['start_date'].strftime('%Y-%m-%d')} to {period['end_date'].strftime('%Y-%m-%d')}\n"
            )
            f.write(f"  Length: {period['length']} days\n")
            f.write(f"  Price Change: {period['price_change_pct']:+.2f}%\n")
            f.write(f"  Max Drawdown: {period['max_drawdown_pct']:+.2f}%\n")
            f.write(f"  Recovery: {period['recovery_pct']:+.2f}%\n")
            f.write(
                f"  Patterns Found: {period['total_patterns']} ({period['gain_patterns']} gains, {period['loss_patterns']} losses)\n"
            )
            f.write(f"  Trend Direction: {'Upward' if period['is_uptrend'] else 'Downward'}\n")

    def save_report(self, total_gains, total_losses, bearish_periods, output_file="report.txt"):
        """Write comprehensive analysis report."""
        logger.info(f"[crossover][{self.ticker}] Writing analysis report to {output_file}")

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path / output_file
        
        # Check if file exists and log overwrite
        if output_path.exists():
            logger.info(f"[crossover][{self.ticker}] Overwriting existing report file {output_path}")

        with open(output_path, "w") as f:
            # Write report sections in logical order
            self._write_executive_summary(f, total_gains, total_losses, bearish_periods)
            self._write_config_section(f)
            self._write_risk_metrics(f)

            # Add trade-by-trade analysis
            self._write_trade_analysis(f, "Winning", total_gains, "gain")
            self._write_trade_analysis(f, "Losing", total_losses, "loss")

            self._write_pattern_performance_summary(f)
            self._write_rsi_analysis(f)
            self._write_bearish_periods_analysis(f, bearish_periods)
            self._write_bearish_period_details(f, bearish_periods)

            # Add timestamp
            f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                "\nDisclaimer: Past performance does not guarantee future results. This analysis is for informational purposes only.\n"
            )

        logger.info(f"[crossover][{self.ticker}] Analysis report written to {output_path}")
