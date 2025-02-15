import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

from hedge_ai.tools.modeling_tools.configs.configs import CrossoverConfig
logger = logging.getLogger(__name__)


class CrossoversPlotter:
    def __init__(self, df: pd.DataFrame, ticker: str, crossover_config: CrossoverConfig, output_path: Path):
        """Initialize the CrossoverPlotter class.

        Args:
            df: DataFrame containing price data
            ticker (str): Stock ticker symbol
            config: Configuration object containing parameters
            output_path (Path): Path to save output plots
        """
        self.df = df
        self.ticker = ticker
        self.crossover_config = crossover_config
        self.output_path = output_path

    def save_plot(self, bearish_periods, filename="plot.png"):
        """Create and save the analysis plot with RSI indicator."""
        logger.info(f"[crossover][{self.ticker}] Creating analysis plot")

        # Set the backend to Agg explicitly
        import matplotlib

        matplotlib.use("Agg")

        try:
            # Create figure with 2 subplots (price and RSI)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), height_ratios=[3, 1], gridspec_kw={"hspace": 0.3})

            # Plot price and SMAs on top subplot (ax1)
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
                label=f"{self.crossover_config.lower_sma} SMA",
                color="orange",
                linewidth=2,
            )
            ax1.plot(
                self.df["timestamp"],
                self.df["SMA_upper"],
                label=f"{self.crossover_config.upper_sma} SMA",
                color="green",
                linewidth=2,
            )

            # Plot buy patterns
            gain_groups = sorted(self.df[self.df["gain"] > 0]["gain"].unique())

            # Plot ditch patterns
            loss_groups = sorted(self.df[self.df["loss"] > 0]["loss"].unique())
            for group in gain_groups:
                # Add "Exit" text annotation only for the first point in each group
                entry_row = self.df[self.df["gain"] == group].iloc[0]
                ax1.annotate(
                    "Entry",
                    (entry_row["timestamp"], entry_row["close"]),
                    xytext=(0, -20),
                    textcoords="offset points",
                    arrowprops=dict(facecolor="green", shrink=0.05),
                    ha="center",
                    va="top",
                    color="green",
                    fontweight="bold",
                )

                exit_row = self.df[self.df["gain"] == group].iloc[-1]
                ax1.annotate(
                    "Exit",
                    (exit_row["timestamp"], exit_row["close"]),
                    xytext=(0, -20),
                    textcoords="offset points",
                    arrowprops=dict(facecolor="green", shrink=0.05),
                    ha="center",
                    va="top",
                    color="green",
                    fontweight="bold",
                )

            for group in loss_groups:
                # Add "Exit" text annotation only for the first point in each group
                entry_row = self.df[self.df["loss"] == group].iloc[0]
                ax1.annotate(
                    "Entry",
                    (entry_row["timestamp"], entry_row["close"]),
                    xytext=(0, -20),
                    textcoords="offset points",
                    arrowprops=dict(facecolor="red", shrink=0.05),
                    ha="center",
                    va="top",
                    color="red",
                    fontweight="bold",
                )

                exit_row = self.df[self.df["loss"] == group].iloc[-1]
                ax1.annotate(
                    "Exit",
                    (exit_row["timestamp"], exit_row["close"]),
                    xytext=(0, -20),
                    textcoords="offset points",
                    arrowprops=dict(facecolor="red", shrink=0.05),
                    ha="center",
                    va="top",
                    color="red",
                    fontweight="bold",
                )

            for period in bearish_periods:
                try:
                    # Get data for this period
                    start_date = period["start_date"]
                    end_date = period["end_date"]
                    period_data = self.df[(self.df["timestamp"] >= start_date) & (self.df["timestamp"] <= end_date)]

                    # Skip if we don't have enough data points
                    if len(period_data) < 2:
                        logger.warning(
                            f"[crossover][{self.ticker}] Insufficient data points for trend line in period {start_date} to {end_date}"
                        )
                        continue

                    # Calculate trend line using numpy's polyfit
                    x = np.arange(len(period_data))
                    y = period_data["close"].values

                    # Handle potential NaN or inf values
                    mask = np.isfinite(y)
                    if not np.any(mask):
                        logger.warning(
                            f"[crossover][{self.ticker}] No valid data points for trend line in period {start_date} to {end_date}"
                        )
                        continue

                    x = x[mask]
                    y = y[mask]

                    # Skip if we don't have enough valid points after filtering
                    if len(x) < 2:
                        logger.warning(
                            f"[crossover][{self.ticker}] Insufficient valid data points for trend line after filtering"
                        )
                        continue

                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)

                    # Plot trend line
                    trend_color = "lime" if period["is_uptrend"] else "red"
                    trend_style = "--"
                    ax1.plot(
                        period_data["timestamp"].iloc[mask],  # Use only valid timestamps
                        p(x),
                        color=trend_color,
                        linestyle=trend_style,
                        linewidth=2,
                        label=(
                            f"Trend ({'Up' if period['is_uptrend'] else 'Down'})"
                            if period == bearish_periods[0]
                            else ""
                        ),
                    )
                except Exception as e:
                    logger.warning(f"[crossover][{self.ticker}] Failed to plot trend line: {str(e)}")
                    continue

            # Plot RSI on bottom subplot (ax2)
            ax2.plot(self.df["timestamp"], self.df["RSI"], color="blue", label="RSI")

            # Add RSI overbought/oversold lines
            ax2.axhline(y=70, color="r", linestyle="--", alpha=0.5, label="Overbought (70)")
            ax2.axhline(y=30, color="g", linestyle="--", alpha=0.5, label="Oversold (30)")
            ax2.axhline(y=50, color="gray", linestyle="-", alpha=0.2)

            # Fill overbought/oversold regions
            ax2.fill_between(
                self.df["timestamp"], 70, self.df["RSI"], where=self.df["RSI"] >= 70, color="red", alpha=0.1
            )
            ax2.fill_between(
                self.df["timestamp"], 30, self.df["RSI"], where=self.df["RSI"] <= 30, color="green", alpha=0.1
            )

            # Highlight RSI signals
            overbought_points = self.df[self.df["RSI_overbought"]]
            oversold_points = self.df[self.df["RSI_oversold"]]

            ax2.scatter(
                overbought_points["timestamp"],
                overbought_points["RSI"],
                color="red",
                marker="v",
                s=50,
                label="Overbought Signal",
            )
            ax2.scatter(
                oversold_points["timestamp"],
                oversold_points["RSI"],
                color="green",
                marker="^",
                s=50,
                label="Oversold Signal",
            )

            # Configure RSI subplot
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("RSI")
            ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)

            # Configure main price subplot
            ax1.set_title(
                f"{self.ticker} Daily Price with {self.crossover_config.lower_sma} and {self.crossover_config.upper_sma} Period SMAs"
            )
            ax1.set_xlabel("")  # Remove x-label from top subplot
            ax1.set_ylabel("Price")
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3)

            # Save plot
            filepath = self.output_path
            filepath.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath / filename, dpi=300, bbox_inches="tight")
            logger.info(f"[crossover][{self.ticker}] Plot saved to {filepath}")

            # Clear the current figure and close to free memory
            plt.clf()
            plt.close(fig)

        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error creating plot: {str(e)}")
            raise
