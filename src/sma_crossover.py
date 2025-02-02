import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
from get_data import StockDataFetcher
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            
        required_columns = {'timestamp', 'close'}
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"Dataframe must contain columns: {required_columns}")

        self.ticker = ticker
        self.config = config or CrossoverConfig(upper_sma=50, lower_sma=20, stop_loss=0.05, take_profit=0.1, pattern_length=10)
        
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
        logger.info(f"[crossover][{self.ticker}] Reading {dataframe.shape[0]} rows of data")
        try:
            # Make a copy to avoid modifying the original
            self.df = dataframe.copy()
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Validate data
            if self.df['timestamp'].isnull().any():
                raise ValueError("Dataset contains null timestamps")
            if self.df['close'].isnull().any():
                raise ValueError("Dataset contains null close prices")
            
            # Sort by timestamp
            self.df.sort_values('timestamp', inplace=True)
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
        self.df['SMA_lower'] = self.df['close'].rolling(window=self.config.lower_sma).mean()
        self.df['SMA_upper'] = self.df['close'].rolling(window=self.config.upper_sma).mean()
        self.df['SMA_lower_below_upper'] = self.df['SMA_lower'] < self.df['SMA_upper']

    def find_crossover_patterns(self):
        """Find and group crossover patterns in the data."""
        logger.info(f"[crossover][{self.ticker}] Finding crossover patterns")
        
        # Calculate SMAs first
        self._calculate_sma()
        
        # Initialize pattern columns
        self.df['pattern_group_buy'] = 0
        self.df['pattern_group_ditch'] = 0
        
        buy_pattern_count = 0
        ditch_pattern_count = 0
        in_pattern = False
        current_pattern_start = None
        initial_price = None
        
        # Iterate through the data to identify patterns
        for i in range(len(self.df) - 1):  # Subtract 1 to ensure we have a next day
            current_price = self.df.iloc[i]['close']
            next_day_price = self.df.iloc[i + 1]['close']  # Price for the next day
            
            if not in_pattern and self.df.iloc[i]['SMA_lower_below_upper']:
                # Start of a new pattern
                in_pattern = True
                current_pattern_start = i
                initial_price = current_price
                pattern_days = 1
            
            elif in_pattern:
                pattern_days = i - current_pattern_start + 1
                price_change = (current_price - initial_price) / initial_price
                
                # Check if we should end the pattern
                end_pattern = False
                pattern_type = None
                
                # Check pattern length condition
                if pattern_days >= self.config.pattern_length:
                    end_pattern = True
                    pattern_type = 'buy'
                
                # Check profit/loss conditions
                if price_change >= self.config.take_profit:
                    end_pattern = True
                    pattern_type = 'buy'
                elif price_change <= -self.config.stop_loss:
                    end_pattern = True
                    pattern_type = 'ditch'
                
                if end_pattern and i < len(self.df) - 1:  # Ensure we have a next day
                    # Mark the pattern in the dataframe, including the next day
                    pattern_range = slice(current_pattern_start, i + 2)  # +2 to include next day
                    if pattern_type == 'buy':
                        buy_pattern_count += 1
                        self.df.loc[self.df.index[pattern_range], 'pattern_group_buy'] = buy_pattern_count
                    else:  # ditch pattern
                        ditch_pattern_count += 1
                        self.df.loc[self.df.index[pattern_range], 'pattern_group_ditch'] = ditch_pattern_count
                    
                    # Reset pattern tracking
                    in_pattern = False
                    current_pattern_start = None
                    initial_price = None
        
        logger.info(f"[crossover][{self.ticker}] Found {buy_pattern_count} buy patterns and {ditch_pattern_count} ditch patterns")

    def save_dataframe(self, output_file="dataframe.csv"):
        """Save the dataframe to a csv file."""
        self.df.to_csv(self.output_path / output_file, index=False)
        logger.info(f"[crossover][{self.ticker}] Dataframe saved to {output_file}")

    @property
    def total_buy_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_buy"].nunique() - 1 # Subtract 1 to exclude 0
    
    @property
    def total_ditch_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_ditch"].nunique() - 1 # Subtract 1 to exclude 0
    

    def write_pattern_statistics(self, output_file="report.txt"):
        """Write pattern statistics to a file."""
        logger.info(f"[crossover][{self.ticker}] Writing pattern statistics to {output_file}")
        if "pattern_group_buy" not in self.df.columns or "pattern_group_ditch" not in self.df.columns:
            logger.error(f"[crossover][{self.ticker}] Pattern group columns not found. Please run find_crossover_patterns first.")
            return

        with open(self.output_path / output_file, 'w') as f:
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
                pattern_data = self.df[self.df['pattern_group_buy'] == pattern_id]
                if not pattern_data.empty:
                    entry_price = pattern_data['close'].iloc[0]
                    exit_price = pattern_data['close'].iloc[-1]
                    take_profit_price = entry_price * (1 + self.config.take_profit)
                    stop_loss_price = entry_price * (1 - self.config.stop_loss)
                    actual_return = ((exit_price - entry_price) / entry_price) * 100
                    
                    f.write(f"\nPattern {pattern_id}:\n")
                    f.write(f"  Entry Price: ${entry_price:.2f}\n")
                    f.write(f"  Exit Price: ${exit_price:.2f}\n")
                    f.write(f"  Take Profit Level (+"
                           f"{self.config.take_profit*100:.1f}%): ${take_profit_price:.2f}\n")
                    f.write(f"  Stop Loss Level (-"
                           f"{self.config.stop_loss*100:.1f}%): ${stop_loss_price:.2f}\n")
                    f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                    f.write(f"  Pattern Length: {len(pattern_data)} days\n")
                    f.write(f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                           f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n")
            
            # Ditch Patterns Price Analysis
            f.write("\nDitch Pattern Price Levels:\n")
            for pattern_id in range(1, self.total_ditch_patterns + 1):
                pattern_data = self.df[self.df['pattern_group_ditch'] == pattern_id]
                if not pattern_data.empty:
                    entry_price = pattern_data['close'].iloc[0]
                    exit_price = pattern_data['close'].iloc[-1]
                    take_profit_price = entry_price * (1 + self.config.take_profit)
                    stop_loss_price = entry_price * (1 - self.config.stop_loss)
                    actual_return = ((exit_price - entry_price) / entry_price) * 100
                    
                    f.write(f"\nPattern {pattern_id}:\n")
                    f.write(f"  Entry Price: ${entry_price:.2f}\n")
                    f.write(f"  Exit Price: ${exit_price:.2f}\n")
                    f.write(f"  Take Profit Level (+"
                           f"{self.config.take_profit*100:.1f}%): ${take_profit_price:.2f}\n")
                    f.write(f"  Stop Loss Level (-"
                           f"{self.config.stop_loss*100:.1f}%): ${stop_loss_price:.2f}\n")
                    f.write(f"  Actual Return: {actual_return:+.2f}%\n")
                    f.write(f"  Pattern Length: {len(pattern_data)} days\n")
                    f.write(f"  Date Range: {pattern_data['timestamp'].min().strftime('%Y-%m-%d')} "
                           f"to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')}\n")
            
            f.write("\n")  # Add spacing before next section

            # Basic pattern statistics
            total_buy_days = len(self.df[self.df['pattern_group_buy'] > 0])
            total_ditch_days = len(self.df[self.df['pattern_group_ditch'] > 0])
            unique_buy_patterns = self.df['pattern_group_buy'].nunique() - 1
            unique_ditch_patterns = self.df['pattern_group_ditch'].nunique() - 1
            
            f.write(f"=== Pattern Statistics for {self.ticker} ===\n")
            f.write(f"Analysis Period: {self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Total days with buy patterns: {total_buy_days}\n")
            f.write(f"Total days with ditch patterns: {total_ditch_days}\n")
            f.write(f"Number of unique buy patterns: {unique_buy_patterns}\n")
            f.write(f"Number of unique ditch patterns: {unique_ditch_patterns}\n\n")

            # Pattern length statistics for buy patterns
            f.write("=== Buy Pattern Length Analysis ===\n")
            buy_lengths = {}
            buy_profits = []
            for pattern_id in range(1, unique_buy_patterns + 1):
                pattern_data = self.df[self.df['pattern_group_buy'] == pattern_id]
                if not pattern_data.empty:
                    length = len(pattern_data)
                    buy_lengths[pattern_id] = length
                    start_price = pattern_data['close'].iloc[0]
                    end_price = pattern_data['close'].iloc[-1]
                    price_change = ((end_price - start_price) / start_price) * 100
                    buy_profits.append(price_change)
                    f.write(f"Pattern {pattern_id}: {length} days, {price_change:+.2f}% return "
                           f"({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to "
                           f"{pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n")
            
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
                f.write(f"  Take Profit Hits: {sum(1 for x in buy_profits if x >= self.config.take_profit * 100)}\n")
                f.write(f"  Stop Loss Hits: {sum(1 for x in buy_profits if x <= -self.config.stop_loss * 100)}\n")
                f.write(f"  Pattern Length Hits: {sum(1 for x in buy_lengths.values() if x >= self.config.pattern_length)}\n\n")

            # Pattern length statistics for ditch patterns
            f.write("=== Ditch Pattern Length Analysis ===\n")
            ditch_lengths = {}
            ditch_losses = []
            for pattern_id in range(1, unique_ditch_patterns + 1):
                pattern_data = self.df[self.df['pattern_group_ditch'] == pattern_id]
                if not pattern_data.empty:
                    length = len(pattern_data)
                    ditch_lengths[pattern_id] = length
                    start_price = pattern_data['close'].iloc[0]
                    end_price = pattern_data['close'].iloc[-1]
                    price_change = ((end_price - start_price) / start_price) * 100
                    ditch_losses.append(price_change)
                    f.write(f"Pattern {pattern_id}: {length} days, {price_change:+.2f}% return "
                           f"({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to "
                           f"{pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n")
            
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
                f.write(f"  Take Profit Hits: {sum(1 for x in ditch_losses if x >= self.config.take_profit * 100)}\n")
                f.write(f"  Stop Loss Hits: {sum(1 for x in ditch_losses if x <= -self.config.stop_loss * 100)}\n")
                f.write(f"  Pattern Length Hits: {sum(1 for x in ditch_lengths.values() if x >= self.config.pattern_length)}\n\n")
            
            # Monthly statistics
            f.write("=== Monthly Analysis ===\n")
            self.df['month'] = self.df['timestamp'].dt.to_period('M')
            monthly_buy = self.df[self.df['pattern_group_buy'] > 0].groupby('month')['pattern_group_buy'].nunique()
            monthly_ditch = self.df[self.df['pattern_group_ditch'] > 0].groupby('month')['pattern_group_ditch'].nunique()
            
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
            self.df['season'] = self.df['timestamp'].dt.month % 12 // 3
            season_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
            self.df['season'] = self.df['season'].map(season_names)
            
            seasonal_buy = self.df[self.df['pattern_group_buy'] > 0].groupby('season')['pattern_group_buy'].nunique()
            seasonal_ditch = self.df[self.df['pattern_group_ditch'] > 0].groupby('season')['pattern_group_ditch'].nunique()
            
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
            f.write(f"Combined Return: {combined_return:+.2f}%\n")
            f.write(f"Average Return per Pattern: {avg_pattern_return:+.2f}%\n")
            f.write(f"Total Number of Patterns: {total_patterns}\n\n")

            # Add timestamp of analysis
            f.write(f"\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"[crossover][{self.ticker}] Statistics written to {output_file}")


    def plot_analysis(self, output_dir="plots"):
        """Create and save the analysis plot."""
        logger.info(f"[crossover][{self.ticker}] Creating analysis plot")
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(self.output_path / output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error creating directory {output_dir}: {str(e)}")
            raise

        try:
            plt.figure(figsize=(15, 8))

            # Plot price and SMAs
            plt.plot(self.df['timestamp'], self.df['close'], 
                    label='Daily Close Price', alpha=0.6, marker='o', markersize=3)
            plt.plot(self.df['timestamp'], self.df['SMA_lower'], 
                    label=f'{self.config.lower_sma} SMA', linewidth=2)
            plt.plot(self.df['timestamp'], self.df['SMA_upper'], 
                    label=f'{self.config.upper_sma} SMA', linewidth=2)

            # Get unique groups
            buy_groups = sorted(self.df[self.df['pattern_group_buy'] > 0]['pattern_group_buy'].unique())
            ditch_groups = sorted(self.df[self.df['pattern_group_ditch'] > 0]['pattern_group_ditch'].unique())
            
            # Plot buy patterns
            for group in buy_groups:
                group_data = self.df[self.df['pattern_group_buy'] == group]
                plt.scatter(group_data['timestamp'],
                          group_data['close'],
                          color='green',
                          marker='^',
                          label=f'Buy Pattern {group}' if group == buy_groups[0] else "",
                          zorder=5, s=100)
                
                # Add annotation for entry point
                entry_point = group_data.iloc[0]
                plt.annotate('Entry', 
                            xy=(entry_point['timestamp'], entry_point['close']),
                            xytext=(10, 30), textcoords='offset points',
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                # Add annotation for exit point
                exit_point = group_data.iloc[-1]
                plt.annotate('Exit', 
                            xy=(exit_point['timestamp'], exit_point['close']),
                            xytext=(10, -30), textcoords='offset points',
                            ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            # Plot ditch patterns
            for group in ditch_groups:
                group_data = self.df[self.df['pattern_group_ditch'] == group]
                plt.scatter(group_data['timestamp'],
                          group_data['close'],
                          color='red',
                          marker='v',
                          label=f'Ditch Pattern {group}' if group == ditch_groups[0] else "",
                          zorder=5, s=100)
                
                # Add annotation for entry point
                entry_point = group_data.iloc[0]
                plt.annotate('Entry', 
                            xy=(entry_point['timestamp'], entry_point['close']),
                            xytext=(-10, 30), textcoords='offset points',
                            ha='right', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                # Add annotation for exit point
                exit_point = group_data.iloc[-1]
                plt.annotate('Exit', 
                            xy=(exit_point['timestamp'], exit_point['close']),
                            xytext=(-10, -30), textcoords='offset points',
                            ha='right', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.title(f'{self.ticker} Daily Price with {self.config.lower_sma} and {self.config.upper_sma} Period SMAs')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.ticker}_sma_analysis_{timestamp}.png"
            filepath = self.output_path / output_dir / filename
            
            # Save the plot with extra space for legend
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"[crossover][{self.ticker}] Plot saved to {filepath}")
            
            # Clear the current figure to free memory
            plt.close()

        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error creating plot: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources and remove output directory if exists."""
        try:
            if hasattr(self, 'output_path') and self.output_path.exists():
                shutil.rmtree(self.output_path)
                logger.info(f"[crossover][{self.ticker}] Cleaned up output directory")
        except Exception as e:
            logger.error(f"[crossover][{self.ticker}] Error during cleanup: {str(e)}")

    def run(self):
        """Main execution function."""

        logger.info(f"Processing {self.ticker}")
        
        self.find_crossover_patterns()
        
        if self.total_buy_patterns > 0:
            self.write_pattern_statistics()
            self.plot_analysis()
            self.save_dataframe()
        else:
            self.cleanup()



if __name__ == "__main__":
    for ticker in ["NVDA", "TSLA", "AAPL", "AMZN", "GOOG", "MSFT", "META", "NFLX", "TSM", "WMT"]:
        fetcher = StockDataFetcher()
        df = fetcher.get_stock_data(ticker, timeframe="1D")
        sma_crossover_analyzer = SMACrossoverAnalyzer(ticker=ticker, dataframe=df)
        sma_crossover_analyzer.run()