import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil

class SMACrossoverAnalyzer:
    def __init__(self, ticker, dataframe, output_dir="data", upper_sma=50, lower_sma=20, logger=None):
        """Initialize the analyzer with data path and logger."""
        self.ticker = ticker
        self.upper_sma = upper_sma
        self.lower_sma = lower_sma

        # Setup logging
        self.logger = logger or self._setup_logger()

        # Read data
        self._read_data(dataframe)

        # Setup output path
        self.output_path = Path(output_dir) / self.ticker

        os.makedirs(self.output_path, exist_ok=True)

        
    def _setup_logger(self):
        """Setup default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def remove_directory(self):
        """Remove the output directory."""
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

    def _read_data(self, dataframe: pd.DataFrame) -> None:
        """Load and prepare the daily data."""
        self.logger.info(f"[crossover][{self.ticker}] Reading {dataframe.shape[0]} rows of data")
        try:
            self.df = dataframe
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_dataframe(self):
        return self.df
    
    def _calculate_sma(self):
        """Calculate SMAs on daily data."""
        self.logger.info(f"[crossover][{self.ticker}] Calculating SMAs")
        self.df['SMA_lower'] = self.df['close'].rolling(window=self.lower_sma).mean()
        self.df['SMA_upper'] = self.df['close'].rolling(window=self.upper_sma).mean()
        self.df['SMA_lower_below_upper'] = self.df['SMA_lower'] < self.df['SMA_upper']

    def find_crossover_patterns(self, pattern_length=10):
        """Calculate consecutive days counter and flag positions with unique pattern IDs.
        Each continuous pattern gets assigned a unique number to track frequency.
        pattern_group_buy: patterns shorter than pattern_length
        pattern_group_ditch: patterns longer than or equal to pattern_length"""

        self._calculate_sma()

        self.logger.info(f"[crossover][{self.ticker}] Finding crossover patterns with pattern_length={pattern_length}")
        consecutive_days = 0
        pattern_group_buy = 1   # Counter for patterns < pattern_length
        pattern_group_ditch = 1 # Counter for patterns >= pattern_length
        temp_positions = []     # Temporary storage for current pattern positions
        
        # Initialize pattern group columns
        self.df['pattern_group_buy'] = 0
        self.df['pattern_group_ditch'] = 0
        
        for i in range(len(self.df)):
            if self.df['SMA_lower_below_upper'].iloc[i]:
                consecutive_days += 1
                temp_positions.append(i)
            else:
                if consecutive_days > 0:
                    if consecutive_days < pattern_length:
                        # Assign buy pattern group
                        self.df.loc[temp_positions, 'pattern_group_buy'] = pattern_group_buy
                        self.logger.debug(f"[crossover][{self.ticker}] Found buy pattern {pattern_group_buy} with length {consecutive_days}")
                        pattern_group_buy += 1
                    else:
                        # Assign ditch pattern group
                        self.df.loc[temp_positions, 'pattern_group_ditch'] = pattern_group_ditch
                        self.logger.debug(f"[crossover][{self.ticker}] Found ditch pattern {pattern_group_ditch} with length {consecutive_days}")
                        pattern_group_ditch += 1
                
                # Reset tracking variables
                consecutive_days = 0
                temp_positions = []
        
        # Handle the last pattern if it exists
        if consecutive_days > 0:
            if consecutive_days < pattern_length:
                self.df.loc[temp_positions, 'pattern_group_buy'] = pattern_group_buy
                self.logger.debug(f"[crossover][{self.ticker}] Found buy pattern {pattern_group_buy} with length {consecutive_days}")
            else:
                self.df.loc[temp_positions, 'pattern_group_ditch'] = pattern_group_ditch
                self.logger.debug(f"[crossover][{self.ticker}] Found ditch pattern {pattern_group_ditch} with length {consecutive_days}")

        total_buy_patterns = pattern_group_buy - 1
        total_ditch_patterns = pattern_group_ditch - 1
        self.logger.info(f"[crossover][{self.ticker}] Found {total_buy_patterns} buy patterns and {total_ditch_patterns} ditch patterns")

    def save_dataframe(self, output_file="dataframe.csv"):
        """Save the dataframe to a csv file."""
        self.df.to_csv(output_file, index=False)
        self.logger.info(f"[crossover][{self.ticker}] Dataframe saved to {output_file}")

    @property
    def total_buy_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_buy"].nunique() - 1 # Subtract 1 to exclude 0
    
    @property
    def total_ditch_patterns(self):
        # How many patterns are there?
        return self.df["pattern_group_ditch"].nunique() - 1 # Subtract 1 to exclude 0
    

    def write_pattern_statistics(self, output_file="metrics.txt"):
        """Write pattern statistics to a file."""
        self.logger.info(f"[crossover][{self.ticker}] Writing pattern statistics to {output_file}")
        if "pattern_group_buy" not in self.df.columns or "pattern_group_ditch" not in self.df.columns:
            self.logger.error(f"[crossover][{self.ticker}] Pattern group columns not found. Please run find_crossover_patterns first.")
            return

        try:
            with open(self.output_path / output_file, 'w') as f:
                # Basic pattern statistics
                total_buy_days = len(self.df[self.df['pattern_group_buy'] > 0])
                total_ditch_days = len(self.df[self.df['pattern_group_ditch'] > 0])
                unique_buy_patterns = self.df['pattern_group_buy'].nunique() - 1  # Subtract 1 to exclude 0
                unique_ditch_patterns = self.df['pattern_group_ditch'].nunique() - 1  # Subtract 1 to exclude 0
                
                f.write(f"=== Pattern Statistics for {self.ticker} ===\n\n")
                f.write(f"Analysis Period: {self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}\n")
                f.write(f"Total days with buy patterns: {total_buy_days}\n")
                f.write(f"Total days with ditch patterns: {total_ditch_days}\n")
                f.write(f"Number of unique buy patterns: {unique_buy_patterns}\n")
                f.write(f"Number of unique ditch patterns: {unique_ditch_patterns}\n\n")

                # Pattern length statistics for buy patterns
                f.write("=== Buy Pattern Length Analysis ===\n")
                buy_lengths = {}
                for pattern_id in range(1, unique_buy_patterns + 1):
                    pattern_data = self.df[self.df['pattern_group_buy'] == pattern_id]
                    if not pattern_data.empty:
                        length = len(pattern_data)
                        buy_lengths[pattern_id] = length
                        f.write(f"Pattern {pattern_id}: {length} days ({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n")
                
                if buy_lengths:
                    avg_length = sum(buy_lengths.values()) / len(buy_lengths)
                    max_length = max(buy_lengths.values())
                    min_length = min(buy_lengths.values())
                    f.write(f"\nBuy Pattern Length Summary:\n")
                    f.write(f"  Average: {avg_length:.2f} days\n")
                    f.write(f"  Maximum: {max_length} days\n")
                    f.write(f"  Minimum: {min_length} days\n\n")

                # Pattern length statistics for ditch patterns
                f.write("=== Ditch Pattern Length Analysis ===\n")
                ditch_lengths = {}
                for pattern_id in range(1, unique_ditch_patterns + 1):
                    pattern_data = self.df[self.df['pattern_group_ditch'] == pattern_id]
                    if not pattern_data.empty:
                        length = len(pattern_data)
                        ditch_lengths[pattern_id] = length
                        f.write(f"Pattern {pattern_id}: {length} days ({pattern_data['timestamp'].min().strftime('%Y-%m-%d')} to {pattern_data['timestamp'].max().strftime('%Y-%m-%d')})\n")
                
                if ditch_lengths:
                    avg_length = sum(ditch_lengths.values()) / len(ditch_lengths)
                    max_length = max(ditch_lengths.values())
                    min_length = min(ditch_lengths.values())
                    f.write(f"\nDitch Pattern Length Summary:\n")
                    f.write(f"  Average: {avg_length:.2f} days\n")
                    f.write(f"  Maximum: {max_length} days\n")
                    f.write(f"  Minimum: {min_length} days\n\n")
                
                # Monthly statistics
                f.write("=== Monthly Analysis ===\n")
                self.df['month'] = self.df['timestamp'].dt.to_period('M')
                monthly_buy = self.df[self.df['pattern_group_buy'] > 0].groupby('month')['pattern_group_buy'].nunique()
                monthly_ditch = self.df[self.df['pattern_group_ditch'] > 0].groupby('month')['pattern_group_ditch'].nunique()
                
                f.write("Buy Patterns:\n")
                f.write(f"  Average per month: {monthly_buy / 12:.2f}\n")
                f.write("Most active months:\n")
                for month, count in monthly_buy.nlargest(3).items():
                    f.write(f"  {month}: {count} patterns\n")
                
                f.write("\nDitch Patterns:\n")
                f.write(f"  Average per month: {monthly_ditch / 12:.2f}\n")
                f.write("Most active months:\n")
                for month, count in monthly_ditch.nlargest(3).items():
                    f.write(f"  {month}: {count} patterns\n")
                f.write("\n")

                # Price movement analysis
                f.write("=== Price Movement Analysis ===\n")
                f.write("Buy Patterns:\n")
                for pattern_id in range(1, unique_buy_patterns + 1):
                    pattern_data = self.df[self.df['pattern_group_buy'] == pattern_id]
                    if not pattern_data.empty:
                        start_price = pattern_data['close'].iloc[0]
                        end_price = pattern_data['close'].iloc[-1]
                        price_change = ((end_price - start_price) / start_price) * 100
                        f.write(f"Pattern {pattern_id}: {price_change:+.2f}% price change\n")
                
                f.write("\nDitch Patterns:\n")
                for pattern_id in range(1, unique_ditch_patterns + 1):
                    pattern_data = self.df[self.df['pattern_group_ditch'] == pattern_id]
                    if not pattern_data.empty:
                        start_price = pattern_data['close'].iloc[0]
                        end_price = pattern_data['close'].iloc[-1]
                        price_change = ((end_price - start_price) / start_price) * 100
                        f.write(f"Pattern {pattern_id}: {price_change:+.2f}% price change\n")

                # Seasonal analysis
                f.write("\n=== Seasonal Distribution ===\n")
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

                # Add timestamp of analysis
                f.write(f"\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            self.logger.info(f"[crossover][{self.ticker}] Statistics written to {output_file}")

        except Exception as e:
            self.logger.error(f"[crossover][{self.ticker}] Error writing statistics to file: {str(e)}")
            raise

    def plot_analysis(self, output_dir="plots"):
        """Create and save the analysis plot."""
        self.logger.info(f"[crossover][{self.ticker}] Creating analysis plot")
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(self.output_path / output_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"[crossover][{self.ticker}] Error creating directory {output_dir}: {str(e)}")
            raise

        try:
            plt.figure(figsize=(15, 8))

            # Plot price and SMAs
            plt.plot(self.df['timestamp'], self.df['close'], 
                    label='Daily Close Price', alpha=0.6, marker='o', markersize=3)
            plt.plot(self.df['timestamp'], self.df['SMA_lower'], 
                    label=f'{self.lower_sma} SMA', linewidth=2)
            plt.plot(self.df['timestamp'], self.df['SMA_upper'], 
                    label=f'{self.upper_sma} SMA', linewidth=2)

            # Plot buy patterns (green)
            buy_patterns = self.df[self.df['pattern_group_buy'] > 0]
            if not buy_patterns.empty:
                plt.scatter(buy_patterns['timestamp'],
                          buy_patterns['close'],
                          color='green', marker='^', 
                          label='Buy Pattern', zorder=5, s=100)

            # Plot ditch patterns (red)
            ditch_patterns = self.df[self.df['pattern_group_ditch'] > 0]
            if not ditch_patterns.empty:
                plt.scatter(ditch_patterns['timestamp'],
                          ditch_patterns['close'],
                          color='red', marker='v', 
                          label='Ditch Pattern', zorder=5, s=100)

            plt.title(f'{self.ticker} Daily Price with {self.lower_sma} and {self.upper_sma} Period SMAs')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.ticker}_sma_analysis_{timestamp}.png"
            filepath = self.output_path / output_dir / filename
            
            # Save the plot
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"[crossover][{self.ticker}] Plot saved to {filepath}")
            
            # Clear the current figure to free memory
            plt.close()

        except Exception as e:
            self.logger.error(f"[crossover][{self.ticker}] Error creating plot: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    for ticker in ["TSLA", "NVDA", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TSM", "WMT"]:
        print(f"Analyzing {ticker}")
        analyzer = SMACrossoverAnalyzer(ticker=ticker, dataframe=pd.read_csv(f"data/{ticker}/daily.csv"), upper_sma=50, lower_sma=20)
        analyzer.find_crossover_patterns()

        if analyzer.total_buy_patterns > 0:  # If we see N patterns, we save the dataframe and plots
            analyzer.write_pattern_statistics()
            analyzer.plot_analysis()
            analyzer.save_dataframe()
        else:
            analyzer.remove_directory()