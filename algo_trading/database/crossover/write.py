"""
This script runs the crossover algorithm on a list of tickers and saves the results to a database.
The data in the database will be accessible by an agent that will decide which stocks to invest in.
"""

from algo_trading.models.crossover import Crossover, CrossoverConfig
from algo_trading.database.crossover.configs import DatabaseCrossoverConfig


if __name__ == "__main__":
    # Algorithm configuration
    crossover_config = CrossoverConfig(
        lower_sma=20,
        upper_sma=50,
        take_profit=0.10,
        stop_loss=0.05,
        crossover_length=10,  # Max number of days holding the stock
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        rsi_underbought=50,  # Dont buy if RSI is over 50
    )
    # Where to save the results
    database_config = DatabaseCrossoverConfig(db_name="crossover.db", table_name="crossover")

    # Run the algorithm
    with open("/Users/deepset/algo-trading/tickers/large_cap_tickers.txt", "r") as f:
        tickers = f.read().splitlines()
    # Which tickers to run the algorithm on, strategy config and database config
    crossover = Crossover(tickers=tickers, crossover_config=crossover_config, database_config=database_config)
    crossover.run()
