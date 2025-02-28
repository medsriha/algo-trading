"""
This script runs the crossover algorithm on a list of tickers and saves the results to a database.
The data in the database will be accessible by an agent that will decide which stocks to invest in.
"""

from algo_trading.models.crossovers import Crossover, CrossoverConfig
from algo_trading.database.crossovers.configs import DatabaseCrossoversConfig


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
    database_config = DatabaseCrossoversConfig(db_name="crossovers.db", table_name="crossovers")

    # Run the algorithm
    with open("/Users/deepset/algo-trading/tickers/large_cap.txt", "r") as f:
        tickers = f.read().splitlines()
    crossover = Crossover(tickers, crossover_config, database_config)
    crossover.run()
