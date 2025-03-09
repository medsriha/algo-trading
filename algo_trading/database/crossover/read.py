import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from algo_trading.database import DatabaseCrossoverConfig
from pathlib import Path
import logging
import sqlite3


class FindCandidateCrossover:
    """
    This class is used to get the candidate crossovers from the database.
    Use the paramaters to gauge the risk and return of the candidate crossovers.
    """

    def __init__(
        self,
        min_return: float = 20.0,
        min_total_trades: int = 5,
        max_number_losses: int = 2,
        config: DatabaseCrossoverConfig = None,
    ):
        self.min_return = min_return
        self.min_total_trades = min_total_trades
        self.max_number_losses = max_number_losses
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized FindCandidateCrossover with parameters: "
            f"min_return={min_return}, min_total_trades={min_total_trades}, "
            f"max_number_losses={max_number_losses}, "
    
        )

    def retrieve_candidates(self) -> list[str]:
        """Retrieve candidate stocks from the database.

        Returns:
            list[str]: List of ticker symbols meeting the filter criteria
        """
        self.logger.debug(f"Connecting to database at {self.config.db_path}")
        # Connect to SQLite database
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        sql = f"""
            WITH LatestData AS (
                SELECT ticker, MAX(data_creation_date) as latest_date
                FROM {self.config.table_name}
                GROUP BY ticker
            )
            SELECT DISTINCT c.ticker, c.combined_return
            FROM {self.config.table_name} c
            JOIN LatestData ld ON c.ticker = ld.ticker AND c.data_creation_date = ld.latest_date
            WHERE c.total_losses <= ?
                AND c.combined_return >= ?
                AND c.total_trades >= ?
            ORDER BY c.combined_return DESC
        """

        try:
            self.logger.debug("Executing SQL query to retrieve candidates with latest data")
            cursor.execute(
                sql,
                (
                    self.max_number_losses,
                    self.min_return,
                    self.min_total_trades
                ),
            )

            # Fetch all results and extract ticker symbols
            candidates = [row for row in cursor.fetchall()]
            self.logger.info(f"Retrieved {len(candidates)} candidate stocks")
            self.logger.debug(f"Candidates: {candidates}")

            return candidates

        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return []

        finally:
            self.logger.debug("Closing database connection")
            conn.close()
