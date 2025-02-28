import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from algo_trading.database.crossovers.configs import DatabaseCrossoversConfig
from pathlib import Path
from typing import Optional
import logging
import sqlite3


class FindCandidateCrossovers:
    """
    This class is used to get the candidate crossovers from the database.
    Use the paramaters to gauge the risk and return of the candidate crossovers.
    """

    def __init__(
        self,
        min_return: float = 20.0,
        min_total_trades: int = 6,
        min_number_gains: int = 5,
        max_number_losses: int = 2,
        all_bearish_uptrend: bool = True,
        num_candidates: Optional[int] = 10,
        config: DatabaseCrossoversConfig = None,
    ):
        self.min_return = min_return
        self.min_total_trades = min_total_trades
        self.min_number_gains = min_number_gains
        self.max_number_losses = max_number_losses
        self.all_bearish_uptrend = all_bearish_uptrend
        self.num_candidates = num_candidates or 10
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized GetCandidateCrossovers with parameters: "
            f"min_return={min_return}, min_total_trades={min_total_trades}, "
            f"min_number_gains={min_number_gains}, max_number_losses={max_number_losses}, "
            f"all_bearish_uptrend={all_bearish_uptrend}"
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
            SELECT DISTINCT ticker 
            FROM {self.config.table_name}
            WHERE total_gains >= ?
                AND total_losses <= ?
                AND all_bearish_uptrend = ?
                AND combined_return >= ?
            ORDER BY combined_return DESC
            LIMIT ?
        """

        try:
            self.logger.debug("Executing SQL query to retrieve candidates")
            cursor.execute(
                sql,
                (
                    self.min_number_gains,
                    self.max_number_losses,
                    self.all_bearish_uptrend,
                    self.min_return,
                    self.num_candidates,
                ),
            )

            # Fetch all results and extract ticker symbols
            candidates = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"Retrieved {len(candidates)} candidate stocks")
            self.logger.debug(f"Candidates: {candidates}")

            return candidates

        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return []

        finally:
            self.logger.debug("Closing database connection")
            conn.close()
