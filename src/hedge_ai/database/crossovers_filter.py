import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hedge_ai.database.config import DatabaseCrossoversConfig, init_db
from pathlib import Path
from typing import Any
import logging
import sqlite3


class CandidateCrossoversFilter:
    def __init__(
        self,
        min_return: float = 20.0,
        min_crossover_count: int = 10,
        min_number_gains: int = 5,
        max_number_losses: int = 2,
        all_bearish_periods_uptrend: bool = True,
        config: DatabaseCrossoversConfig = None,
    ):
        self.min_return = min_return
        self.min_crossover_count = min_crossover_count
        self.min_number_gains = min_number_gains
        self.max_number_losses = max_number_losses
        self.all_bearish_periods_uptrend = all_bearish_periods_uptrend
        self.config = config

    def retrieve_candidates(self) -> list[str]:
        """Retrieve candidate stocks from the database.

        Returns:
            list[str]: List of ticker symbols meeting the filter criteria
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        sql = """
            SELECT DISTINCT ticker 
            FROM ?
            WHERE 
                crossover_count >= ?
                AND total_gains >= ?
                AND total_losses <= ?
                AND all_bearish_periods_uptrend = ?
                AND combined_return >= ?
            ORDER BY combined_return DESC
        """

        try:
            cursor.execute(
                sql,
                (
                    self.config.table_name,
                    self.min_crossover_count,
                    self.min_number_gains,
                    self.max_number_losses,
                    self.all_bearish_periods_uptrend,
                    self.min_return,
                ),
            )

            # Fetch all results and extract ticker symbols
            candidates = [row[0] for row in cursor.fetchall()]

            return candidates

        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return []

        finally:
            conn.close()
