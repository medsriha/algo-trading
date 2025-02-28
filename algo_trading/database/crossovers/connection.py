import sqlite3
import logging
from contextlib import contextmanager
from algo_trading.database.crossovers.configs import DatabaseCrossoversConfig
from pathlib import Path


@contextmanager
def get_db_connection(config: DatabaseCrossoversConfig):
    """Context manager for database connections.

    Args:
        config: Database configuration object

    Yields:
        sqlite3.Connection: Database connection object
    """
    conn = None
    try:
        conn = sqlite3.connect(config.connection_string)
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def check_database_exists(table_name: str) -> bool:
    """Check if the database file exists.

    Args:
        config: Database configuration object

    Returns:
        bool: True if database exists, False otherwise
    """
    return Path(table_name).exists()


def init_db(config: DatabaseCrossoversConfig):
    """Initialize the database with required tables.

    Args:
        config: Database configuration object
    """
    # Create directory if it doesn't exist
    config.db_dir.mkdir(parents=True, exist_ok=True)

    with get_db_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(config.get_create_table_sql())
        conn.commit()
