from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DatabaseCrossoversConfig:
    """Database configuration settings."""

    db_name: str = "crossovers.db"
    table_name: str = "crossovers"
    db_dir: Path = Path(".")
    columns: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        """Set default columns if none provided."""
        if self.columns is None:
            self.columns = {
                "data_creation_date": {"type": "DATE", "constraints": "NOT NULL"},
                "timestamp_date_start": {"type": "DATE", "constraints": "NOT NULL"},
                "timestamp_date_end": {"type": "DATE", "constraints": "NOT NULL"},
                "data_source": {"type": "TEXT", "constraints": "NOT NULL"},
                "ticker": {"type": "TEXT", "constraints": "NOT NULL"},
                "total_trades": {"type": "INTEGER", "constraints": "NOT NULL"},
                "total_gains": {"type": "INTEGER", "constraints": "NOT NULL"},
                "total_losses": {"type": "INTEGER", "constraints": "NOT NULL"},
                "all_bearish_uptrend": {"type": "BOOLEAN", "constraints": "NOT NULL"},
                "combined_return": {"type": "REAL", "constraints": "NOT NULL"},
                "created_at": {"type": "TIMESTAMP", "constraints": "DEFAULT CURRENT_TIMESTAMP"},
            }

    @property
    def db_path(self) -> Path:
        """Get the full database path."""
        return self.db_dir / self.db_name

    @property
    def connection_string(self) -> str:
        """Get the database connection string."""
        return str(self.db_path.absolute())

    def get_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement based on columns configuration."""
        column_definitions = []
        for col_name, col_info in self.columns.items():
            col_def = f"{col_name} {col_info['type']}"
            if col_info.get("constraints"):
                col_def += f" {col_info['constraints']}"
            column_definitions.append(col_def)

        return f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {", ".join(column_definitions)}
            )
        """
