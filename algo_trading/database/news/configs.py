from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class FinnhubNewsDbConfig:
    """Database configuration for news data."""
    
    db_name: str = "news.db"
    table_name: str = "finnhub_news_articles"
    db_dir: Path = Path("/Users/deepset/algo-trading/warehouse")
    columns: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default columns if none provided."""
        if self.columns is None:
            self.columns = {
                "id": {"type": "INTEGER", "constraints": "PRIMARY KEY"},
                "ticker": {"type": "TEXT", "constraints": "NOT NULL"},
                "datetime": {"type": "INTEGER", "constraints": "NOT NULL"},
                "headline": {"type": "TEXT", "constraints": "NOT NULL"},
                "category": {"type": "TEXT", "constraints": ""},
                "source": {"type": "TEXT", "constraints": ""},
                "summary": {"type": "TEXT", "constraints": ""},
                "url": {"type": "TEXT", "constraints": ""},
                "image": {"type": "TEXT", "constraints": ""},
                "related": {"type": "TEXT", "constraints": ""},
                "retrieved_at": {"type": "TIMESTAMP", "constraints": "DEFAULT CURRENT_TIMESTAMP"},
                "content": {"type": "TEXT", "constraints": ""},
                "scrape_failed": {"type": "BOOLEAN", "constraints": "DEFAULT 0"},
                "page_title": {"type": "TEXT", "constraints": ""},
                "page_description": {"type": "TEXT", "constraints": ""},
                "metadata": {"type": "TEXT", "constraints": ""}
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

@dataclass
class AlphaVantageNewsDbConfig:
    """Database configuration for Alpha Vantage news data."""
    
    db_name: str = "news.db"
    db_dir: Path = Path("/Users/deepset/algo-trading/warehouse")
    
    # Main articles table
    articles_table_name: str = "alphavantage_news_articles"
    
    # Additional tables for normalized data
    topics_table_name: str = "alphavantage_news_topics"
    ticker_sentiment_table_name: str = "alphavantage_ticker_sentiment"
    
    columns: Dict[str, Dict[str, Any]] = None
    topics_columns: Dict[str, Dict[str, Any]] = None
    ticker_sentiment_columns: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default columns if none provided."""
        if self.columns is None:
            self.columns = {
                "url": {"type": "TEXT", "constraints": "PRIMARY KEY"},
                "title": {"type": "TEXT", "constraints": "NOT NULL"},
                "time_published": {"type": "TEXT", "constraints": ""},
                "authors": {"type": "TEXT", "constraints": ""},  # JSON array
                "summary": {"type": "TEXT", "constraints": ""},
                "banner_image": {"type": "TEXT", "constraints": ""},
                "source": {"type": "TEXT", "constraints": ""},
                "category_within_source": {"type": "TEXT", "constraints": ""},
                "source_domain": {"type": "TEXT", "constraints": ""},
                "overall_sentiment_score": {"type": "REAL", "constraints": ""},
                "overall_sentiment_label": {"type": "TEXT", "constraints": ""},
                "retrieved_at": {"type": "TIMESTAMP", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            }
            
        if self.topics_columns is None:
            self.topics_columns = {
                "id": {"type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                "url": {"type": "TEXT", "constraints": "NOT NULL"},
                "topic": {"type": "TEXT", "constraints": "NOT NULL"},
                "relevance_score": {"type": "REAL", "constraints": ""},
                "FOREIGN KEY(url)": {"type": "", "constraints": "REFERENCES " + self.articles_table_name + "(url) ON DELETE CASCADE"}
            }
            
        if self.ticker_sentiment_columns is None:
            self.ticker_sentiment_columns = {
                "id": {"type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                "url": {"type": "TEXT", "constraints": "NOT NULL"},
                "ticker": {"type": "TEXT", "constraints": "NOT NULL"},
                "relevance_score": {"type": "REAL", "constraints": ""},
                "ticker_sentiment_score": {"type": "REAL", "constraints": ""},
                "ticker_sentiment_label": {"type": "TEXT", "constraints": ""},
                "FOREIGN KEY(url)": {"type": "", "constraints": "REFERENCES " + self.articles_table_name + "(url) ON DELETE CASCADE"}
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
        """Generate CREATE TABLE SQL statement for the main articles table."""
        column_definitions = []
        for col_name, col_info in self.columns.items():
            col_def = f"{col_name} {col_info['type']}"
            if col_info.get("constraints"):
                col_def += f" {col_info['constraints']}"
            column_definitions.append(col_def)
        
        return f"""
            CREATE TABLE IF NOT EXISTS {self.articles_table_name} (
                {", ".join(column_definitions)}
            )
        """
    
    def get_create_topics_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement for the topics table."""
        column_definitions = []
        for col_name, col_info in self.topics_columns.items():
            col_def = f"{col_name} {col_info['type']}"
            if col_info.get("constraints"):
                col_def += f" {col_info['constraints']}"
            column_definitions.append(col_def)
        
        return f"""
            CREATE TABLE IF NOT EXISTS {self.topics_table_name} (
                {", ".join(column_definitions)}
            )
        """
    
    def get_create_ticker_sentiment_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement for the ticker sentiment table."""
        column_definitions = []
        for col_name, col_info in self.ticker_sentiment_columns.items():
            col_def = f"{col_name} {col_info['type']}"
            if col_info.get("constraints"):
                col_def += f" {col_info['constraints']}"
            column_definitions.append(col_def)
        
        return f"""
            CREATE TABLE IF NOT EXISTS {self.ticker_sentiment_table_name} (
                {", ".join(column_definitions)}
            )
        """