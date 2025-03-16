from dataclasses import dataclass
import os


@dataclass
class CompanyOverviewDbConfig:
    """Configuration for the company overview database."""
    api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY")
    db_path: str = "/Users/deepset/algo-trading/warehouse/companies.db"
    table_name: str = "company_overview"
    api_url: str = "https://www.alphavantage.co/query"
    request_delay: int = 12  # Delay between API requests in seconds (Alpha Vantage free tier limit) 