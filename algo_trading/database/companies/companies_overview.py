import requests
import sqlite3
import os
import time
from typing import List, Dict, Any, Optional
from algo_trading.database.companies.configs import CompanyOverviewDbConfig


class CompanyOverviewExtractor:
    """Extracts company overview data from Alpha Vantage API and stores it in a database."""
    
    def __init__(self, config: CompanyOverviewDbConfig):
        """
        Initialize the extractor.
        
        Args:
            config: Configuration for the extractor
        """
        self.config = config
        self._ensure_db_exists()
    
    def _ensure_db_exists(self) -> None:
        """Ensure the database and table exist."""
        # Check if database file exists
        db_exists = os.path.exists(self.config.db_path)
        
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.config.table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print(f"Creating table {self.config.table_name} in database {self.config.db_path}")
            # Create the table if it doesn't exist
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                Symbol TEXT PRIMARY KEY,
                AssetType TEXT,
                Name TEXT,
                Description TEXT,
                CIK TEXT,
                Exchange TEXT,
                Currency TEXT,
                Country TEXT,
                Sector TEXT,
                Industry TEXT,
                Address TEXT,
                OfficialSite TEXT,
                FiscalYearEnd TEXT,
                LatestQuarter TEXT,
                MarketCapitalization TEXT,
                EBITDA TEXT,
                PERatio TEXT,
                PEGRatio TEXT,
                BookValue TEXT,
                DividendPerShare TEXT,
                DividendYield TEXT,
                EPS TEXT,
                RevenuePerShareTTM TEXT,
                ProfitMargin TEXT,
                OperatingMarginTTM TEXT,
                ReturnOnAssetsTTM TEXT,
                ReturnOnEquityTTM TEXT,
                RevenueTTM TEXT,
                GrossProfitTTM TEXT,
                DilutedEPSTTM TEXT,
                QuarterlyEarningsGrowthYOY TEXT,
                QuarterlyRevenueGrowthYOY TEXT,
                AnalystTargetPrice TEXT,
                AnalystRatingStrongBuy TEXT,
                AnalystRatingBuy TEXT,
                AnalystRatingHold TEXT,
                AnalystRatingSell TEXT,
                AnalystRatingStrongSell TEXT,
                TrailingPE TEXT,
                ForwardPE TEXT,
                PriceToSalesRatioTTM TEXT,
                PriceToBookRatio TEXT,
                EVToRevenue TEXT,
                EVToEBITDA TEXT,
                Beta TEXT,
                "52WeekHigh" TEXT,
                "52WeekLow" TEXT,
                "50DayMovingAverage" TEXT,
                "200DayMovingAverage" TEXT,
                SharesOutstanding TEXT,
                DividendDate TEXT,
                ExDividendDate TEXT,
                LastUpdated TEXT,
                DateAdded TEXT
            )
            ''')
        else:
            print(f"Table {self.config.table_name} already exists in database {self.config.db_path}")
            
            # Check if the DateAdded column exists, add it if it doesn't
            cursor.execute(f"PRAGMA table_info({self.config.table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'DateAdded' not in columns:
                print(f"Adding DateAdded column to {self.config.table_name}")
                cursor.execute(f"ALTER TABLE {self.config.table_name} ADD COLUMN DateAdded TEXT")
        
        conn.commit()
        conn.close()
    
    def fetch_company_overview(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company overview data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company overview data or None if the request failed
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.config.api_key
        }
        
        try:
            response = requests.get(self.config.api_url, params=params)
            data = response.json()
            
            # Check if the response contains an error message
            if 'Error Message' in data or 'Information' in data or not data:
                print(f"Error fetching data for {ticker}: {data}")
                return None
                
            return data
        except Exception as e:
            print(f"Exception fetching data for {ticker}: {e}")
            return None
    
    def store_company_overview(self, data: Dict[str, Any]) -> None:
        """
        Store company overview data in the database.
        
        Args:
            data: Company overview data
        """
        if not data or 'Symbol' not in data:
            print("Invalid data, skipping")
            return
            
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        # Create a copy of the data to avoid modifying the original
        data_to_store = data.copy()
        
        today_date = time.strftime('%Y-%m-%d')

        # Add today's date to the DateAdded column if it's a new record
        cursor.execute(f"SELECT Symbol FROM {self.config.table_name} WHERE Symbol = ?", (data_to_store['Symbol'],))
        if not cursor.fetchone():
            data_to_store['DateAdded'] = today_date
        
        # Prepare column names and placeholders for the SQL query
        # Wrap column names in quotes to handle special characters and names starting with numbers
        columns = ', '.join([f'"{k}"' for k in data_to_store.keys()])
        placeholders = ', '.join(['?' for _ in data_to_store])
        values = tuple(data_to_store.values())
        
        # Insert or replace the data
        cursor.execute(f'''
        INSERT OR REPLACE INTO {self.config.table_name} ({columns})
        VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()
        
        print(f"Stored data for {data['Symbol']} with date {today_date}")
    
