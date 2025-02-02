from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
import dotenv
import datetime
import pandas as pd
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class StockDataFetcher:
    """A class to fetch historical stock data from Alpaca API."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the StockDataFetcher with API credentials.
        
        :param api_key: Alpaca API key. If None, will look for API_KEY in environment
        :param secret_key: Alpaca secret key. If None, will look for SECRET_KEY in environment
        """
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("API_KEY")
        self.secret_key = secret_key or os.getenv("SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API credentials not found. Please provide them or set in environment.")
            
        self.client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)

    def get_stock_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = "1D",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for given symbol(s).

        :param symbols: Single stock symbol or list of symbols
        :param timeframe: Time interval ('1H' for hourly, '1D' for daily)
        :param start_date: Start date in 'YYYY-MM-DD' format. Defaults to start of current year
        :param end_date: End date in 'YYYY-MM-DD' format. Defaults to today
            
        Returns:
            pandas DataFrame with historical data
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Set default dates if not provided (Year to Date)
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
            
        # Map timeframe string to TimeFrame enum
        timeframe_map = {
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Invalid timeframe. Must be one of: {list(timeframe_map.keys())}")
            
        try:
            params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_map[timeframe],
                start=start_date,
                end=end_date,
                adjustment="all"
            )
            
            bars = self.client.get_stock_bars(params)
            df = bars.df.reset_index()
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")