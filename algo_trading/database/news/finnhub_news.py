import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from algo_trading.database import FinnhubNewsDbConfig
import sqlite3
from typing import List, Dict, Any, Optional
import logging
from firecrawl import FirecrawlApp
import time
import json

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class FinnhubNewsExtractor:
    """Extract news for a ticker and save to SQLite database."""
    
    def __init__(self, api_key: str = None, db_config: FinnhubNewsDbConfig = None, firecrawl_api_key: str = None):
        """Initialize the news extractor.
        
        Args:
            api_key: Finnhub API key (defaults to env variable if None)
            db_config: Database configuration
            firecrawl_api_key: Firecrawl API key (defaults to env variable if None)
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_key:
            raise ValueError("Firecrawl API key is required")
            
        self.client = finnhub.Client(api_key=self.api_key)
        self.scraper = FirecrawlApp(api_key=firecrawl_key)

        self.db_config = db_config or FinnhubNewsDbConfig()
        self._init_db()
    
    def _init_db(self):
        """Initialize the database."""
        # Create directory if it doesn't exist
        self.db_config.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create table if it doesn't exist
        with sqlite3.connect(self.db_config.connection_string) as conn:
            cursor = conn.cursor()
            
            # Add additional columns for metadata and scraping status
            create_table_sql = self.db_config.get_create_table_sql()
            
            # Check if we need to add new columns
            if "scrape_failed" not in create_table_sql and "metadata" not in create_table_sql:
                # Modify the SQL to add our new columns
                create_table_sql = create_table_sql.replace(
                    "retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                    "scrape_failed BOOLEAN DEFAULT 0, "
                    "page_title TEXT, "
                    "page_description TEXT, "
                    "metadata TEXT"
                )
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info(f"Initialized database at {self.db_config.db_path}")

    @staticmethod
    def _convert_unix_to_datetime(unix_timestamp: int) -> str:
        """Convert Unix timestamp to readable datetime string.
        
        Args:
            unix_timestamp: Unix timestamp in seconds
            
        Returns:
            Formatted datetime string
        """
        return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    def get_company_news(self, ticker: str, days_back: int = 20) -> List[Dict]:
        """Get company news for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching news for {ticker} from {from_date} to {to_date}")
        
        try:
            news = self.client.company_news(ticker, _from=from_date, to=to_date)
            logger.info(f"Retrieved {len(news)} news articles for {ticker}")
            
            # Add readable datetime to each article
            for article in news:
                if 'datetime' in article:
                    article['datetime_unix'] = article['datetime']  # Store original unix timestamp
                    article['datetime'] = self._convert_unix_to_datetime(article['datetime'])
            
            return news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def scrape_article_content(self, url: str, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
        """Scrape the content of a news article using firecrawl.
        
        Args:
            url: URL of the news article
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            Dictionary with content and metadata, or empty dict with error flag if scraping fails
        """
        if not url:
            logger.warning("No URL provided for scraping")
            return {"content": "", "scrape_failed": True}
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping content from {url}")
                scrape_result = self.scraper.scrape_url(url, params={'formats': ['markdown']})
                
                # Extract useful metadata
                result = {
                    "content": scrape_result.get("markdown", ""),
                    "scrape_failed": False,
                    "page_title": None,
                    "page_description": None,
                    "metadata": None
                }
                
                # Extract metadata if available
                metadata = scrape_result.get("metadata", {})
                if metadata:
                    # Store the full metadata as JSON
                    result["metadata"] = json.dumps(metadata)
                    
                    # Extract specific useful fields
                    result["page_title"] = metadata.get("title", metadata.get("ogTitle", [None])[0] if isinstance(metadata.get("ogTitle"), list) else metadata.get("ogTitle"))
                    
                    # Handle description which might be in different formats
                    description = metadata.get("description", [])
                    if isinstance(description, list) and description:
                        result["page_description"] = description[0]
                    else:
                        result["page_description"] = metadata.get("ogDescription", None)
                
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed to scrape {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
        
        logger.error(f"Failed to scrape content from {url} after {max_retries} attempts")
        return {"content": "", "scrape_failed": True}
    
    def save_news_to_db(self, ticker: str, news_articles: List[Dict], batch_size: int = 10):
        """Save news articles to database in batches.
        
        Args:
            ticker: Stock ticker symbol
            news_articles: List of news articles
            batch_size: Number of articles to process in each batch
            
        Returns:
            Number of articles inserted
        """
        if not news_articles:
            logger.warning(f"No news articles to save for {ticker}")
            return 0
        
        total_inserted = 0
        
        # Filter out articles that already exist in the database
        filtered_articles = []
        with sqlite3.connect(self.db_config.connection_string) as conn:
            cursor = conn.cursor()
            
            for article in news_articles:
                if 'id' not in article:
                    logger.warning(f"Article missing ID, skipping: {article}")
                    continue
                    
                # Check if article already exists
                cursor.execute(f"SELECT 1 FROM {self.db_config.table_name} WHERE id = ?", (article['id'],))
                if cursor.fetchone() is None:
                    filtered_articles.append(article)
        
        if not filtered_articles:
            logger.info(f"All {len(news_articles)} articles for {ticker} already exist in database")
            return 0
            
        logger.info(f"Processing {len(filtered_articles)} new articles out of {len(news_articles)} total for {ticker}")
        
        # Process in batches
        for i in range(0, len(filtered_articles), batch_size):
            batch = filtered_articles[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(filtered_articles) + batch_size - 1) // batch_size} for {ticker}")
            
            with sqlite3.connect(self.db_config.connection_string) as conn:
                cursor = conn.cursor()
                
                # Prepare insert query
                non_default_columns = [col for col in self.db_config.columns if col != "retrieved_at"]
                placeholders = ", ".join(["?"] * len(non_default_columns))
                columns = ", ".join(non_default_columns)
                
                insert_query = f"""
                    INSERT OR IGNORE INTO {self.db_config.table_name} ({columns})
                    VALUES ({placeholders})
                """
                
                # Insert each article in the batch
                inserted_count = 0
                for article in batch:
                    # Ensure ticker is included
                    article_with_ticker = article.copy()
                    article_with_ticker['ticker'] = ticker
                    
                    # Scrape article content if URL is available
                    scrape_result = {"content": None, "scrape_failed": False, "page_title": None, "page_description": None, "metadata": None}
                    if 'url' in article and article['url']:
                        scrape_result = self.scrape_article_content(article['url'])
                    
                    # Merge the article data with scrape results
                    article_with_ticker.update(scrape_result)
                    
                    # Extract values in the correct order
                    values = []
                    for col in non_default_columns:
                        values.append(article_with_ticker.get(col, None))
                    
                    try:
                        cursor.execute(insert_query, values)
                        if cursor.rowcount > 0:
                            inserted_count += 1
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting article {article.get('id')}: {e}")
                
                conn.commit()
                total_inserted += inserted_count
                logger.info(f"Batch complete: Saved {inserted_count} news articles for {ticker} to database")
        
        logger.info(f"Total: Saved {total_inserted} news articles for {ticker} to database")
        return total_inserted
    
    def extract_and_save_news(self, ticker: str, days_back: int = 20, batch_size: int = 10):
        """Extract news for a ticker and save to database in batches.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for news
            batch_size: Number of articles to process in each batch
            
        Returns:
            List of news articles
        """
        news = self.get_company_news(ticker, days_back)
        self.save_news_to_db(ticker, news, batch_size)
        return news