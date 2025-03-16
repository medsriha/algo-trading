import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from algo_trading.database.news.configs import AlphaVantageNewsDbConfig
import sqlite3
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
import json

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlphaVantageNewsExtractor:
    """Extract news and sentiment for tickers and topics from Alpha Vantage API."""
    
    # Valid sort options
    SORT_LATEST = "LATEST"
    SORT_EARLIEST = "EARLIEST"
    SORT_RELEVANCE = "RELEVANCE"
    
    # Valid topics
    TOPICS = {
        "blockchain": "blockchain",
        "earnings": "earnings",
        "ipo": "ipo",
        "mergers_and_acquisitions": "mergers_and_acquisitions",
        "financial_markets": "financial_markets",
        "economy_fiscal": "economy_fiscal",
        "economy_monetary": "economy_monetary",
        "economy_macro": "economy_macro",
        "energy_transportation": "energy_transportation",
        "finance": "finance",
        "life_sciences": "life_sciences",
        "manufacturing": "manufacturing",
        "real_estate": "real_estate",
        "retail_wholesale": "retail_wholesale",
        "technology": "technology"
    }
    
    def __init__(self, api_key: str = None, db_config: AlphaVantageNewsDbConfig = None):
        """Initialize the news extractor.
        
        Args:
            api_key: Alpha Vantage API key (defaults to env variable if None)
            db_config: Database configuration
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
            
        self.base_url = "https://www.alphavantage.co/query"
        self.db_config = db_config or AlphaVantageNewsDbConfig()
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with all required tables."""
        # Create directory if it doesn't exist
        self.db_config.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tables if they don't exist
        with sqlite3.connect(self.db_config.connection_string) as conn:
            # Enable foreign key support
            conn.execute("PRAGMA foreign_keys = ON")
            
            cursor = conn.cursor()
            
            # Create main articles table
            cursor.execute(self.db_config.get_create_table_sql())
            
            # Create topics table
            cursor.execute(self.db_config.get_create_topics_table_sql())
            
            # Create ticker sentiment table
            cursor.execute(self.db_config.get_create_ticker_sentiment_table_sql())
            
            # Create indexes for faster lookups
            try:
                # Index for the main table
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_article_time 
                    ON {self.db_config.articles_table_name} (time_published)
                """)
                
                # Index for the topics table
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_topic_url 
                    ON {self.db_config.topics_table_name} (url)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_topic_name 
                    ON {self.db_config.topics_table_name} (topic)
                """)
                
                # Index for the ticker sentiment table
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ticker_url 
                    ON {self.db_config.ticker_sentiment_table_name} (url)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ticker_symbol 
                    ON {self.db_config.ticker_sentiment_table_name} (ticker)
                """)
            except sqlite3.Error as e:
                logger.warning(f"Could not create indexes: {e}")
                
            conn.commit()
            logger.info(f"Initialized database at {self.db_config.db_path}")

    @staticmethod
    def _convert_time_published(time_published: str) -> str:
        """Convert Alpha Vantage time format to readable datetime string.
        
        Args:
            time_published: Time in format 'YYYYMMDDTHHMMSS'
            
        Returns:
            Formatted datetime string
        """
        try:
            dt = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.warning(f"Could not parse time format: {time_published}")
            return time_published
    
    @staticmethod
    def format_datetime(dt: datetime) -> str:
        """Format datetime object to Alpha Vantage format.
        
        Args:
            dt: Datetime object
            
        Returns:
            Formatted string in 'YYYYMMDDTHHMM' format
        """
        return dt.strftime('%Y%m%dT%H%M')
    
    def get_news_sentiment(
        self, 
        tickers: Union[str, List[str]] = None, 
        time_from: Union[str, datetime] = None,
        sort: str = SORT_LATEST,
    ) -> Dict:
        """Get news and sentiment based on specified parameters.
        
        : param tickers: Stock/crypto/forex symbols (comma-separated string or list)
        : param topics: News topics (comma-separated string or list)
        : param time_from: Start time in YYYYMMDDTHHMM format or datetime object
        : param sort: Sort order (LATEST, EARLIEST, or RELEVANCE)
        : param limit: Maximum number of results (up to 1000)
        """
        # Validate and prepare parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key
        }
        
        # Handle tickers parameter
        if tickers:
            if isinstance(tickers, list):
                tickers = ','.join(tickers)
            params['tickers'] = tickers
            
            
        # Handle time parameters
        if time_from:
            if isinstance(time_from, datetime):
                time_from = self.format_datetime(time_from)
            params['time_from'] = time_from
            
            
        # Handle sort parameter
        if sort not in [self.SORT_LATEST, self.SORT_EARLIEST, self.SORT_RELEVANCE]:
            logger.warning(f"Invalid sort parameter: {sort}. Using default: {self.SORT_LATEST}")
            sort = self.SORT_LATEST
        params['sort'] = sort

        
        # Log query parameters
        query_desc = []
        if 'tickers' in params:
            query_desc.append(f"tickers={params['tickers']}")
        if 'time_from' in params:
            query_desc.append(f"from {params['time_from']}")
            
        logger.info(f"Fetching news for {', '.join(query_desc)}")

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' not in data:
                logger.error(f"No news feed in response: {data}")
                return {}
                
            logger.info(f"Retrieved {len(data['feed'])} news articles")
            
            # Add readable datetime to each article
            for article in data['feed']:
                if 'time_published' in article:
                    article['time_published_original'] = article['time_published']  # Store original format
                    article['time_published'] = self._convert_time_published(article['time_published'])
            
            return data
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return {}
    
    def _is_article_duplicate(self, cursor, article: Dict) -> bool:
        """Check if an article already exists in the database.
        
        Args:
            cursor: SQLite cursor
            article: Article data
            
        Returns:
            True if article is a duplicate, False otherwise
        """
        if 'url' not in article:
            logger.warning(f"Article missing URL, cannot check for duplicates")
            return False
            
        # Get the article's URL
        url = article.get('url')
        
        # Check if the article exists in the main table
        cursor.execute(
            f"SELECT 1 FROM {self.db_config.articles_table_name} WHERE url = ?", 
            (url,)
        )
        if cursor.fetchone():
            return True
                
        return False
    
    def save_news_to_db(self, news_data: Dict, target_tickers: Union[str, List[str]], batch_size: int = 10):
        """Save news articles to database in batches with normalized topic and ticker data.
        
        Args:
            news_data: Dictionary containing news feed and metadata
            target_tickers: Ticker or list of tickers to store sentiment data for
            batch_size: Number of articles to process in each batch
            
        Returns:
            Number of articles inserted
        """
        if not news_data or 'feed' not in news_data or not news_data['feed']:
            logger.warning("No news articles to save")
            return 0
        
        # Normalize target_tickers to a list
        if isinstance(target_tickers, str):
            target_tickers = [target_tickers]
        target_tickers = [t.upper() for t in target_tickers]
        
        news_articles = news_data['feed']
        total_inserted = 0
        
        # Filter out articles that already exist in the database
        filtered_articles = []
        with sqlite3.connect(self.db_config.connection_string) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()
            
            for article in news_articles:
                # Skip articles without required fields
                if not article.get('url'):
                    logger.warning(f"Article missing URL, skipping: {article.get('title', 'Unknown title')}")
                    continue
                    
                # Check if article is a duplicate
                if not self._is_article_duplicate(cursor, article):
                    filtered_articles.append(article)
        
        if not filtered_articles:
            logger.info(f"All {len(news_articles)} articles already exist in database")
            return 0
            
        logger.info(f"Processing {len(filtered_articles)} new articles out of {len(news_articles)} total")
        
        # Process in batches
        for i in range(0, len(filtered_articles), batch_size):
            batch = filtered_articles[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(filtered_articles) + batch_size - 1) // batch_size}")
            
            with sqlite3.connect(self.db_config.connection_string) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()
                
                inserted_count = 0
                for article in batch:
                    try:
                        # 1. Insert into main articles table
                        article_data = {
                            'url': article.get('url'),
                            'title': article.get('title'),
                            'time_published': article.get('time_published'),
                            'authors': json.dumps(article.get('authors', [])),
                            'summary': article.get('summary'),
                            'banner_image': article.get('banner_image'),
                            'source': article.get('source'),
                            'category_within_source': article.get('category_within_source'),
                            'source_domain': article.get('source_domain'),
                            'overall_sentiment_score': article.get('overall_sentiment_score'),
                            'overall_sentiment_label': article.get('overall_sentiment_label')
                        }
                        
                        # Prepare columns and values for SQL
                        columns = ', '.join(article_data.keys())
                        placeholders = ', '.join(['?'] * len(article_data))
                        values = list(article_data.values())
                        
                        # Insert the article
                        insert_query = f"""
                            INSERT OR IGNORE INTO {self.db_config.articles_table_name} 
                            ({columns}) VALUES ({placeholders})
                        """
                        
                        cursor.execute(insert_query, values)
                        
                        # Check if the article was inserted
                        if cursor.rowcount > 0:
                            inserted_count += 1
                            
                            # 2. Insert topics data
                            if 'topics' in article and article['topics']:
                                for topic_data in article['topics']:
                                    topic_insert = {
                                        'url': article.get('url'),
                                        'topic': topic_data.get('topic'),
                                        'relevance_score': float(topic_data.get('relevance_score', 0))
                                    }
                                    
                                    topic_columns = ', '.join(topic_insert.keys())
                                    topic_placeholders = ', '.join(['?'] * len(topic_insert))
                                    topic_values = list(topic_insert.values())
                                    
                                    cursor.execute(f"""
                                        INSERT INTO {self.db_config.topics_table_name} 
                                        ({topic_columns}) VALUES ({topic_placeholders})
                                    """, topic_values)
                            
                            # 3. Insert ticker sentiment data - modified to only store target tickers
                            if 'ticker_sentiment' in article and article['ticker_sentiment']:
                                for ticker_data in article['ticker_sentiment']:
                                    if ticker_data.get('ticker', '').upper() in target_tickers:
                                        ticker_insert = {
                                            'url': article.get('url'),
                                            'ticker': ticker_data.get('ticker'),
                                            'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                            'ticker_sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                            'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label')
                                        }
                                        
                                        ticker_columns = ', '.join(ticker_insert.keys())
                                        ticker_placeholders = ', '.join(['?'] * len(ticker_insert))
                                        ticker_values = list(ticker_insert.values())
                                        
                                        cursor.execute(f"""
                                            INSERT INTO {self.db_config.ticker_sentiment_table_name} 
                                            ({ticker_columns}) VALUES ({ticker_placeholders})
                                        """, ticker_values)
                        
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting article {article.get('url')}: {e}")
                
                conn.commit()
                total_inserted += inserted_count
                logger.info(f"Batch complete: Saved {inserted_count} news articles to database")
        
        logger.info(f"Total: Saved {total_inserted} news articles to database")
        return total_inserted
    
    def extract_and_save_news_for_ticker(self, ticker: str, from_date: str, batch_size: int = 10):
        """Extract news for a specific ticker and save to database.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for news
            batch_size: Number of articles to process in each batch
            limit: Maximum number of results to retrieve
            
        Returns:
            Dictionary containing news feed and metadata
        """
        
        news_data = self.get_news_sentiment(
            tickers=ticker,
            time_from=from_date
        )
        self.save_news_to_db(news_data, target_tickers=ticker, batch_size=batch_size)
        return news_data
    
    def extract_and_save_news_for_topic(self, topic: str, days_back: int = 20, batch_size: int = 10, limit: int = 50):
        """Extract news for a specific topic and save to database.
        
        Args:
            topic: News topic (must be one of the valid topics)
            days_back: Number of days to look back for news
            batch_size: Number of articles to process in each batch
            limit: Maximum number of results to retrieve
            
        Returns:
            Dictionary containing news feed and metadata
        """
        if topic not in self.TOPICS.values():
            valid_topics = ", ".join(self.TOPICS.values())
            raise ValueError(f"Invalid topic: {topic}. Must be one of: {valid_topics}")
            
        time_from = datetime.now() - timedelta(days=days_back)
        
        news_data = self.get_news_sentiment(
            topics=topic,
            time_from=time_from
        )
        
        self.save_news_to_db(news_data, target_tickers=topic, batch_size=batch_size)
        return news_data
    
    def extract_and_save_news_for_multiple_tickers(self, tickers: List[str], days_back: int = 20, batch_size: int = 10, limit: int = 50):
        """Extract news mentioning multiple tickers and save to database.
        
        Args:
            tickers: List of stock ticker symbols
            days_back: Number of days to look back for news
            batch_size: Number of articles to process in each batch
            limit: Maximum number of results to retrieve
            
        Returns:
            Dictionary containing news feed and metadata
        """
        time_from = datetime.now() - timedelta(days=days_back)
        
        news_data = self.get_news_sentiment(
            tickers=tickers,
            time_from=time_from
        )
        
        self.save_news_to_db(news_data, target_tickers=tickers, batch_size=batch_size)
        return news_data
    
    def get_articles_by_topic(self, topic: str) -> List[Dict]:
        """Get articles related to a specific topic.
        
        :param topic: Topic to search for
        """
        with sqlite3.connect(self.db_config.connection_string) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT a.* 
                FROM {self.db_config.articles_table_name} a
                JOIN {self.db_config.topics_table_name} t ON a.url = t.url
                WHERE t.topic = ?
                ORDER BY a.time_published DESC
            """
            
            cursor.execute(query, (topic,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_articles_by_ticker(self, ticker: str, from_date: str, to_date: str) -> List[Dict]:
        """Get articles related to a specific ticker.
        
        :param ticker: Ticker symbol to search for
        
        """
        with sqlite3.connect(self.db_config.connection_string) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT a.* 
                FROM {self.db_config.articles_table_name} a
                JOIN {self.db_config.ticker_sentiment_table_name} ts ON a.url = ts.url
                WHERE ts.ticker = ?
                AND a.time_published BETWEEN ? AND ?
                ORDER BY a.time_published DESC
            """
            
            cursor.execute(query, (ticker, from_date, to_date))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_ticker_sentiment_for_article(self, url: str) -> List[Dict]:
        """Get ticker sentiment data for a specific article.
        
        Args:
            url: Article URL
            
        Returns:
            List of ticker sentiment dictionaries
        """
        with sqlite3.connect(self.db_config.connection_string) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT * 
                FROM {self.db_config.ticker_sentiment_table_name}
                WHERE url = ?
                ORDER BY relevance_score DESC
            """
            
            cursor.execute(query, (url,))
            return [dict(row) for row in cursor.fetchall()]
    
    def call_news_api(self, tickers: List[str], days_back: int = 5) -> Dict[str, List[Dict]]:
        """Get recent news articles for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            days_back: Number of days to look back
            
        Returns:
            Dictionary mapping tickers to their articles
        """
        result = {}
        now = datetime.now()
        from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d 23:59:59")  # Include full current day
        
        # Add rate limiting
        api_calls = 0
        API_CALL_LIMIT = 5
        API_CALL_INTERVAL = 60  # seconds

        for ticker in tickers:
            try:
                # Get articles from database first
                articles = self.get_articles_by_ticker(ticker, from_date, to_date)
               
                # If we don't have recent articles or today's articles, fetch new ones
                needs_update = (
                    not articles or 
                    not any(article['time_published'].startswith(now.strftime("%Y-%m-%d")) for article in articles)
                )

                if needs_update:
                    # Check API rate limiting
                    if api_calls >= API_CALL_LIMIT:
                        logger.info("Waiting for API rate limit reset...")
                        time.sleep(API_CALL_INTERVAL)
                        api_calls = 0

                    logger.info(f"Fetching new articles for {ticker}")
                    self.extract_and_save_news_for_ticker(ticker, from_date=from_date)
                    api_calls += 1
                    
                    # Get the updated articles with proper date range
                    articles = self.get_articles_by_ticker(ticker, from_date, to_date)

                # For each article, add its topics and ticker sentiment
                enriched_articles = []
                for article in articles:
                    try:
                        article_copy = dict(article)
                        article_copy['ticker_sentiment'] = self.get_ticker_sentiment_for_article(article['url'])
                        enriched_articles.append(article_copy)
                    except Exception as e:
                        logger.error(f"Error enriching article {article.get('url')}: {e}")
                        continue

                result[ticker] = enriched_articles

            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                result[ticker] = []  # Return empty list for failed ticker
                continue
        
        return result