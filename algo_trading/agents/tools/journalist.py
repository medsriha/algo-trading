from typing import Dict, List, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class Journalist:
    """
    A tool for fetching and analyzing news data for tickers.
    Provides news articles and sentiment analysis.
    """
    
    def __init__(self, news_extractor, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the Journalist tool.
        
        Args:
            news_extractor: News extraction service (e.g., AlphaVantageNewsExtractor)
            llm: Language model for enhanced analysis (optional)
        """
        self.news_extractor = news_extractor
        self.llm = llm
        
    def get_news_with_sentiment(self, tickers: List[str], days_back: int = 5) -> Dict[str, Any]:
        """
        Fetch recent news articles and perform sentiment analysis for the given tickers.
        
        Args:
            tickers: List of ticker symbols to fetch news for
            days_back: Number of days to look back for news
            
        Returns:
            Dictionary containing ticker news and sentiment analysis
        """
        logger.info(f"Journalist fetching news for {len(tickers)} tickers, looking back {days_back} days")
        
        result = {
            "ticker_news": {},
            "ticker_sentiment_analysis": {}
        }
        
        try:
            # Get recent news for all tickers
            ticker_news = self.news_extractor.get_recent_articles_for_tickers(tickers, days_back=days_back)
            
            # Add to result
            result["ticker_news"] = ticker_news
            
            # Calculate sentiment analysis for each ticker
            sentiment_analysis = {}
            
            for ticker, articles in ticker_news.items():
                if not articles:
                    sentiment_analysis[ticker] = {
                        "average_sentiment_score": None,
                        "sentiment_distribution": {},
                        "sentiment_trend": "No data",
                        "most_relevant_articles": []
                    }
                    continue
                
                # Extract sentiment data for this ticker
                sentiment_scores = []
                sentiment_labels = []
                article_relevance = []
                
                for article in articles:
                    # Find sentiment specific to this ticker
                    ticker_specific_sentiment = None
                    ticker_relevance = 0
                    
                    if 'ticker_sentiment' in article:
                        for sentiment in article['ticker_sentiment']:
                            if sentiment['ticker'] == ticker:
                                ticker_specific_sentiment = sentiment
                                try:
                                    ticker_relevance = float(sentiment.get('relevance_score', 0))
                                except (ValueError, TypeError):
                                    ticker_relevance = 0
                                break
                    
                    # Use overall sentiment if ticker-specific not available
                    if ticker_specific_sentiment:
                        try:
                            score = float(ticker_specific_sentiment.get('ticker_sentiment_score', 0))
                            sentiment_scores.append(score)
                            sentiment_labels.append(ticker_specific_sentiment.get('ticker_sentiment_label', 'Neutral'))
                            article_relevance.append((article, ticker_relevance))
                        except (ValueError, TypeError):
                            pass
                    elif 'overall_sentiment_score' in article:
                        try:
                            score = float(article.get('overall_sentiment_score', 0))
                            sentiment_scores.append(score)
                            sentiment_labels.append(article.get('overall_sentiment_label', 'Neutral'))
                            article_relevance.append((article, 0.1))  # Lower relevance for overall sentiment
                        except (ValueError, TypeError):
                            pass
                
                # Calculate average sentiment score
                avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
                
                # Calculate sentiment distribution
                distribution = {}
                for label in sentiment_labels:
                    if label in distribution:
                        distribution[label] += 1
                    else:
                        distribution[label] = 1
                
                # Sort by percentage
                total = len(sentiment_labels)
                distribution = {k: (v / total * 100) for k, v in distribution.items()} if total > 0 else {}
                
                # Determine sentiment trend
                if len(sentiment_scores) >= 3:
                    # Sort articles by time
                    sorted_articles = sorted(
                        [(a, s, r) for (a, r), s in zip(article_relevance, sentiment_scores)],
                        key=lambda x: x[0].get('time_published', ''),
                        reverse=True  # Most recent first
                    )
                    
                    recent_scores = [s for _, s, _ in sorted_articles[:min(5, len(sorted_articles))]]
                    if len(recent_scores) >= 3:
                        if recent_scores[0] > recent_scores[-1]:
                            trend = "Improving"
                        elif recent_scores[0] < recent_scores[-1]:
                            trend = "Declining"
                        else:
                            trend = "Stable"
                    else:
                        trend = "Insufficient data"
                else:
                    trend = "Insufficient data"
                
                # Get most relevant articles
                most_relevant = sorted(article_relevance, key=lambda x: x[1], reverse=True)
                most_relevant_articles = [
                    {
                        "title": a.get('title', 'No title'),
                        "time_published": a.get('time_published', 'Unknown date'),
                        "source": a.get('source', 'Unknown source'),
                        "relevance_score": r,
                        "url": a.get('url', '')
                    }
                    for a, r in most_relevant[:3]
                ]
                
                # Store sentiment analysis
                sentiment_analysis[ticker] = {
                    "average_sentiment_score": avg_score,
                    "sentiment_distribution": distribution,
                    "sentiment_trend": trend,
                    "most_relevant_articles": most_relevant_articles
                }
            
            # Add sentiment analysis to result
            result["ticker_sentiment_analysis"] = sentiment_analysis
            
            # Use LLM for enhanced analysis if available
            if self.llm:
                self._enhance_with_llm(result)
            
        except Exception as e:
            logger.error(f"Error in Journalist tool: {e}")
            
        return result
    
    def _enhance_with_llm(self, news_data: Dict[str, Any]) -> None:
        """
        Enhance news analysis using LLM if more sophisticated analysis is needed.
        This is a placeholder for future enhancements.
        
        Args:
            news_data: The news data to enhance
        """
        if not self.llm:
            return
            
        # This could be expanded to provide more sophisticated analysis
        # For example, summarizing key themes across articles or identifying
        # potential market impacts not captured by simple sentiment scores
        pass
    
    def get_news_summary(self, ticker: str, news_data: Dict[str, Any]) -> str:
        """
        Generate a concise summary of news for a specific ticker.
        
        Args:
            ticker: The ticker symbol
            news_data: News data from get_news_with_sentiment
            
        Returns:
            A formatted string with news summary
        """
        if not self.llm or ticker not in news_data["ticker_news"]:
            return self._format_basic_news_summary(ticker, news_data)
            
        # Use LLM to generate a more insightful summary
        articles = news_data["ticker_news"].get(ticker, [])
        sentiment = news_data["ticker_sentiment_analysis"].get(ticker, {})
        
        if not articles:
            return f"No recent news found for {ticker}."
            
        # Format news data for LLM
        news_context = f"Recent news for {ticker}:\n\n"
        for i, article in enumerate(articles[:5]):
            news_context += f"{i+1}. {article.get('title', 'No title')} ({article.get('time_published', 'Unknown date')})\n"
            news_context += f"   Source: {article.get('source', 'Unknown source')}\n"
            news_context += f"   Summary: {article.get('summary', 'No summary')}\n\n"
            
        # Add sentiment information
        if sentiment.get("average_sentiment_score") is not None:
            news_context += f"\nOverall sentiment: {sentiment.get('sentiment_trend', 'Unknown')} "
            news_context += f"(score: {sentiment.get('average_sentiment_score', 0):.2f})\n"
            
        # Create prompt for LLM
        prompt = ChatPromptTemplate.from_template(
            "You are a financial journalist summarizing recent news for investors.\n\n"
            "Please provide a concise, insightful summary of the following news for {ticker}. "
            "Focus on potential market impact and key developments. "
            "Keep your summary under 200 words.\n\n"
            "{news_context}"
        )
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"ticker": ticker, "news_context": news_context})
            return result.content.strip()
        except Exception as e:
            logger.error(f"Error generating news summary with LLM: {e}")
            return self._format_basic_news_summary(ticker, news_data)
    
    def _format_basic_news_summary(self, ticker: str, news_data: Dict[str, Any]) -> str:
        """Format a basic news summary without using LLM"""
        articles = news_data["ticker_news"].get(ticker, [])
        sentiment = news_data["ticker_sentiment_analysis"].get(ticker, {})
        
        if not articles:
            return f"No recent news found for {ticker}."
            
        summary = f"Recent news for {ticker}:\n\n"
        
        # Add up to 3 most recent articles
        for i, article in enumerate(articles[:3]):
            summary += f"{i+1}. {article.get('title', 'No title')} ({article.get('time_published', 'Unknown date')})\n"
            summary += f"   Source: {article.get('source', 'Unknown source')}\n"
            
        # Add sentiment information if available
        if sentiment.get("average_sentiment_score") is not None:
            summary += f"\nOverall sentiment: {sentiment.get('sentiment_trend', 'Unknown')} "
            summary += f"(score: {sentiment.get('average_sentiment_score', 0):.2f})\n"
            
        return summary
        
    def get_all_ticker_articles_with_sentiment(self, tickers: List[str], days_back: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all articles with their sentiment data for each ticker over the specified period.
        
        Args:
            tickers: List of ticker symbols to fetch news for
            days_back: Number of days to look back for news (default: 5)
            
        Returns:
            Dictionary mapping each ticker to a list of articles with sentiment data
        """
        logger.info(f"Fetching all articles with sentiment for {len(tickers)} tickers, looking back {days_back} days")
        
        result = {}
        
        try:
            # Get recent news for all tickers
            ticker_news = self.news_extractor.get_recent_articles_for_tickers(tickers, days_back=days_back)
            
            # Process each ticker's articles
            for ticker, articles in ticker_news.items():
                ticker_articles = []
                
                for article in articles:
                    # Extract basic article info
                    article_data = {
                        "title": article.get("title", "No title"),
                        "time_published": article.get("time_published", "Unknown date"),
                        "source": article.get("source", "Unknown source"),
                        "url": article.get("url", ""),
                        "summary": article.get("summary", ""),
                    }
                    
                    # Extract sentiment specific to this ticker
                    ticker_specific_sentiment = None
                    if 'ticker_sentiment' in article:
                        for sentiment in article['ticker_sentiment']:
                            if sentiment['ticker'] == ticker:
                                ticker_specific_sentiment = sentiment
                                break
                    
                    # Add sentiment data
                    if ticker_specific_sentiment:
                        article_data["sentiment"] = {
                            "score": ticker_specific_sentiment.get("ticker_sentiment_score", 0),
                            "label": ticker_specific_sentiment.get("ticker_sentiment_label", "Neutral"),
                            "relevance_score": ticker_specific_sentiment.get("relevance_score", 0)
                        }
                    elif 'overall_sentiment_score' in article:
                        article_data["sentiment"] = {
                            "score": article.get("overall_sentiment_score", 0),
                            "label": article.get("overall_sentiment_label", "Neutral"),
                            "relevance_score": 0.1  # Lower relevance for overall sentiment
                        }
                    else:
                        article_data["sentiment"] = {
                            "score": 0,
                            "label": "Unknown",
                            "relevance_score": 0
                        }
                    
                    ticker_articles.append(article_data)
                
                # Sort articles by publication date (most recent first)
                ticker_articles.sort(
                    key=lambda x: x.get("time_published", ""), 
                    reverse=True
                )
                
                # Calculate average sentiment per day
                daily_sentiment = {}
                
                for article in ticker_articles:
                    # Extract date from time_published (format might vary)
                    time_str = article.get("time_published", "")
                    date_str = time_str.split("T")[0] if "T" in time_str else time_str.split(" ")[0]
                    
                    # Skip if no valid date
                    if not date_str:
                        continue
                        
                    # Initialize day entry if not exists
                    if date_str not in daily_sentiment:
                        daily_sentiment[date_str] = {
                            "scores": [],
                            "articles": []
                        }
                    
                    # Add sentiment score and article reference
                    sentiment_score = float(article["sentiment"]["score"])
                    daily_sentiment[date_str]["scores"].append(sentiment_score)
                    daily_sentiment[date_str]["articles"].append(article)
                
                # Calculate average and categorize sentiment for each day
                daily_sentiment_summary = {}
                
                for date, data in daily_sentiment.items():
                    if not data["scores"]:
                        continue
                        
                    avg_score = sum(data["scores"]) / len(data["scores"])
                    
                    # Categorize sentiment based on score
                    if avg_score <= -0.35:
                        sentiment_category = "Bearish"
                    elif -0.35 < avg_score <= -0.15:
                        sentiment_category = "Somewhat-Bearish"
                    elif -0.15 < avg_score < 0.15:
                        sentiment_category = "Neutral"
                    elif 0.15 <= avg_score < 0.35:
                        sentiment_category = "Somewhat-Bullish"
                    else:  # avg_score >= 0.35
                        sentiment_category = "Bullish"
                    
                    daily_sentiment_summary[date] = {
                        "average_score": avg_score,
                        "sentiment_category": sentiment_category,
                        "article_count": len(data["articles"]),
                        "articles": data["articles"]
                    }
                
                # Add daily sentiment summary to the ticker's data
                ticker_articles_with_daily = {
                    "articles": ticker_articles,
                    "daily_sentiment": daily_sentiment_summary
                }
                
                result[ticker] = ticker_articles_with_daily
                
        except Exception as e:
            logger.error(f"Error fetching articles with sentiment: {e}")
            
        return result