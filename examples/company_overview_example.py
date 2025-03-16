#!/usr/bin/env python3
"""
Example script demonstrating how to use the CompanyOverviewExtractor.
"""

import os
import sys
import argparse

# Add the parent directory to the path so we can import the algo_trading package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algo_trading.database.companies.configs import CompanyOverviewDbConfig
from algo_trading.database.companies.companies_overview import CompanyOverviewExtractor


def main():
    parser = argparse.ArgumentParser(description='Fetch company overview data from Alpha Vantage API')
    parser.add_argument('--api-key', type=str, default=os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo'),
                        help='Alpha Vantage API key (default: ALPHA_VANTAGE_API_KEY environment variable or "demo")')
    parser.add_argument('--db-path', type=str, default='companies.db',
                        help='Path to the SQLite database (default: companies.db)')
    parser.add_argument('--tickers', type=str, nargs='+', default=['IBM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                        help='List of ticker symbols to process (default: IBM, AAPL, MSFT, GOOGL, AMZN)')
    args = parser.parse_args()

    # Create the configuration
    config = CompanyOverviewDbConfig(
        api_key=args.api_key,
        db_path=args.db_path
    )

    # Create the extractor
    extractor = CompanyOverviewExtractor(config)

    # Process the tickers
    print(f"Processing {len(args.tickers)} tickers: {', '.join(args.tickers)}")
    extractor.process_tickers(args.tickers)

    # Get all stored tickers
    stored_tickers = extractor.get_stored_tickers()
    print(f"Stored tickers: {stored_tickers}")


if __name__ == '__main__':
    main() 