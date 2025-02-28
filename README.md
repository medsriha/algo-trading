# Algo Trading

A comprehensive algorithmic trading platform for developing, testing, and deploying trading strategies.

## Overview

This platform provides tools for:
- Developing and backtesting trading strategies
- Analyzing market data
- Managing risk profiles
- Executing trades based on technical indicators
- Integrating with news and sentiment analysis

## Key Features

- **SMA Crossover Strategy**: Identifies stocks with specific moving average crossover patterns
- **Risk Management**: Configurable risk profiles (conservative, moderate, aggressive)
- **Data Integration**: Historical and real-time market data processing
- **Database Storage**: Persistent storage of strategy results and market data
- **Agent-based Architecture**: Flexible workflow for strategy execution

## Project Structure

```
algo_trading/
├── strategies/         # Trading strategies implementation
├── data_providers/     # Market data acquisition and processing
├── models/             # Machine learning and statistical models
├── database/           # Database connections and operations
├── scripts/            # Utility scripts for maintenance and setup
└── tests/              # Test suite
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see pyproject.toml)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algo-trading.git
cd algo-trading

# Install dependencies
pip install -e .
```

### Usage

```python
from algo_trading.core.engine import TradingEngine
from algo_trading.strategies.sma_crossover import SMACrossoverStrategy

# Initialize the trading engine
engine = TradingEngine()

# Create and add a strategy
strategy = SMACrossoverStrategy(
    lower_sma_length=20,
    upper_sma_length=50,
    take_profit=0.10,
    stop_loss=0.05
)
engine.add_strategy(strategy)

# Run the engine
engine.run()
```

## License

MIT

Strategies:
- SMA Crossover and trend analysis:
    - Find stock with bearish trend where the 20 SMA is below the 50 SMA.
    - The above must be part of a bullish trend
    - Buy when 20 SMA is below 50 SMA and the stock is trending upward for less than average upward trend length from historical data
    - Sell after 10 days trading or take profit at 10% gain or take loss at 5%

