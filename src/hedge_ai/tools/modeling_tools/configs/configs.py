from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CrossoverConfig:
    """Configuration for crossover analysis."""
    lower_sma: int = 20
    upper_sma: int = 50
    take_profit: float = 0.10
    stop_loss: float = 0.05
    crossover_length: int = 10
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_underbought: int = 50
    start_date: str = (datetime.now() - timedelta(days=365 + (upper_sma * 2))).strftime("%Y-%m-%d") 
    end_date: str = datetime.now().strftime("%Y-%m-%d")