from dataclasses import dataclass


@dataclass
class CrossoverConfig:
    lower_sma: int
    upper_sma: int
    stop_loss: float
    take_profit: float
    pattern_max_length: int
    output_dir: str = "../data"
