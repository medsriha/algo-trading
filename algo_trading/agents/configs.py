from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict

class RiskProfile(BaseModel):
    """Risk profile parameters for filtering crossovers"""

    min_return: Annotated[float, Field(gt=0)] = 10.0  # Minimum return in percentage
    max_number_losses: Annotated[int, Field(ge=0)] = 2  # Maximum number of losses


class State(TypedDict):
    """State dictionary for the crossover agent workflow"""
    risk_level: Literal["conservative", "moderate", "aggressive"]
    risk_profile: RiskProfile
    tickers: list[str]
    analyst_report: dict[str, str]
    entry: dict[str, bool]
    analyst: str