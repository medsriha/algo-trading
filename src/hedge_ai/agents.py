from typing import Annotated, Literal, TypedDict
from langgraph.graph import Graph, START, END

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from hedge_ai.database.extract_crossovers import GetCandidateCrossovers
from hedge_ai.database.config.crossovers_config import DatabaseCrossoversConfig

from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("crossover_agent.log"),  # Output to file
    ],
)

logger = logging.getLogger(__name__)


class RiskProfile(BaseModel):
    """Risk profile parameters for filtering crossovers"""

    min_return: Annotated[float, Field(gt=0)] = 15.0  # Minimum return in percentage
    min_number_gains: Annotated[int, Field(gt=0)] = 3  # Minimum number of gains
    max_number_losses: Annotated[int, Field(ge=0)] = 2  # Maximum number of losses
    all_bearish_uptrend: bool = True  # All crossovers must be bearish uptrend


class State(TypedDict):
    input: str
    risk_level: str
    risk_profile: RiskProfile
    crossovers: list[str]


RiskLevel = Literal["conservative", "moderate", "aggressive"]

RISK_PROMPT = """You are a financial risk assessment expert. Based on the user's input, categorize their risk tolerance into one of three levels: conservative, moderate, or aggressive.

User input: {user_input}

Respond with ONLY ONE of these three words: conservative, moderate, or aggressive

Response:"""


def create_crossover_agent(db_config: DatabaseCrossoversConfig, llm: ChatOpenAI) -> Graph:
    """Creates an agent that processes risk appetite and extracts matching crossovers"""

    def chatbot(state: State) -> State:
        """Converts natural language input into a standardized risk level"""
        logger.info(f"Processing user input: {state['input']}")
        prompt = ChatPromptTemplate.from_template(RISK_PROMPT)
        chain = prompt | llm
        result = chain.invoke({"user_input": state["input"]})
        risk_level = result.content.strip().lower()

        if risk_level not in ("conservative", "moderate", "aggressive"):
            logger.error(f"Invalid risk level returned by LLM: {risk_level}")
            raise ValueError("Invalid risk level returned by LLM")

        return {"risk_level": risk_level}

    def process_risk_input(state: State) -> State:
        """Converts risk level to specific parameters"""
        logger.info(f"Processing risk level: {state['risk_level']}")
        if state["risk_level"] == "conservative":
            return {
                "risk_profile": RiskProfile(
                    min_return=10.0, min_number_gains=5, max_number_losses=1, all_bearish_uptrend=True
                )
            }
        elif state["risk_level"] == "moderate":
            return {"risk_profile": RiskProfile()}  # Use defaults
        elif state["risk_level"] == "aggressive":
            return {
                "risk_profile": RiskProfile(
                    min_return=25.0, min_number_gains=2, max_number_losses=3, all_bearish_uptrend=True
                )
            }
        else:
            raise ValueError("Invalid risk level")

    def extract_crossovers(state: State) -> list[str]:
        """Extracts crossover candidates based on risk profile"""
        extractor = GetCandidateCrossovers(
            min_return=state["risk_profile"].min_return,
            min_number_gains=state["risk_profile"].min_number_gains,
            max_number_losses=state["risk_profile"].max_number_losses,
            all_bearish_uptrend=state["risk_profile"].all_bearish_uptrend,
            config=db_config,
        )
        logger.info(f"Extracting crossovers with profile: {state['risk_profile']}")
        return {"crossovers": extractor.retrieve_candidates()}

    def get_ticker_latest_news(state: State) -> State:
        """Gets the latest news for the tickers in the crossovers"""
        return {"news": f"Very good news for {state['crossovers']}"}

    def news_router(state: State) -> Literal["get_ticker_latest_news", END]:
        """Routes to the appropriate news node based on the state"""
        if state["crossovers"]:
            return "get_ticker_latest_news"
        return END

    # Create and connect the graph
    graph = Graph()
    graph.add_node("chatbot", chatbot)
    graph.add_node("process_risk", process_risk_input)
    graph.add_node("extract_crossovers", extract_crossovers)
    graph.add_node("get_ticker_latest_news", get_ticker_latest_news)

    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", "process_risk")
    graph.add_edge("process_risk", "extract_crossovers")
    graph.add_edge("extract_crossovers", "get_ticker_latest_news")
    graph.add_conditional_edges("extract_crossovers", news_router)
    graph.add_edge("get_ticker_latest_news", END)

    return graph


def get_crossover_recommendations(db_config: DatabaseCrossoversConfig, llm: ChatOpenAI) -> list[str]:
    """
    Helper function to get crossover recommendations based on natural language risk description

    Args:
        user_input: Natural language description of risk tolerance
        db_config: Database configuration
        llm: Language model instance

    Returns:
        List of ticker symbols matching the risk criteria
    """
    graph = create_crossover_agent(db_config, llm)
    return graph.compile()


l = get_crossover_recommendations(
    DatabaseCrossoversConfig(db_name="crossovers.db", table_name="crossovers"), ChatOpenAI(model="gpt-4o-mini")
)

print(l.invoke({"input": "I am an aggressive investor"}))
