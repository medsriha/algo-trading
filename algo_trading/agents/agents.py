from typing import Annotated, Literal, Union
from langgraph.graph import Graph, START, END
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from algo_trading.database import FindCandidateCrossovers, DatabaseCrossoversConfig
from algo_trading.strategy import CrossoverEntry
from algo_trading.models import CrossoverConfig
from algo_trading.agents.prompt_hub import ANALYST_PROMPT, RISK_PROFILE_PROMPT
from algo_trading.agents.configs import RiskProfile, State


import json
import re

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






def get_risk_profile(risk_level: Literal["conservative", "moderate", "aggressive"], llm: ChatOpenAI) -> RiskProfile:
    """
    Uses an LLM to dynamically determine risk profile parameters based on risk level.
    
    Args:
        risk_level: The risk level (conservative, moderate, or aggressive)
        llm: Language model instance for generating parameters
        
    Returns:
        RiskProfile instance with LLM-determined parameters
    """
    logger.info(f"Generating risk profile parameters for level: {risk_level}")
    
    # Use LLM to determine parameters based on risk level
    prompt = ChatPromptTemplate.from_template(RISK_PROFILE_PROMPT)
    chain = prompt | llm
    
    try:
        result = chain.invoke({"risk_level": risk_level})
        response_content = result.content.strip()
        
        
        # Try to find JSON pattern in the response
        json_match = re.search(r'({.*})', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            params = json.loads(json_str)
        else:
            logger.warning(f"No JSON pattern found in response: {response_content}")
            # If no JSON pattern found, try to parse the entire response
            params = json.loads(response_content)
        
        # Create RiskProfile with parameters from LLM
        profile = RiskProfile(
            min_return=float(params.get("min_return", 10.0)),
            max_number_losses=int(params.get("max_number_losses", 2)),
        )
        
        logger.info(f"Generated risk profile: {profile}")
        return profile
        
    except Exception as e:
        logger.error(f"Error generating risk profile with LLM: {e}")
        # Fallback to default profiles if LLM fails
        if risk_level == "conservative":
            return RiskProfile(
                min_return=7.0, max_number_losses=1
            )
        elif risk_level == "moderate":
            return RiskProfile()  # Use defaults
        elif risk_level == "aggressive":
            return RiskProfile(
                min_return=20.0, max_number_losses=3
            )
        else:
            raise ValueError(f"Invalid risk level: {risk_level}")
            



def create_crossover_agent(
    db_config: DatabaseCrossoversConfig, llm: ChatOpenAI, crossover_config: CrossoverConfig
) -> Graph:
    """Creates an agent that processes risk appetite and extracts matching crossovers"""
            
    def process_risk_input(state: State) -> State:
        """Converts risk level to specific parameters"""
        logger.info(f"Processing risk level: {state['risk_level']}")
        try:
            state["risk_profile"] = get_risk_profile(llm=llm, risk_level= state['risk_level'])
        except ValueError as e:
            logger.error(f"Error processing risk level: {e}")
            raise
        
        return state

    def find_crossovers_tickers(state: State) -> Union[State, END]:
        """Extracts crossover candidates based on risk profile"""
        extractor = FindCandidateCrossovers(
            min_return=state["risk_profile"].min_return,
            max_number_losses=state["risk_profile"].max_number_losses,
            config=db_config,
        )
        logger.info(f"Extracting crossovers with profile: {state['risk_profile']}")
        candidates = extractor.retrieve_candidates()

        if len(candidates) == 0 or not candidates:
            logger.warning("No candidates found. Exiting...")
            return END

        state["tickers"] = candidates
        return state

    def get_ticker_latest_analyst_report(state: State) -> State:
        """Gets the latest analyst report for the tickers in the crossovers"""
        reports = {}

        for ticker, date in state["tickers"]:
            with open(f"data/{ticker}/report.txt", "r") as f:
                report = f.read()
                reports[ticker] = {
                    "report": report,
                    "report_date": date
                }

        state["analyst_report"] = reports
        return state

    def is_entry(state: State) -> State:
        """Checks if the tickers are a valid entry point"""
        entry = CrossoverEntry(crossover_config=crossover_config)
        tickers = [ticker for ticker, _ in state["tickers"]]
        market_data = entry.get_market_data(tickers)
        results = entry.is_today_an_entry(market_data)
        
        state["entry"] = results
        return state

    def analyst(state: State) -> State:

        if state["analyst_report"]:
            reports = "\n".join(
                [f"Ticker: {ticker}\n\nReport: {report}" for ticker, report in state["analyst_report"].items()]
            )

            template = ChatPromptTemplate.from_template(ANALYST_PROMPT)
            chain = template | llm

            result = chain.invoke(
                {
                    "reports": reports,
                    "risk_level": state["risk_level"],
                    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    "end_date": datetime.now().strftime("%Y-%m-%d"),
                    "entry": state["entry"],
                }
            )
            return {"analyst": result.content.strip()}
        else:
            return {"analyst": "No information found"}

    def analyst_router(state: State) -> Literal["get_ticker_latest_analyst_report", END]:
        """Routes to the appropriate news node based on the state"""
        if len(state["tickers"]) > 0:
            return "get_ticker_latest_analyst_report"

        logger.warning("No candidate tickers found. Exiting...")        
        return END

    # Create and connect the graph
    graph = Graph()
    graph.add_node("process_risk", process_risk_input)
    graph.add_node("find_crossovers_tickers", find_crossovers_tickers)
    graph.add_node("get_ticker_latest_analyst_report", get_ticker_latest_analyst_report)
    graph.add_node("analyst", analyst)
    graph.add_node("is_entry", is_entry)

    # Define the main flow
    graph.add_edge(START, "process_risk")
    graph.add_edge("process_risk", "find_crossovers_tickers")
    graph.add_conditional_edges("find_crossovers_tickers", analyst_router)

    graph.add_edge("get_ticker_latest_analyst_report", "analyst")
    graph.add_edge("is_entry", "analyst")
    graph.add_edge("analyst", END)

    return graph


def get_ticker_recommendations(
    db_config: DatabaseCrossoversConfig, llm: ChatOpenAI, crossover_config: CrossoverConfig
) -> list[str]:
    """
    Helper function to get crossover recommendations based on natural language risk description

    Args:
        user_input: Natural language description of risk tolerance
        db_config: Database configuration
        llm: Language model instance

    Returns:
        List of ticker symbols matching the risk criteria
    """
    graph = create_crossover_agent(db_config, llm, crossover_config)
    return graph.compile()


l = get_ticker_recommendations(
    DatabaseCrossoversConfig(db_name="crossovers.db", table_name="crossovers"),
    ChatOpenAI(model="gpt-4o-mini"),
    CrossoverConfig(
        upper_sma=50,
        lower_sma=20,
        take_profit=0.10,
        stop_loss=0.05,
        crossover_length=10,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_underbought=50,
    )
)

res = l.invoke({"risk_level": "aggressive"})
print(res)
