from typing import  Literal, Union
from langgraph.graph import Graph, START, END
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from algo_trading.database import FindCandidateCrossover, DatabaseCrossoverConfig
from algo_trading.strategy import CrossoverEntry
from algo_trading.models import CrossoverConfig
from algo_trading.agents.configs import RiskProfile, State

import yaml
import json
import re

from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor

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

def load_prompts(file_path: str) -> dict:
    """Loads prompts from a YAML file"""
    with open(file_path, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts

prompts = load_prompts("/Users/deepset/algo-trading/algo_trading/agents/prompt_hub.yaml")


def get_risk_profile(risk_level: Literal["conservative", "moderate", "aggressive"], llm: ChatOpenAI) -> RiskProfile:
    """
    Uses an LLM to dynamically determine risk profile parameters based on risk level.
    
    :param risk_level: The risk level (conservative, moderate, or aggressive)
    :param llm: Language model instance for generating parameters
    """
    logger.info(f"Generating risk profile parameters for level: {risk_level}")
    
    # Use LLM to determine parameters based on risk level
    prompt = ChatPromptTemplate.from_template(prompts["RISK_PROFILE_PROMPT"]["template"])
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
    db_config: DatabaseCrossoverConfig, llm: ChatOpenAI, crossover_config: CrossoverConfig
) -> Graph:
    """Creates an agent that processes risk appetite and extracts matching crossover"""
            
    def process_risk_input(state: State) -> State:
        """Converts risk level to specific parameters"""
        logger.info(f"Processing risk level: {state['risk_level']}")
        try:
            state["risk_profile"] = get_risk_profile(llm=llm, risk_level= state['risk_level'])
        except ValueError as e:
            logger.error(f"Error processing risk level: {e}")
            raise
        
        return state

    def find_crossover_tickers(state: State) -> Union[State, END]:
        """Extracts crossover candidates based on risk profile"""
        extractor = FindCandidateCrossover(
            min_return=state["risk_profile"].min_return,
            max_number_losses=state["risk_profile"].max_number_losses,
            config=db_config,
        )
        logger.info(f"Extracting crossover with profile: {state['risk_profile']}")
        candidates = extractor.retrieve_candidates()

        if len(candidates) == 0 or not candidates:
            logger.warning("No candidates found. Exiting...")
            return END

        state["tickers"] = candidates
        return state

    def get_ticker_latest_analyst_report(state: State) -> State:
        """Get the latest analyst report for a ticker"""
        reports = {}

        for ticker in state["tickers_to_invest"]:

            with open(f"data/{ticker}/report.txt", "r") as f:
                report = f.read()
                reports[ticker] = report

        state["analyst_report"] = reports
        return state

    def is_entry(state: State) -> State:
        """Checks if the tickers are a valid entry point and returns the tickers that are valid"""
        candidates = state["tickers"]
        entry = CrossoverEntry(crossover_config=crossover_config)

        # Collect all valid entry tickers first
        tickers_to_invest = []
        tickers_not_invest = []
        
        try:
            # Extract just the ticker symbols from the candidates
            ticker_symbols = [ticker for ticker, _ in candidates]
            
            # Pass only the ticker symbols to get_market_data
            market_data = entry.get_market_data(ticker_symbols)
            results = entry.is_today_an_entry(market_data)

            valid_entries = []
            not_valid_entries = []

            for ticker, is_valid_entry in results.items():
                # Find the return value from the original tickers list
                return_value = None
                for t, ret_value in state["tickers"]:
                    if t == ticker:
                        return_value = ret_value
                        break
                
                if is_valid_entry:
                    valid_entries.append((ticker, return_value))
                else:
                    not_valid_entries.append((ticker, return_value))
            
            # Sort the valid entries by their return value (descending)
            valid_entries.sort(key=lambda x: x[1] if x[1] is not None else -float('inf'), reverse=True)
            
            # Take only up to max_tickers
            tickers_to_invest = [ticker for ticker, _ in valid_entries]
            tickers_not_invest = [ticker for ticker, _ in not_valid_entries]

            logger.info(f"Tickers to invest: {tickers_to_invest}")
        except Exception as e:
            logger.error(f"Error in is_entry function: {e}")
                
        state["tickers_to_invest"] = tickers_to_invest
        state["tickers_not_invest"] = tickers_not_invest

        return state

    def analyst(state: State) -> State:
        if state["analyst_report"]:
            results = {}
            
            def process_ticker(ticker_report_pair):
                ticker, report = ticker_report_pair

                context = f"Ticker: {ticker}\n\nReport: {report}"
                
                template = ChatPromptTemplate.from_template(prompts["ANALYST_PROMPT"]["template"])
                chain = template | llm
                
                result = chain.invoke(
                    {
                        "context": context,
                        "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                        "end_date": datetime.now().strftime("%Y-%m-%d")
                    }
                )
                return ticker, result.content.strip()
            
            # Use ThreadPoolExecutor to process all tickers in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                for ticker, content in executor.map(process_ticker, state["analyst_report"].items()):
                    results[ticker] = content

            return {
                "analyst": results,
                "not_entry_candidates": state["tickers_not_invest"],
                "entry_candidates": state["tickers_to_invest"],
            }
        
        return {"analyst": "No information found"}
    # Create and connect the graph
    graph = Graph()
    graph.add_node("process_risk", process_risk_input)
    graph.add_node("find_crossover_tickers", find_crossover_tickers)
    graph.add_node("get_ticker_latest_analyst_report", get_ticker_latest_analyst_report)
    graph.add_node("analyst", analyst)
    graph.add_node("is_entry", is_entry)


    # Define the main flow
    graph.add_edge(START, "process_risk")
    graph.add_edge("process_risk", "find_crossover_tickers")
    graph.add_edge("find_crossover_tickers", "is_entry")
    graph.add_edge("is_entry", "get_ticker_latest_analyst_report")
    graph.add_edge("get_ticker_latest_analyst_report", "analyst")
    graph.add_edge("analyst", END)

    return graph


def get_ticker_recommendations(
    db_config: DatabaseCrossoverConfig, llm: ChatOpenAI, crossover_config: CrossoverConfig
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
