from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Literal
import logging
import os.path

from langchain_openai import ChatOpenAI
from algo_trading.database import DatabaseCrossoverConfig
from algo_trading.models import CrossoverConfig
from algo_trading.agents.agents import get_ticker_recommendations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log"),
    ],
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Algo Trading API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class RiskLevelRequest(BaseModel):
    risk_level: Literal["conservative", "moderate", "aggressive"]

# Define default crossover config
default_crossover_config = CrossoverConfig(
    upper_sma=50,
    lower_sma=20,
    rsi_period=14,
    rsi_underbought=50,
    start_date="2025-01-01",

)

# Define default database config
default_db_config = DatabaseCrossoverConfig(
    db_name="crossover.db", 
    table_name="crossover"
)

@app.post("/analyze")
async def analyze_tickers(request: RiskLevelRequest) -> Dict[str, Any]:
    """
    Analyze tickers based on the given risk level
    """
    logger.info(f"Received analysis request with risk level: {request.risk_level}")
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4o")
        
        # Get the agent
        agent_executor = get_ticker_recommendations(
            db_config=default_db_config,
            llm=llm,
            crossover_config=default_crossover_config
        )
        
        # Run the agent with the provided risk level
        result = agent_executor.invoke({"risk_level": request.risk_level})
        
        logger.info(f"Analysis completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/data/{ticker}/{timeframe}/{filename}")
async def get_ticker_data(ticker: str, timeframe: str, filename: str):
    """
    Serve CSV data for a specific ticker
    """
    logger.info(f"Received request for {ticker} {timeframe} {filename}")
    
    # Use the absolute path as specified
    file_path = f"/Users/deepset/algo-trading/data/{ticker}/{timeframe}/{filename}"
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {ticker}/{timeframe}/{filename}")
    
    logger.info(f"Serving file: {file_path}")
    return FileResponse(file_path)

@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running
    """
    return {"message": "Algo Trading API is running"} 