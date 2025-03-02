from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Literal
import logging
import os.path

from langchain_openai import ChatOpenAI
from algo_trading.database import DatabaseCrossoversConfig
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
    take_profit=0.10,
    stop_loss=0.05,
    crossover_length=10,
    rsi_period=14,
    rsi_overbought=70,
    rsi_oversold=30,
    rsi_underbought=50,
)

# Define default database config
default_db_config = DatabaseCrossoversConfig(
    db_name="crossovers.db", 
    table_name="crossovers"
)

@app.post("/analyze")
async def analyze_tickers(request: RiskLevelRequest) -> Dict[str, Any]:
    """
    Analyze tickers based on the given risk level
    """
    logger.info(f"Received analysis request with risk level: {request.risk_level}")
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Get the agent
        agent_executor = get_ticker_recommendations(
            db_config=default_db_config,
            llm=llm,
            crossover_config=default_crossover_config
        )
        
        # Run the agent with the provided risk level
        result = agent_executor.invoke({"risk_level": request.risk_level})
        
        # Log chart paths to confirm they're included in the response
        if 'chart_paths' in result:
            logger.info(f"Chart paths included in response: {list(result['chart_paths'].keys())}")
        else:
            logger.warning("No chart paths found in analysis results")
        
        logger.info(f"Analysis completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running
    """
    return {"message": "Algo Trading API is running"}

@app.get("/api/chart/{ticker}")
async def get_ticker_chart(ticker: str):
    """
    Serve the chart image for a specific ticker
    """
    chart_path = f"/Users/deepset/algo-trading/data/{ticker}/plot.png"
    
    try:
        if not os.path.isfile(chart_path):
            logger.error(f"Chart not found for ticker: {ticker}")
            raise HTTPException(status_code=404, detail=f"Chart not found for ticker: {ticker}")
        
        logger.info(f"Serving chart for ticker: {ticker}")
        return FileResponse(
            chart_path, 
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
        )
    except Exception as e:
        logger.error(f"Error serving chart for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving chart: {str(e)}") 