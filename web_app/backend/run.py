import uvicorn
import sys
import os

def add_parent_to_path():
    # Add parent directory to sys.path so we can import from algo_trading
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(os.path.dirname(current))
    if parent not in sys.path:
        sys.path.append(parent)

if __name__ == "__main__":
    add_parent_to_path()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 