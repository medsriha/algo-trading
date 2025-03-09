# Algo Trading Analysis Web Application

A beautiful web application that provides stock trading recommendations based on algorithmic analysis and risk profiles.

## Features

- Modern, responsive UI built with React and Material UI
- Risk profile selection (Conservative, Moderate, Aggressive)
- AI-powered analysis of stock crossover patterns
- Interactive display of analysis results and recommendations
- FastAPI backend integration with the algo trading system

## Project Structure

```
web_app/
├── backend/              # FastAPI server code
│   ├── main.py           # API endpoints
│   ├── requirements.txt  # Python dependencies
│   └── run.py            # Server startup script
│
└── frontend/             # React frontend
    ├── public/           # Static assets
    └── src/              # React components
        ├── components/   # UI components
        ├── App.js        # Main application component
        └── index.js      # Application entry point
```

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd web_app/backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the parent algo-trading module is available in your Python path.

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd web_app/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```
   or 
   ```
   yarn install
   ```

## Running the Application

### Start the Backend Server

1. From the backend directory:
   ```
   python run.py
   ```
   
   The API will be available at http://localhost:8000

### Start the Frontend Development Server

1. From the frontend directory:
   ```
   npm start
   ```
   or
   ```
   yarn start
   ```
   
   The application will be available at http://localhost:3000

## Usage

1. Open the application in your web browser
2. Select a risk profile from the dropdown menu (Conservative, Moderate, or Aggressive)
3. Click "Analyze" to get stock recommendations
4. View the analysis results, including ticker information and AI analyst commentary

## Technical Details

- The frontend is built with React and Material UI for a slick, professional interface
- Framer Motion provides smooth animations and transitions
- The backend uses FastAPI to provide a fast, modern API interface
- The application integrates with the existing algo-trading system to leverage its analysis capabilities 