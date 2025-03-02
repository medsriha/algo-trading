import React from 'react';
import { 
  Box, 
  Typography, 
  Chip, 
  Paper, 
  Grid,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

// Move styles outside component to prevent recreation on each render
const markdownStyles = {
  '& h1, & h2, & h3, & h4, & h5, & h6': {
    marginTop: '1rem',
    marginBottom: '0.5rem',
    fontWeight: 'bold',
  },
  '& p': {
    marginBottom: '1rem',
  },
  '& ul, & ol': {
    paddingLeft: '2rem',
    marginBottom: '1rem',
  },
  '& blockquote': {
    borderLeft: '3px solid #9e9e9e',
    paddingLeft: '1rem',
    fontStyle: 'italic',
    margin: '1rem 0',
  },
  '& code': {
    fontFamily: 'monospace',
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
    padding: '0.2rem 0.4rem',
    borderRadius: '3px',
  },
  '& pre': {
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
    padding: '1rem',
    borderRadius: '4px',
    overflowX: 'auto',
    marginBottom: '1rem',
  },
  '& a': {
    color: '#3f51b5',
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
  '& img': {
    maxWidth: '100%',
    height: 'auto',
  },
  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    marginBottom: '1rem',
  },
  '& th, & td': {
    border: '1px solid #e0e0e0',
    padding: '0.5rem',
  },
};

const MotionWrapper = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
  >
    {children}
  </motion.div>
);

const TickerCard = ({ ticker, date, isEntryPoint, chartPath, index }) => (
  <Grid item xs={12} sm={6} md={4} key={index}>
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
    >
      <Card 
        elevation={2} 
        sx={{ 
          borderRadius: 2,
          position: 'relative',
          overflow: 'visible',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          '&:after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '4px',
            background: 'linear-gradient(to right, #3f51b5, #f50057)',
            borderTopLeftRadius: '8px',
            borderTopRightRadius: '8px'
          }
        }}
      >
        <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
            {ticker}
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Crossover Date: {new Date(date).toLocaleDateString()}
          </Typography>
          
          {/* Display the chart */}
          <Box sx={{ mt: 2, mb: 2, textAlign: 'center', flexGrow: 1, display: 'flex', 
                     justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
            {chartPath ? (
              <img 
                src={`/api/chart/${ticker}`} 
                alt={`${ticker} price chart`} 
                style={{ 
                  width: '100%', 
                  maxHeight: '220px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                }} 
                onError={(e) => {
                  console.error(`Failed to load chart for ${ticker}`);
                  e.target.style.display = 'none';
                }}
              />
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                Chart not available
              </Typography>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 'auto', justifyContent: 'center' }}>
            {isEntryPoint ? (
              <Chip 
                icon={<CheckCircleIcon />} 
                label="Entry Point Valid" 
                color="success" 
                size="small" 
                variant="outlined"
                sx={{ fontWeight: 'bold' }}
              />
            ) : (
              <Chip 
                icon={<CancelIcon />} 
                label="Not Entry Point" 
                color="error" 
                size="small" 
                variant="outlined"
                sx={{ fontWeight: 'bold' }}
              />
            )}
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  </Grid>
);

const ResultsDisplay = ({ results }) => {
  // Check if results has the analyst field
  if (!results || !results.analyst) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No analysis results available.
        </Typography>
      </Box>
    );
  }

  // Parse ticker information if available
  const tickers = results.tickers || [];
  const entry = results.entry || {};
  const analystReport = results.analyst || '';
  const chartPaths = results.chart_paths || {};

  // Debug log to ensure chart paths are received
  console.log("Chart paths:", chartPaths);
  
  return (
    <MotionWrapper>
      <Box sx={{ mt: 4 }}>
        <Typography 
          variant="h5" 
          gutterBottom 
          sx={{ 
            borderLeft: '4px solid #3f51b5', 
            pl: 2,
            mb: 3
          }}
        >
          Analysis Results
        </Typography>

        {/* Analyst Report */}
        <Paper 
          elevation={3} 
          sx={{ 
            p: 3, 
            mb: 4, 
            borderRadius: 2,
            background: 'linear-gradient(to right, rgba(63, 81, 181, 0.1), rgba(0, 0, 0, 0))'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
            <FormatQuoteIcon sx={{ fontSize: 40, color: '#3f51b5', mr: 2, transform: 'scaleX(-1)' }} />
            <Typography variant="h6" gutterBottom>
              AI Analyst Summary
            </Typography>
          </Box>
          
          <Box sx={{ pl: 2 }}>
            <Box className="markdown-content" sx={markdownStyles}>
              <ReactMarkdown>{analystReport}</ReactMarkdown>
            </Box>
          </Box>
        </Paper>

        {/* Ticker List */}
        {Array.isArray(tickers) && tickers.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
              Analyzed Tickers with Charts
            </Typography>
            
            <Grid container spacing={3}>
              {tickers.map(([ticker, date], index) => (
                <TickerCard 
                  key={`${ticker}-${index}`}
                  ticker={ticker}
                  date={date}
                  isEntryPoint={entry && entry[ticker]}
                  chartPath={chartPaths[ticker]}
                  index={index}
                />
              ))}
            </Grid>
          </Box>
        )}
      </Box>
    </MotionWrapper>
  );
};

export default ResultsDisplay; 