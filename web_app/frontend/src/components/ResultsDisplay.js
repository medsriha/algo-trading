import React, { useState, useEffect } from 'react';
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
  AccordionDetails,
  Button,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  DialogActions,
  TextField,
  Tooltip,
  Snackbar,
  Alert,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import HideChartIcon from '@mui/icons-material/BarChart';
import CloseIcon from '@mui/icons-material/Close';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import NotificationsIcon from '@mui/icons-material/Notifications';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';
import axios from 'axios';

// Theme constants
const THEME = {
  colors: {
    background: '#f9f9fa',
    cardBackground: '#ffffff',
    primary: '#4c6ef5',
    success: '#48bb78',
    error: '#f56565',
    textPrimary: '#2d3748',
    textSecondary: '#718096',
    border: '#e2e8f0'
  },
  shadows: {
    small: '0 2px 8px rgba(0,0,0,0.05)',
    medium: '0 4px 12px rgba(0,0,0,0.08)',
    large: '0 8px 24px rgba(0,0,0,0.12)'
  },
  radius: {
    small: '8px',
    medium: '12px',
    large: '16px'
  }
};

// Move styles outside component to prevent recreation on each render
const markdownStyles = {
  '& h1, & h2, & h3, & h4, & h5, & h6': {
    marginTop: '1rem',
    marginBottom: '0.5rem',
    fontWeight: 'bold',
    color: THEME.colors.textPrimary,
  },
  '& p': {
    marginBottom: '1rem',
    color: THEME.colors.textSecondary,
    lineHeight: 1.6,
  },
  '& ul, & ol': {
    paddingLeft: '2rem',
    marginBottom: '1rem',
    color: THEME.colors.textSecondary,
  },
  '& blockquote': {
    borderLeft: `3px solid ${THEME.colors.border}`,
    paddingLeft: '1rem',
    fontStyle: 'italic',
    margin: '1rem 0',
    color: THEME.colors.textSecondary,
  },
  '& code': {
    fontFamily: 'monospace',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    padding: '0.2rem 0.4rem',
    borderRadius: THEME.radius.small,
  },
  '& pre': {
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    padding: '1rem',
    borderRadius: THEME.radius.small,
    overflowX: 'auto',
    marginBottom: '1rem',
  },
  '& a': {
    color: THEME.colors.primary,
    textDecoration: 'none',
    transition: 'all 0.2s ease',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
  '& img': {
    maxWidth: '100%',
    height: 'auto',
    borderRadius: THEME.radius.small,
  },
  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    marginBottom: '1rem',
    borderRadius: THEME.radius.small,
    overflow: 'hidden',
  },
  '& th, & td': {
    border: `1px solid ${THEME.colors.border}`,
    padding: '0.75rem',
    color: THEME.colors.textSecondary,
  },
  '& th': {
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    color: THEME.colors.textPrimary,
    fontWeight: 600,
  }
};

// Chart Modal Component
const ChartModal = ({ open, onClose, ticker }) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="lg"
      PaperProps={{
        sx: {
          borderRadius: THEME.radius.medium,
          boxShadow: THEME.shadows.large,
          height: '80vh',
          maxHeight: '800px',
        }
      }}
    >
      <DialogTitle sx={{ 
        borderBottom: `1px solid ${THEME.colors.border}`,
        px: 3,
        py: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: `${THEME.colors.primary}08`,
      }}>
        <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
          <ShowChartIcon sx={{ mr: 1.5 }} /> 
          {ticker} - Price and SMA Crossover Chart
        </Typography>
        <IconButton 
          onClick={onClose} 
          size="small"
          sx={{
            backgroundColor: 'rgba(0,0,0,0.05)',
            '&:hover': {
              backgroundColor: 'rgba(0,0,0,0.1)',
            }
          }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      
      <DialogContent sx={{ p: { xs: 1, md: 3 }, height: 'calc(100% - 64px)' }}>
        <StockChart ticker={ticker} isFullscreen={true} />
      </DialogContent>
    </Dialog>
  );
};

// Text Analysis Modal Component
const TextAnalysisModal = ({ open, onClose, ticker, analysisText }) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="lg"
      PaperProps={{
        sx: {
          borderRadius: THEME.radius.medium,
          boxShadow: THEME.shadows.large,
          height: '80vh',
          maxHeight: '800px',
        }
      }}
    >
      <DialogTitle sx={{ 
        borderBottom: `1px solid ${THEME.colors.border}`,
        px: 3,
        py: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: `${THEME.colors.primary}08`,
      }}>
        <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
          <FormatQuoteIcon sx={{ mr: 1.5 }} /> 
          {ticker} - Analysis
        </Typography>
        <IconButton 
          onClick={onClose} 
          size="small"
          sx={{
            backgroundColor: 'rgba(0,0,0,0.05)',
            '&:hover': {
              backgroundColor: 'rgba(0,0,0,0.1)',
            }
          }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      
      <DialogContent sx={{ p: { xs: 2, md: 4 }, height: 'calc(100% - 64px)', overflowY: 'auto' }}>
        <Box 
          className="markdown-content" 
          sx={{
            ...markdownStyles,
            fontSize: '1rem',
          }}
        >
          <ReactMarkdown>{analysisText}</ReactMarkdown>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

// Stock Chart Component
const StockChart = ({ ticker, isFullscreen = false }) => {
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    
    const fetchChartData = async () => {
      try {
        setLoading(true);
        // Use the direct file path as specified
        const response = await axios.get(`/api/data/${ticker}/today/crossover.csv`);
        
        // Parse CSV data
        const parsedData = parseCSVData(response.data);
        
        if (isMounted) {
          setChartData(parsedData);
          setLoading(false);
        }
      } catch (err) {
        console.error('Error fetching chart data:', err);
        if (isMounted) {
          setError('Failed to load chart data');
          setLoading(false);
        }
      }
    };

    fetchChartData();
    
    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [ticker]);

  // Parse CSV data function
  const parseCSVData = (csvText) => {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
      const values = line.split(',');
      const dataPoint = {};
      
      headers.forEach((header, index) => {
        // Convert numeric values
        const value = values[index];
        dataPoint[header.trim()] = isNaN(value) ? value : parseFloat(value);
      });
      
      return dataPoint;
    });
  };

  // Determine height based on whether it's fullscreen or not
  const chartHeight = isFullscreen ? '100%' : 200;
  const fontSize = isFullscreen ? 12 : 10;

  if (loading) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: chartHeight, 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        flexDirection: 'column',
        p: 3,
        backgroundColor: `${THEME.colors.background}50`,
        borderRadius: THEME.radius.small,
        border: `1px solid ${THEME.colors.border}`,
      }}>
        <CircularProgress size={isFullscreen ? 50 : 30} sx={{ mb: 2 }} />
        <Typography variant={isFullscreen ? "body1" : "caption"} color={THEME.colors.textSecondary}>
          Loading chart data for {ticker}...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ 
        p: 3, 
        color: THEME.colors.error,
        backgroundColor: `${THEME.colors.error}10`,
        borderRadius: THEME.radius.small,
        border: `1px solid ${THEME.colors.error}30`,
        textAlign: 'center',
        height: chartHeight,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
      }}>
        <Typography variant={isFullscreen ? "h6" : "body2"} sx={{ fontWeight: 500 }}>
          {error}
        </Typography>
        <Typography variant={isFullscreen ? "body1" : "caption"} sx={{ display: 'block', mt: 1, color: THEME.colors.textSecondary }}>
          Please try again later
        </Typography>
      </Box>
    );
  }

  // If no data
  if (!chartData || chartData.length === 0) {
    return (
      <Box sx={{ 
        p: 3, 
        color: THEME.colors.textSecondary,
        backgroundColor: `${THEME.colors.background}50`,
        borderRadius: THEME.radius.small,
        border: `1px solid ${THEME.colors.border}`,
        textAlign: 'center',
        height: chartHeight,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
      }}>
        <Typography variant={isFullscreen ? "h6" : "body2"}>
          No chart data available for {ticker}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      width: '100%', 
      height: chartHeight, 
      mt: isFullscreen ? 0 : 2,
      mb: isFullscreen ? 0 : 2,
      p: isFullscreen ? 0 : 1,
      backgroundColor: isFullscreen ? 'transparent' : `${THEME.colors.background}50`,
      borderRadius: isFullscreen ? 0 : THEME.radius.small,
      border: isFullscreen ? 'none' : `1px solid ${THEME.colors.border}`,
    }}>
      {!isFullscreen && (
        <Typography 
          variant="subtitle2" 
          sx={{ 
            mb: 1, 
            fontWeight: 600, 
            color: THEME.colors.textSecondary,
            fontSize: '0.75rem',
            textAlign: 'center'
          }}
        >
          Price and SMA Crossover Chart
        </Typography>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={chartData.slice(-30)} 
          margin={{ 
            top: isFullscreen ? 20 : 5, 
            right: isFullscreen ? 30 : 5, 
            left: isFullscreen ? 20 : 0, 
            bottom: isFullscreen ? 20 : 5 
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={`${THEME.colors.border}`} />
          <XAxis 
            dataKey="timestamp" 
            tick={{ fontSize: isFullscreen ? 14 : fontSize }}
            tickFormatter={(value) => {
              if (!value) return '';
              const date = new Date(value);
              return `${date.getMonth()+1}/${date.getDate()}`;
            }}
            label={isFullscreen ? { 
              value: 'Date', 
              position: 'insideBottomRight', 
              offset: -10,
              style: { fontSize: '14px', fill: THEME.colors.textSecondary }
            } : undefined}
          />
          <YAxis 
            tick={{ fontSize: isFullscreen ? 14 : fontSize }}
            label={isFullscreen ? { 
              value: 'Price', 
              angle: -90, 
              position: 'insideLeft',
              style: { fontSize: '14px', fill: THEME.colors.textSecondary }
            } : undefined}
          />
          <RechartsTooltip 
            contentStyle={{ 
              backgroundColor: THEME.colors.cardBackground,
              border: `1px solid ${THEME.colors.border}`,
              borderRadius: THEME.radius.small,
              boxShadow: THEME.shadows.small,
              fontSize: isFullscreen ? '14px' : '12px',
              padding: isFullscreen ? '10px 14px' : '8px 10px',
            }}
            formatter={(value, name) => [parseFloat(value).toFixed(2), name]}
            labelFormatter={(label) => {
              if (!label) return '';
              const date = new Date(label);
              return date.toLocaleDateString();
            }}
          />
          <Legend 
            wrapperStyle={{ 
              fontSize: isFullscreen ? '14px' : '10px',
              paddingTop: isFullscreen ? '20px' : '0',
            }} 
            verticalAlign={isFullscreen ? "bottom" : "top"}
            height={isFullscreen ? 40 : 20}
          />
          <Line 
            type="monotone" 
            dataKey="close" 
            stroke={THEME.colors.primary} 
            strokeWidth={isFullscreen ? 3 : 2}
            dot={isFullscreen ? { r: 3 } : false}
            activeDot={{ r: isFullscreen ? 8 : 6 }}
            name="Price"
          />
          <Line 
            type="monotone" 
            dataKey="SMA_lower" 
            stroke={THEME.colors.success} 
            strokeWidth={isFullscreen ? 2 : 1.5}
            dot={false}
            name="SMA Lower"
          />
          <Line 
            type="monotone" 
            dataKey="SMA_upper" 
            stroke={THEME.colors.error} 
            strokeWidth={isFullscreen ? 2 : 1.5}
            dot={false}
            name="SMA Upper"
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

const MotionWrapper = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
  >
    {children}
  </motion.div>
);

const TickerCard = ({ ticker, isEntryPoint, index, analystText, returnValue }) => {
  const [showChartModal, setShowChartModal] = useState(false);
  const [showTextAnalysisModal, setShowTextAnalysisModal] = useState(false);

  const handleOpenChartModal = () => {
    setShowChartModal(true);
  };

  const handleCloseChartModal = () => {
    setShowChartModal(false);
  };

  const handleOpenTextAnalysisModal = () => {
    setShowTextAnalysisModal(true);
  };

  const handleCloseTextAnalysisModal = () => {
    setShowTextAnalysisModal(false);
  };

  return (
    <Grid item xs={12} sm={12} md={6} lg={4} key={index}>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      >
        <Card 
          elevation={0} 
          sx={{ 
            borderRadius: THEME.radius.medium,
            position: 'relative',
            overflow: 'visible',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: THEME.colors.cardBackground,
            border: `1px solid ${THEME.colors.border}`,
            transition: 'all 0.3s ease',
            boxShadow: THEME.shadows.small,
            '&:hover': {
              boxShadow: THEME.shadows.medium,
              transform: 'translateY(-4px)',
            }
          }}
        >
          <CardContent sx={{ 
            flexGrow: 1, 
            display: 'flex', 
            flexDirection: 'column',
            p: { xs: 2, md: 3 },
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography 
                variant="h6" 
                component="div" 
                sx={{ 
                  fontWeight: '700',
                  color: THEME.colors.textPrimary,
                  fontSize: { xs: '1.1rem', md: '1.25rem' }
                }}
              >
                {ticker}
              </Typography>
            </Box>
            
            {returnValue !== undefined && (
              <Typography 
                variant="body2" 
                sx={{ 
                  mb: 1.5,
                  color: parseFloat(returnValue) >= 0 
                    ? THEME.colors.success 
                    : THEME.colors.error,
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {parseFloat(returnValue).toFixed(2)}%
              </Typography>
            )}
            
            <Box sx={{ mt: 'auto', justifyContent: 'center', display: 'flex', alignItems: 'center', pt: 2 }}>
              <Chip 
                label={isEntryPoint ? "Buy today" : "Not a good day to buy"}
                color={isEntryPoint ? "success" : "default"}
                icon={isEntryPoint ? <CheckCircleIcon /> : <CancelIcon />}
                sx={{ 
                  fontWeight: isEntryPoint ? '600' : '500',
                  backgroundColor: isEntryPoint ? `${THEME.colors.success}20` : `${THEME.colors.textSecondary}10`,
                  color: isEntryPoint ? THEME.colors.success : THEME.colors.textSecondary,
                  '& .MuiChip-icon': { 
                    color: 'inherit' 
                  },
                  borderRadius: '8px',
                  px: 1,
                  py: 2.5,
                }}
              />
            </Box>

            {isEntryPoint && (
              <>
                <Divider sx={{ my: 2 }} />

                {/* Action Buttons - only shown for entry tickers */}
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, px: 3, pb: 2 }}>
                  <Tooltip title="View price chart and SMA crossover analysis" arrow placement="top">
                    <IconButton
                      color="primary"
                      onClick={handleOpenChartModal}
                      size="large"
                      sx={{
                        border: `2px solid ${THEME.colors.border}`,
                        borderRadius: THEME.radius.small,
                        p: 1.5,
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          borderColor: THEME.colors.primary,
                          backgroundColor: `${THEME.colors.primary}15`,
                          transform: 'translateY(-2px)',
                          boxShadow: THEME.shadows.small,
                        }
                      }}
                    >
                      <ShowChartIcon fontSize="large" />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="View detailed analyst text and investment recommendations" arrow placement="top">
                    <IconButton
                      color="primary"
                      onClick={handleOpenTextAnalysisModal}
                      size="large"
                      sx={{
                        border: `2px solid ${THEME.colors.border}`,
                        borderRadius: THEME.radius.small,
                        p: 1.5,
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          borderColor: THEME.colors.primary,
                          backgroundColor: `${THEME.colors.primary}15`,
                          transform: 'translateY(-2px)',
                          boxShadow: THEME.shadows.small,
                        }
                      }}
                    >
                      <FormatQuoteIcon fontSize="large" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </>
            )}
          </CardContent>
          
          {/* Modals and Dialogs - only for entry tickers */}
          {isEntryPoint && (
            <>
              <ChartModal 
                open={showChartModal} 
                onClose={handleCloseChartModal} 
                ticker={ticker} 
              />
              
              <TextAnalysisModal
                open={showTextAnalysisModal}
                onClose={handleCloseTextAnalysisModal}
                ticker={ticker}
                analysisText={analystText}
              />
            </>
          )}
        </Card>
      </motion.div>
    </Grid>
  );
};

const ResultsDisplay = ({ results }) => {
  // Add state hooks for visible items
  const [visibleEntryTickers, setVisibleEntryTickers] = useState(6);
  const [visibleNonEntryTickers, setVisibleNonEntryTickers] = useState(6);

  // Check if results has the analyst field
  if (!results || !results.analyst) {
    return (
      <Box sx={{ 
        p: 4, 
        textAlign: 'center',
        backgroundColor: THEME.colors.background,
        borderRadius: THEME.radius.large,
        boxShadow: THEME.shadows.small
      }}>
        <Typography variant="body1" color={THEME.colors.textSecondary}>
          No analysis results available.
        </Typography>
      </Box>
    );
  }

  // Parse ticker information if available
  const tickers = results.tickers || [];
  const entryTickers = results.entry_candidates || [];
  const notEntryTickers = results.not_entry_candidates || [];
  const entry = results.entry || {};
  const analystReport = results.analyst || {};
  
  // Check if analyst report is an object or string (for backward compatibility)
  const isAnalystObject = typeof analystReport === 'object' && analystReport !== null;
  
  // Helper function to get analyst text for a ticker
  const getAnalystTextForTicker = (ticker) => {
    if (isAnalystObject) {
      return analystReport[ticker] || '';
    } else {
      // If analyst report is a string, we'll show the same text for all tickers
      return analystReport;
    }
  };
  
  // Get current date for display
  const strategyDate = new Date().toLocaleDateString();
  
  // Get return values for each ticker
  const getReturnValueForTicker = (ticker) => {
    // Look in entry data if available
    if (entry && entry[ticker]) {
      return entry[ticker];
    }
    
    return undefined;
  };
  
  // Handler for showing more entry tickers
  const handleShowMoreEntryTickers = () => {
    setVisibleEntryTickers(entryTickers.length);
  };

  // Handler for showing more non-entry tickers
  const handleShowMoreNonEntryTickers = () => {
    setVisibleNonEntryTickers(notEntryTickers.length);
  };
  
  return (
    <MotionWrapper>
      <Box sx={{ 
        mt: 4, 
        backgroundColor: THEME.colors.background,
        borderRadius: THEME.radius.large,
        p: { xs: 2, sm: 4 },
        maxWidth: '1400px',
        mx: 'auto',
        boxShadow: THEME.shadows.small,
      }}>
        <Typography 
          variant="body1" 
          sx={{ 
            mb: 4, 
            pl: { xs: 1, md: 2 },
            color: THEME.colors.textSecondary,
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            '& svg': {
              mr: 1,
              fontSize: '1.2rem',
              color: THEME.colors.primary
            }
          }}
        >
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
            <CalendarTodayIcon fontSize="small" />
            Strategy date: {strategyDate}
          </Box>
        </Typography>

        {/* Entry Point Tickers */}
        {Array.isArray(entryTickers) && entryTickers.length > 0 && (
          <Box sx={{ 
            mb: 5, 
            backgroundColor: THEME.colors.cardBackground,
            borderRadius: THEME.radius.medium,
            p: { xs: 2, md: 3 },
            boxShadow: THEME.shadows.medium,
            border: `1px solid ${THEME.colors.success}30`,
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '4px',
              backgroundColor: THEME.colors.success,
              borderTopLeftRadius: THEME.radius.medium,
              borderTopRightRadius: THEME.radius.medium,
            }
          }}>
            <Typography 
              variant="h6" 
              sx={{ 
                mb: 3,
                color: THEME.colors.success,
                display: 'flex',
                alignItems: 'center',
                fontWeight: 600,
                fontSize: { xs: '1.2rem', md: '1.4rem' },
                pl: { xs: 1, md: 1 },
              }}
            >
              <CheckCircleIcon sx={{ mr: 1.5, fontSize: '1.6rem' }} /> 
              Today is a good day to buy these tickers
            </Typography>
            
            <Grid container spacing={3}>
              {entryTickers.slice(0, visibleEntryTickers).map((ticker, index) => (
                <TickerCard 
                  key={`entry-${ticker}-${index}`}
                  ticker={ticker}
                  isEntryPoint={true}
                  index={index}
                  analystText={getAnalystTextForTicker(ticker)}
                  returnValue={getReturnValueForTicker(ticker)}
                />
              ))}
            </Grid>
            
            {entryTickers.length > visibleEntryTickers && (
              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <motion.div 
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                >
                  <Button 
                    variant="outlined"
                    onClick={handleShowMoreEntryTickers}
                    sx={{ 
                      borderRadius: THEME.radius.small,
                      borderColor: THEME.colors.border,
                      color: THEME.colors.primary,
                      px: 3,
                      py: 1,
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      textTransform: 'none',
                      boxShadow: 'none',
                      '&:hover': {
                        borderColor: THEME.colors.primary,
                        backgroundColor: `${THEME.colors.primary}08`,
                        boxShadow: 'none',
                      }
                    }}
                    endIcon={<ArrowForwardIcon />}
                  >
                    See {entryTickers.length - visibleEntryTickers} more tickers
                  </Button>
                </motion.div>
              </Box>
            )}
          </Box>
        )}

        {/* Non-Entry Tickers */}
        {Array.isArray(notEntryTickers) && notEntryTickers.length > 0 && (
          <Box sx={{ 
            mb: 4, 
            backgroundColor: THEME.colors.cardBackground,
            borderRadius: THEME.radius.medium,
            p: { xs: 2, md: 3 },
            boxShadow: THEME.shadows.small,
            border: `1px solid ${THEME.colors.border}`,
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '4px',
              backgroundColor: THEME.colors.textSecondary,
              borderTopLeftRadius: THEME.radius.medium,
              borderTopRightRadius: THEME.radius.medium,
            }
          }}>
            <Typography 
              variant="h6" 
              sx={{ 
                mb: 3,
                color: THEME.colors.textSecondary,
                display: 'flex',
                alignItems: 'center',
                fontWeight: 600,
                fontSize: { xs: '1.1rem', md: '1.25rem' },
                pl: { xs: 1, md: 1 },
              }}
            >
              <CancelIcon sx={{ mr: 1.5 }} /> 
              Not recommended for purchase today
              <Typography 
                variant="body2" 
                component="span" 
                sx={{ 
                  ml: 2, 
                  color: THEME.colors.primary,
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '0.8rem'
                }}
              >
                <NotificationsIcon sx={{ mr: 0.5, fontSize: '0.9rem' }} />
                Click the bell icon to get alerts
              </Typography>
            </Typography>
            
            <Grid container spacing={3}>
              {notEntryTickers.slice(0, visibleNonEntryTickers).map((ticker, index) => (
                <TickerCard 
                  key={`not-entry-${ticker}-${index}`}
                  ticker={ticker}
                  isEntryPoint={false}
                  index={index}
                  analystText={getAnalystTextForTicker(ticker)}
                  returnValue={getReturnValueForTicker(ticker)}
                />
              ))}
            </Grid>
            
            {notEntryTickers.length > visibleNonEntryTickers && (
              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <motion.div 
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                >
                  <Button 
                    variant="outlined"
                    onClick={handleShowMoreNonEntryTickers}
                    sx={{ 
                      borderRadius: THEME.radius.small,
                      borderColor: THEME.colors.border,
                      color: THEME.colors.primary,
                      px: 3,
                      py: 1,
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      textTransform: 'none',
                      boxShadow: 'none',
                      '&:hover': {
                        borderColor: THEME.colors.primary,
                        backgroundColor: `${THEME.colors.primary}08`,
                        boxShadow: 'none',
                      }
                    }}
                    endIcon={<ArrowForwardIcon />}
                  >
                    See {notEntryTickers.length - visibleNonEntryTickers} more tickers
                  </Button>
                </motion.div>
              </Box>
            )}
          </Box>
        )}
      </Box>
    </MotionWrapper>
  );
};

export default ResultsDisplay; 