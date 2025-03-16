import React, { useState, useEffect, useMemo } from 'react';
import { 
  Box, 
  Typography, 
  Chip,
  Grid,
  Card,
  CardContent,
  Button,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Tooltip,
  Divider,
  Snackbar,
  Alert,
  Popover,
  Avatar,
  Stack,
  Paper
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import CloseIcon from '@mui/icons-material/Close';
import NotificationsIcon from '@mui/icons-material/Notifications';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import NotificationsOffIcon from '@mui/icons-material/NotificationsOff';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import CircleIcon from '@mui/icons-material/Circle';
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

// Notification Dialog Component
const NotificationDialog = ({ open, onClose, ticker }) => {
  const [notificationEnabled, setNotificationEnabled] = useState(false);
  
  // Check if notification is already enabled for this ticker
  useEffect(() => {
    if (open) {
      // Get saved notifications from localStorage
      const savedNotifications = JSON.parse(localStorage.getItem('stockNotifications') || '{}');
      setNotificationEnabled(!!savedNotifications[ticker]);
    }
  }, [open, ticker]);
  
  const handleToggleNotification = () => {
    // Get current notifications
    const savedNotifications = JSON.parse(localStorage.getItem('stockNotifications') || '{}');
    
    if (notificationEnabled) {
      // Remove notification
      delete savedNotifications[ticker];
    } else {
      // Add notification
      savedNotifications[ticker] = {
        enabled: true,
        timestamp: new Date().toISOString(),
      };
    }
    
    // Save back to localStorage
    localStorage.setItem('stockNotifications', JSON.stringify(savedNotifications));
    
    // Update state
    setNotificationEnabled(!notificationEnabled);
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: THEME.radius.medium,
          boxShadow: THEME.shadows.large,
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
          <NotificationsIcon sx={{ mr: 1.5 }} /> 
          {ticker} - Notifications
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
      
      <DialogContent sx={{ p: 3 }}>
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <Typography variant="h6" sx={{ mb: 3 }}>
            {notificationEnabled 
              ? "You'll be notified when it's a good day to buy this stock" 
              : "Get notified when it's a good day to buy this stock"}
          </Typography>
          
          <Button
            variant={notificationEnabled ? "outlined" : "contained"}
            color={notificationEnabled ? "error" : "primary"}
            size="large"
            startIcon={notificationEnabled ? <NotificationsOffIcon /> : <NotificationsActiveIcon />}
            onClick={handleToggleNotification}
            sx={{ 
              px: 3, 
              py: 1.5,
              borderRadius: THEME.radius.small,
              boxShadow: notificationEnabled ? 'none' : THEME.shadows.small,
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: notificationEnabled ? 'none' : THEME.shadows.medium,
              }
            }}
          >
            {notificationEnabled ? "Disable Notifications" : "Enable Notifications"}
          </Button>
          
          <Typography variant="body2" sx={{ mt: 3, color: THEME.colors.textSecondary }}>
            {notificationEnabled 
              ? "You will receive notifications when our algorithm detects a good buying opportunity for this stock."
              : "Our algorithm will monitor this stock and notify you when it's a good time to buy."}
          </Typography>
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

  // Calculate the number of data points to show based on screen size
  // Show more data points in fullscreen mode for better granularity
  const dataPointsToShow = isFullscreen ? chartData.length : Math.min(30, chartData.length);
  const displayData = chartData.slice(-dataPointsToShow);

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
          data={displayData} 
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
            // Show more ticks in fullscreen mode
            interval={isFullscreen ? 'preserveStartEnd' : 'equidistantPreserveStart'}
          />
          <YAxis 
            tick={{ fontSize: isFullscreen ? 14 : fontSize }}
            label={isFullscreen ? { 
              value: 'Price', 
              angle: -90, 
              position: 'insideLeft',
              style: { fontSize: '14px', fill: THEME.colors.textSecondary }
            } : undefined}
            // Add domain padding for better visualization
            domain={['auto', 'auto']}
            allowDecimals={true}
            // Show more precision in fullscreen mode
            tickCount={isFullscreen ? 10 : 5}
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
            // Show more detailed tooltip
            itemSorter={(item) => -item.value}
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
            // Show dots for better granularity
            dot={isFullscreen ? { r: 2, strokeWidth: 1 } : { r: 1, strokeWidth: 1 }}
            activeDot={{ r: isFullscreen ? 8 : 6 }}
            name="Price"
            // Connect null data points
            connectNulls={true}
            // Add animation for better UX
            animationDuration={1000}
            // Improve curve for better visualization
            isAnimationActive={true}
          />
          <Line 
            type="monotone" 
            dataKey="SMA_lower" 
            stroke={THEME.colors.success} 
            strokeWidth={isFullscreen ? 2 : 1.5}
            dot={false}
            name="SMA Lower"
            strokeDasharray={isFullscreen ? "" : "5 5"}
          />
          <Line 
            type="monotone" 
            dataKey="SMA_upper" 
            stroke={THEME.colors.error} 
            strokeWidth={isFullscreen ? 2 : 1.5}
            dot={false}
            name="SMA Upper"
            strokeDasharray={isFullscreen ? "" : "5 5"}
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

// Add NewsCircles component before TickerCard component
const NewsCircles = ({ newsData, ticker, sentimentData }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);

  const handlePopoverOpen = (event, date) => {
    setAnchorEl(event.currentTarget);
    setSelectedDate(date);
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
    setSelectedDate(null);
  };

  const open = Boolean(anchorEl);

  if (!newsData || !newsData[ticker]) return null;

  // Sort dates from oldest to newest for display order (oldest closer to ticker)
  // We still want newest first in the array so we can map them in reverse
  const dates = Object.keys(newsData[ticker]).sort((a, b) => new Date(b) - new Date(a));

  // Helper function to get circle color based on sentiment score
  const getCircleColor = (date) => {
    // Default color if no sentiment data is available
    if (!sentimentData || !sentimentData[ticker]) return THEME.colors.primary;
    
    // Get sentiment data for this ticker
    const tickerSentiment = sentimentData[ticker];
    
    // Use the average score to determine color
    const score = tickerSentiment.average_score;
    
    if (score >= 0.3) return THEME.colors.success; // Strong bullish
    if (score >= 0.1) return `${THEME.colors.success}CC`; // Somewhat bullish
    if (score <= -0.3) return THEME.colors.error; // Strong bearish
    if (score <= -0.1) return `${THEME.colors.error}CC`; // Somewhat bearish
    return THEME.colors.primary; // Neutral
  };

  // Helper function to get sentiment label for tooltip
  const getSentimentLabel = () => {
    if (!sentimentData || !sentimentData[ticker]) return "No sentiment data";
    
    const tickerSentiment = sentimentData[ticker];
    return tickerSentiment.sentiment_category || "Neutral";
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      {/* Reverse the array to display oldest dates first (closest to ticker) */}
      {[...dates].reverse().map((date) => {
        const articles = newsData[ticker][date];
        if (!articles || articles.length === 0) return null;

        // Calculate how recent the date is for styling
        const daysDiff = Math.floor((new Date() - new Date(date)) / (1000 * 60 * 60 * 24));
        
        // Get sentiment for this date's articles
        const dateSentiment = articles.reduce((avg, article) => {
          return avg + (article.overall_sentiment_score || 0);
        }, 0) / articles.length;
        
        // Determine color based on sentiment
        let circleColor;
        if (dateSentiment >= 0.3) circleColor = THEME.colors.success;
        else if (dateSentiment >= 0.1) circleColor = `${THEME.colors.success}CC`;
        else if (dateSentiment <= -0.3) circleColor = THEME.colors.error;
        else if (dateSentiment <= -0.1) circleColor = `${THEME.colors.error}CC`;
        else circleColor = THEME.colors.primary;
        
        // Get sentiment label
        let sentimentLabel = "Neutral";
        if (dateSentiment >= 0.3) sentimentLabel = "Bullish";
        else if (dateSentiment >= 0.1) sentimentLabel = "Somewhat-Bullish";
        else if (dateSentiment <= -0.3) sentimentLabel = "Bearish";
        else if (dateSentiment <= -0.1) sentimentLabel = "Somewhat-Bearish";
        
        return (
          <Box key={date}>
            <Tooltip 
              title={`${new Date(date).toLocaleDateString()} - ${articles.length} articles - ${sentimentLabel}`}
              arrow
              placement="top"
            >
              <IconButton
                size="small"
                onClick={(e) => handlePopoverOpen(e, date)}
                sx={{
                  padding: 0.2,
                  '&:hover': {
                    backgroundColor: `${circleColor}15`,
                  }
                }}
              >
                <CircleIcon 
                  sx={{ 
                    fontSize: '0.8rem',
                    // Use sentiment-based color
                    color: circleColor,
                    opacity: Math.max(0.4, 1 - (daysDiff * 0.15)),
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      opacity: 1,
                      transform: 'scale(1.2)',
                    }
                  }} 
                />
              </IconButton>
            </Tooltip>

            <Popover
              sx={{
                pointerEvents: 'auto', // Allow interaction with popover content
              }}
              open={open && selectedDate === date}
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'center',
              }}
              transformOrigin={{
                vertical: 'top',
                horizontal: 'center',
              }}
              onClose={handlePopoverClose}
              disableRestoreFocus
              slotProps={{
                paper: {
                  onMouseLeave: handlePopoverClose
                }
              }}
            >
              <Paper sx={{ 
                p: 2, 
                maxWidth: 400,
                maxHeight: 400,
                overflowY: 'auto',
                backgroundColor: THEME.colors.cardBackground,
                boxShadow: THEME.shadows.medium,
                border: `1px solid ${THEME.colors.border}`,
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ color: THEME.colors.textSecondary, fontWeight: 600 }}>
                    <CalendarTodayIcon sx={{ fontSize: '0.9rem', mr: 0.5, verticalAlign: 'text-bottom' }} />
                    {new Date(date).toLocaleDateString()} - {articles.length} articles
                  </Typography>
                  <IconButton 
                    size="small" 
                    onClick={handlePopoverClose}
                    sx={{ 
                      padding: 0.5,
                      '&:hover': {
                        backgroundColor: `${THEME.colors.error}15`,
                      }
                    }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
                
                {/* Add sentiment summary */}
                <Box 
                  sx={{ 
                    mb: 2, 
                    p: 1.5, 
                    borderRadius: THEME.radius.small,
                    backgroundColor: `${circleColor}15`,
                    border: `1px solid ${circleColor}40`,
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 600, color: circleColor }}>
                    Daily Sentiment: {sentimentLabel}
                  </Typography>
                  <Typography variant="caption" sx={{ color: THEME.colors.textSecondary }}>
                    Average sentiment score: {dateSentiment.toFixed(2)}
                  </Typography>
                </Box>
                
                <Stack spacing={2}>
                  {/* Sort articles by time published, newest first */}
                  {articles
                    .sort((a, b) => new Date(b.time_published) - new Date(a.time_published))
                    .map((article, idx) => (
                    <Box 
                      key={idx}
                      component="a"
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      sx={{
                        textDecoration: 'none',
                        color: 'inherit',
                        display: 'flex',
                        gap: 2,
                        p: 1,
                        borderRadius: THEME.radius.small,
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          backgroundColor: `${THEME.colors.primary}08`,
                        }
                      }}
                    >
                      {article.banner_image && (
                        <Avatar
                          variant="rounded"
                          src={article.banner_image}
                          alt={article.title}
                          sx={{ 
                            width: 60, 
                            height: 60,
                            borderRadius: THEME.radius.small,
                          }}
                        />
                      )}
                      <Box sx={{ flex: 1 }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            fontWeight: 600,
                            color: THEME.colors.textPrimary,
                            mb: 0.5,
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                          }}
                        >
                          {article.title}
                        </Typography>
                        <Typography 
                          variant="caption" 
                          sx={{ 
                            color: THEME.colors.textSecondary,
                            display: 'block',
                          }}
                        >
                          {article.source} • {new Date(article.time_published).toLocaleTimeString()}
                        </Typography>
                        <Chip
                          size="small"
                          label={article.overall_sentiment_label || "Neutral"}
                          sx={{
                            mt: 0.5,
                            height: 20,
                            fontSize: '0.65rem',
                            backgroundColor: getSentimentColor(article.overall_sentiment_label),
                            color: '#fff',
                          }}
                        />
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            </Popover>
          </Box>
        );
      })}
    </Box>
  );
};

// Helper function to get color based on sentiment
const getSentimentColor = (sentiment) => {
  if (!sentiment) return THEME.colors.textSecondary;
  
  switch(sentiment) {
    case 'Bullish':
      return THEME.colors.success;
    case 'Somewhat-Bullish':
      return `${THEME.colors.success}CC`;
    case 'Bearish':
      return THEME.colors.error;
    case 'Somewhat-Bearish':
      return `${THEME.colors.error}CC`;
    default:
      return THEME.colors.textSecondary;
  }
};

// Modify TickerCard component to include NewsCircles with sentiment data
const TickerCard = ({ ticker, isEntryPoint, index, analystText, returnValue, newsData, sentimentData }) => {
  const [showChartModal, setShowChartModal] = useState(false);
  const [showTextAnalysisModal, setShowTextAnalysisModal] = useState(false);
  const [showNotificationDialog, setShowNotificationDialog] = useState(false);
  const [isNotificationEnabled, setIsNotificationEnabled] = useState(false);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');

  // Check if notification is enabled for this ticker
  useEffect(() => {
    if (!isEntryPoint) {
      const savedNotifications = JSON.parse(localStorage.getItem('stockNotifications') || '{}');
      setIsNotificationEnabled(!!savedNotifications[ticker]);
    }
  }, [ticker, isEntryPoint]);

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
  
  const handleOpenNotificationDialog = () => {
    setShowNotificationDialog(true);
  };
  
  const handleCloseNotificationDialog = () => {
    setShowNotificationDialog(false);
    
    // Check if notification status changed
    const savedNotifications = JSON.parse(localStorage.getItem('stockNotifications') || '{}');
    const newStatus = !!savedNotifications[ticker];
    
    if (newStatus !== isNotificationEnabled) {
      setIsNotificationEnabled(newStatus);
      
      // Show snackbar
      setSnackbarMessage(newStatus 
        ? `You'll be notified when ${ticker} is a good buy` 
        : `Notifications disabled for ${ticker}`);
      setSnackbarSeverity(newStatus ? 'success' : 'info');
      setShowSnackbar(true);
    }
  };
  
  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setShowSnackbar(false);
  };

  // Check if news data exists for this ticker
  const hasNewsData = newsData && newsData[ticker] && Object.keys(newsData[ticker]).length > 0;

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
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
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
                {hasNewsData && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <NewsCircles 
                        newsData={newsData} 
                        ticker={ticker} 
                        sentimentData={sentimentData}
                      />
                      <Tooltip 
                        title="Each circle represents a day with news articles. Colors indicate sentiment: green (bullish), red (bearish), blue (neutral). Click to view insights." 
                        arrow
                        placement="top"
                      >
                        <Box 
                          component="span" 
                          sx={{ 
                            display: 'inline-flex',
                            alignItems: 'center',
                            ml: 0.5,
                            cursor: 'help',
                            color: THEME.colors.textSecondary,
                            '&:hover': { color: THEME.colors.primary }
                          }}
                        >
                          <NewspaperIcon sx={{ fontSize: '0.9rem' }} />
                        </Box>
                      </Tooltip>
                    </Box>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        fontSize: '0.65rem', 
                        color: THEME.colors.textSecondary,
                        mt: -0.5,
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      Recent news insights
                    </Typography>
                  </Box>
                )}
              </Box>
              
              {/* Add notification bell for non-entry tickers */}
              {!isEntryPoint && (
                <Tooltip 
                  title={isNotificationEnabled 
                    ? "Notifications enabled - click to manage" 
                    : "Get notified when it's a good day to buy"}
                  arrow
                  placement="top"
                >
                  <IconButton
                    onClick={handleOpenNotificationDialog}
                    size="small"
                    color={isNotificationEnabled ? "primary" : "default"}
                    sx={{
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        color: THEME.colors.primary,
                        transform: 'scale(1.1)',
                      },
                      ...(isNotificationEnabled && {
                        animation: 'pulse 2s infinite',
                        '@keyframes pulse': {
                          '0%': {
                            boxShadow: `0 0 0 0 ${THEME.colors.primary}40`,
                          },
                          '70%': {
                            boxShadow: `0 0 0 6px ${THEME.colors.primary}00`,
                          },
                          '100%': {
                            boxShadow: `0 0 0 0 ${THEME.colors.primary}00`,
                          },
                        },
                      })
                    }}
                  >
                    {isNotificationEnabled ? <NotificationsActiveIcon /> : <NotificationsIcon />}
                  </IconButton>
                </Tooltip>
              )}
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
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, px: 2, pb: 2 }}>
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
          
          {/* Modals and Dialogs */}
          {isEntryPoint ? (
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
          ) : (
            <NotificationDialog
              open={showNotificationDialog}
              onClose={handleCloseNotificationDialog}
              ticker={ticker}
            />
          )}
          
          {/* Snackbar for notification status changes */}
          <Snackbar
            open={showSnackbar}
            autoHideDuration={4000}
            onClose={handleCloseSnackbar}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            <Alert 
              onClose={handleCloseSnackbar} 
              severity={snackbarSeverity} 
              sx={{ 
                width: '100%',
                borderRadius: THEME.radius.small,
                boxShadow: THEME.shadows.medium,
              }}
            >
              {snackbarMessage}
            </Alert>
          </Snackbar>
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
                  newsData={results.daily_news}
                  sentimentData={results.daily_sentiment}
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
                  newsData={results.daily_news}
                  sentimentData={results.daily_sentiment}
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