import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import { THEME_CONSTANTS } from '../index';

const Loading = () => {
  // Array of loading messages to cycle through
  const loadingMessages = [
    "Analyzing market patterns...",
    "Calculating risk profiles...",
    "Identifying opportunities...",
    "Evaluating market conditions...",
    "Examining technical indicators...",
    "Checking the news..."
  ];
  
  const [messageIndex, setMessageIndex] = React.useState(0);
  
  // Cycle through loading messages
  React.useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prevIndex) => (prevIndex + 1) % loadingMessages.length);
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          py: 6,
          px: 2,
          my: 2,
          borderRadius: THEME_CONSTANTS.radius.medium,
          backgroundColor: THEME_CONSTANTS.colors.background,
          border: `1px solid ${THEME_CONSTANTS.colors.border}`,
          boxShadow: THEME_CONSTANTS.shadows.small,
        }}
      >
        <CircularProgress 
          size={60} 
          thickness={4} 
          sx={{ mb: 3, color: THEME_CONSTANTS.colors.primary }} 
        />
        
        <motion.div
          key={messageIndex}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -5 }}
          transition={{ duration: 0.5 }}
        >
          <Typography 
            variant="h6" 
            align="center" 
            sx={{ 
              mb: 1, 
              color: THEME_CONSTANTS.colors.textPrimary,
              fontWeight: 600 
            }}
          >
            {loadingMessages[messageIndex]}
          </Typography>
        </motion.div>
        
        <Typography 
          variant="body2" 
          align="center" 
          sx={{ color: THEME_CONSTANTS.colors.textSecondary }}
        >
          This may take a minute or two. Please wait...
        </Typography>
      </Box>
    </motion.div>
  );
};

export default Loading; 