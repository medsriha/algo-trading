import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { motion } from 'framer-motion';

const Loading = () => {
  // Array of loading messages to cycle through
  const loadingMessages = [
    "Analyzing market patterns...",
    "Calculating risk profiles...",
    "Processing historical data...",
    "Identifying crossover opportunities...",
    "Evaluating market conditions...",
    "Consulting with AI algorithms...",
    "Examining technical indicators..."
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
          borderRadius: 3,
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
        }}
      >
        <CircularProgress size={60} thickness={4} sx={{ mb: 3 }} />
        
        <motion.div
          key={messageIndex}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -5 }}
          transition={{ duration: 0.5 }}
        >
          <Typography variant="h6" align="center" sx={{ mb: 1 }}>
            {loadingMessages[messageIndex]}
          </Typography>
        </motion.div>
        
        <Typography variant="body2" align="center" color="text.secondary">
          This may take a minute or two. Please wait...
        </Typography>
      </Box>
    </motion.div>
  );
};

export default Loading; 