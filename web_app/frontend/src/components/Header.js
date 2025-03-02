import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <AppBar 
      position="static" 
      color="transparent" 
      elevation={0}
      sx={{ 
        background: 'transparent',
        borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
        mb: 3
      }}
    >
      <Toolbar>
        <motion.div
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ShowChartIcon 
              sx={{ 
                mr: 1.5, 
                fontSize: '2rem',
                color: '#3f51b5'
              }} 
            />
            <Typography 
              variant="h5" 
              component="div" 
              sx={{ 
                fontWeight: 500,
                background: 'linear-gradient(45deg, #3f51b5 30%, #f50057 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              AlgoTrade Pro
            </Typography>
          </Box>
        </motion.div>
        
        <Box sx={{ flexGrow: 1 }} />
        
        <motion.div
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Typography variant="body2" color="text.secondary">
            Powered by AI and Market Intelligence
          </Typography>
        </motion.div>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 