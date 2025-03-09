import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { motion } from 'framer-motion';
import { THEME_CONSTANTS } from '../index';

const Header = () => {
  return (
    <AppBar 
      position="static" 
      color="transparent" 
      elevation={0}
      sx={{ 
        background: THEME_CONSTANTS.colors.cardBackground,
        borderBottom: `1px solid ${THEME_CONSTANTS.colors.border}`,
        mb: 3,
        borderRadius: 0
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
                color: THEME_CONSTANTS.colors.primary
              }} 
            />
            <Typography 
              variant="h5" 
              component="div" 
              sx={{ 
                fontWeight: 700,
                color: THEME_CONSTANTS.colors.textPrimary
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
          <Typography 
            variant="body2"
            sx={{ 
              color: THEME_CONSTANTS.colors.textSecondary,
              fontWeight: 500
            }}
          >
            Powered by AI and Market Intelligence
          </Typography>
        </motion.div>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 