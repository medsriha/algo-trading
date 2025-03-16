import React from 'react';
import { 
  Box, 
  Button, 
  Slider, 
  Typography,
  Paper
} from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import BalanceIcon from '@mui/icons-material/Balance';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';
import { motion } from 'framer-motion';
import { THEME_CONSTANTS } from '../index';

const RiskSelector = ({ onSelect, selectedRisk }) => {
  const [sliderValue, setSliderValue] = React.useState(50);
  
  const handleSliderChange = (event, newValue) => {
    setSliderValue(newValue);
  };

  const handleSubmit = () => {
    // Convert slider value to risk profile
    let risk;
    if (sliderValue <= 33) {
      risk = 'conservative';
    } else if (sliderValue <= 66) {
      risk = 'moderate';
    } else {
      risk = 'aggressive';
    }
    onSelect(risk);
  };

  // Get the appropriate color based on slider value
  const getColor = (value) => {
    if (value <= 33) return THEME_CONSTANTS.colors.success;
    if (value <= 66) return THEME_CONSTANTS.colors.primary;
    return THEME_CONSTANTS.colors.error;
  };

  // Get the appropriate description based on slider value
  const getDescription = (value) => {
    if (value <= 33) {
      return "You're feeling cautious today. Let's focus on stable, lower-risk opportunities.";
    } else if (value <= 66) {
      return "You're feeling balanced today. We'll look for moderate opportunities with reasonable risk.";
    }
    return "You're feeling adventurous today. We'll explore higher-risk opportunities with greater potential returns.";
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <Paper 
        elevation={0}
        sx={{ 
          p: 4, 
          borderRadius: THEME_CONSTANTS.radius.medium,
          border: `1px solid ${THEME_CONSTANTS.colors.border}`,
          bgcolor: THEME_CONSTANTS.colors.background,
          maxWidth: 600,
          mx: 'auto',
          mb: 4
        }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <SentimentSatisfiedAltIcon /> How are you feeling about the market today?
          </Typography>
          
          <Box sx={{ px: 2, mt: 4 }}>
            <Slider
              value={sliderValue}
              onChange={handleSliderChange}
              aria-labelledby="risk-slider"
              valueLabelDisplay="off"
              sx={{
                '& .MuiSlider-thumb': {
                  height: 24,
                  width: 24,
                  backgroundColor: getColor(sliderValue),
                },
                '& .MuiSlider-track': {
                  backgroundColor: getColor(sliderValue),
                },
                '& .MuiSlider-rail': {
                  opacity: 0.5,
                  backgroundColor: '#bfbfbf',
                },
              }}
            />
          </Box>

          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            mt: 1,
            color: THEME_CONSTANTS.colors.textSecondary,
            position: 'relative'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <SecurityIcon sx={{ color: THEME_CONSTANTS.colors.success }} />
              <Typography variant="body2">Conservative</Typography>
            </Box>
            <Box sx={{ 
              position: 'absolute', 
              left: '50%', 
              transform: 'translateX(-50%)',
              display: 'flex', 
              alignItems: 'center',
              gap: 0.5
            }}>
              <BalanceIcon sx={{ color: THEME_CONSTANTS.colors.primary }} />
              <Typography variant="body2">Moderate</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <TrendingUpIcon sx={{ color: THEME_CONSTANTS.colors.error }} />
              <Typography variant="body2">Aggressive</Typography>
            </Box>
          </Box>
        </Box>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            gap: 2,
            alignItems: 'center',
            textAlign: 'center'
          }}>
            <Typography 
              variant="body1" 
              sx={{ 
                color: getColor(sliderValue),
                fontWeight: 500
              }}
            >
              {getDescription(sliderValue)}
            </Typography>

            <Button
              variant="contained"
              color="primary"
              onClick={handleSubmit}
              sx={{ 
                borderRadius: THEME_CONSTANTS.radius.small, 
                py: 1.5,
                px: 4,
                minWidth: 200
              }}
            >
              Analyze
            </Button>
          </Box>
        </motion.div>
      </Paper>
    </motion.div>
  );
};

export default RiskSelector; 