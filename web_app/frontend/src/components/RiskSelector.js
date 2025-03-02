import React from 'react';
import { FormControl, InputLabel, Select, MenuItem, Box, Button, Tooltip } from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import BalanceIcon from '@mui/icons-material/Balance';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InfoIcon from '@mui/icons-material/Info';
import { motion } from 'framer-motion';

const RiskSelector = ({ onSelect, selectedRisk }) => {
  const [risk, setRisk] = React.useState('');
  
  const handleChange = (event) => {
    setRisk(event.target.value);
  };

  const handleSubmit = () => {
    if (risk) {
      onSelect(risk);
    }
  };

  // Define risk option information
  const riskInfo = {
    conservative: {
      icon: <SecurityIcon sx={{ color: '#4caf50' }} />,
      description: 'Lower risk, more stable returns, fewer losses'
    },
    moderate: {
      icon: <BalanceIcon sx={{ color: '#2196f3' }} />,
      description: 'Balanced approach with moderate risk and return potential'
    },
    aggressive: {
      icon: <TrendingUpIcon sx={{ color: '#f44336' }} />,
      description: 'Higher risk for potentially greater returns, more tolerance for losses'
    }
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' }, 
        alignItems: 'center',
        gap: 2,
        mb: 4,
        mx: 'auto',
        maxWidth: 600
      }}>
        <FormControl fullWidth variant="outlined">
          <InputLabel id="risk-selector-label">Risk Profile</InputLabel>
          <Select
            labelId="risk-selector-label"
            id="risk-selector"
            value={risk}
            onChange={handleChange}
            label="Risk Profile"
            sx={{ 
              borderRadius: 2,
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(255, 255, 255, 0.23)'
              }
            }}
          >
            {Object.entries(riskInfo).map(([key, { icon, description }]) => (
              <MenuItem key={key} value={key} sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  {icon}
                  <span style={{ textTransform: 'capitalize' }}>{key}</span>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={!risk}
          sx={{ 
            borderRadius: 2, 
            py: 1.5,
            px: 4,
            minWidth: { xs: '100%', sm: '150px' }
          }}
        >
          Analyze
        </Button>
      </Box>

      {risk && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1.5,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            p: 2,
            borderRadius: 2,
            mb: 3
          }}>
            <InfoIcon sx={{ color: '#64b5f6' }} />
            <Box sx={{ color: '#ccc' }}>
              {riskInfo[risk].description}
            </Box>
          </Box>
        </motion.div>
      )}
    </motion.div>
  );
};

export default RiskSelector; 