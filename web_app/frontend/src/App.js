import React, { useState } from 'react';
import { Container, Box, Typography, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import RiskSelector from './components/RiskSelector';
import ResultsDisplay from './components/ResultsDisplay';
import Loading from './components/Loading';
import Header from './components/Header';
import axios from 'axios';
import { THEME_CONSTANTS } from './index';

function App() {
  const [riskLevel, setRiskLevel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleRiskSelection = async (selectedRisk) => {
    setRiskLevel(selectedRisk);
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/analyze', {
        risk_level: selectedRisk
      });
      
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during analysis');
      console.error('Error during analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      style={{ backgroundColor: THEME_CONSTANTS.colors.background, minHeight: '100vh' }}
    >
      <Container maxWidth="lg">
        <Header />
        
        <Box sx={{ my: 4 }}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: { xs: 2, sm: 4 }, 
              borderRadius: THEME_CONSTANTS.radius.large,
              background: THEME_CONSTANTS.colors.cardBackground,
              border: `1px solid ${THEME_CONSTANTS.colors.border}`,
              boxShadow: THEME_CONSTANTS.shadows.small
            }}
          >
            <Typography 
              variant="h4" 
              component="h1" 
              gutterBottom 
              sx={{ 
                textAlign: 'center',
                fontWeight: 600,
                mb: 4,
                color: THEME_CONSTANTS.colors.textPrimary
              }}
            >
              Algo Trading Analysis Tool
            </Typography>
            
            <Typography 
              variant="h6" 
              component="h2" 
              sx={{ mb: 3, textAlign: 'center', color: THEME_CONSTANTS.colors.textSecondary }}
            >
              Select your risk profile to get stock recommendations
            </Typography>
            
            <RiskSelector onSelect={handleRiskSelection} selectedRisk={riskLevel} />
            
            {loading && <Loading />}
            
            {error && (
              <Box sx={{ 
                mt: 4, 
                p: 3, 
                bgcolor: 'rgba(245,101,101,0.1)', 
                borderRadius: THEME_CONSTANTS.radius.medium, 
                color: THEME_CONSTANTS.colors.error,
                border: `1px solid ${THEME_CONSTANTS.colors.error}20`
              }}>
                <Typography variant="body1">{error}</Typography>
              </Box>
            )}
            
            {!loading && results && <ResultsDisplay results={results} />}
          </Paper>
        </Box>
      </Container>
    </motion.div>
  );
}

export default App; 