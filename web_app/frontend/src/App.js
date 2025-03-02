import React, { useState } from 'react';
import { Container, Box, Typography, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import RiskSelector from './components/RiskSelector';
import ResultsDisplay from './components/ResultsDisplay';
import Loading from './components/Loading';
import Header from './components/Header';
import axios from 'axios';

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
    >
      <Container maxWidth="lg">
        <Header />
        
        <Box sx={{ my: 4 }}>
          <Paper 
            elevation={4} 
            sx={{ 
              p: 4, 
              borderRadius: 3,
              background: 'linear-gradient(145deg, #1e1e1e 0%, #2d2d2d 100%)',
              boxShadow: '0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23)'
            }}
          >
            <Typography 
              variant="h4" 
              component="h1" 
              gutterBottom 
              sx={{ 
                textAlign: 'center',
                fontWeight: 500,
                mb: 4,
                background: 'linear-gradient(45deg, #3f51b5 10%, #f50057 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              Algo Trading Analysis Tool
            </Typography>
            
            <Typography 
              variant="h6" 
              component="h2" 
              sx={{ mb: 3, textAlign: 'center', color: '#ccc' }}
            >
              Select your risk profile to get stock recommendations
            </Typography>
            
            <RiskSelector onSelect={handleRiskSelection} selectedRisk={riskLevel} />
            
            {loading && <Loading />}
            
            {error && (
              <Box sx={{ mt: 4, p: 2, bgcolor: 'error.dark', borderRadius: 2, color: 'white' }}>
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