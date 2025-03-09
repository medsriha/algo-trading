import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Define global theme constants
export const THEME_CONSTANTS = {
  colors: {
    background: '#f9f9fa',
    cardBackground: '#ffffff',
    primary: '#4c6ef5',
    secondary: '#f50057',
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

// Create a light theme with off-white background
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: THEME_CONSTANTS.colors.primary,
    },
    secondary: {
      main: THEME_CONSTANTS.colors.secondary,
    },
    success: {
      main: THEME_CONSTANTS.colors.success,
    },
    error: {
      main: THEME_CONSTANTS.colors.error,
    },
    background: {
      default: THEME_CONSTANTS.colors.background,
      paper: THEME_CONSTANTS.colors.cardBackground,
    },
    text: {
      primary: THEME_CONSTANTS.colors.textPrimary,
      secondary: THEME_CONSTANTS.colors.textSecondary,
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    h2: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    h3: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    h4: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    h5: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    h6: {
      fontWeight: 600,
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    body1: {
      color: THEME_CONSTANTS.colors.textPrimary,
    },
    body2: {
      color: THEME_CONSTANTS.colors.textSecondary,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: THEME_CONSTANTS.radius.small,
          textTransform: 'none',
          fontWeight: 600,
        },
        contained: {
          boxShadow: THEME_CONSTANTS.shadows.small,
        },
        outlined: {
          borderColor: THEME_CONSTANTS.colors.border,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: THEME_CONSTANTS.radius.medium,
          boxShadow: THEME_CONSTANTS.shadows.small,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: THEME_CONSTANTS.radius.medium,
          boxShadow: THEME_CONSTANTS.shadows.small,
          border: `1px solid ${THEME_CONSTANTS.colors.border}`,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: THEME_CONSTANTS.radius.small,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: THEME_CONSTANTS.radius.small,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: THEME_CONSTANTS.radius.small,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={lightTheme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
); 