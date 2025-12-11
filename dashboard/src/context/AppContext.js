import React, { createContext, useContext, useReducer, useEffect, useRef } from 'react';
import axios from 'axios';

const AppContext = createContext();

// Initial state
const initialState = {
  devices: [],
  jobs: [],
  models: [],
  stats: {
    totalDevices: 0,
    activeDevices: 0,
    totalJobs: 0,
    runningJobs: 0,
    completedJobs: 0,
    totalTrainingTime: 0
  },
  loading: {
    devices: false,
    jobs: false,
    models: false
  },
  error: null
};

// Action types
const ActionTypes = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_DEVICES: 'SET_DEVICES',
  SET_JOBS: 'SET_JOBS',
  SET_MODELS: 'SET_MODELS',
  UPDATE_DEVICE: 'UPDATE_DEVICE',
  UPDATE_JOB: 'UPDATE_JOB',
  ADD_DEVICE: 'ADD_DEVICE',
  ADD_JOB: 'ADD_JOB',
  REMOVE_DEVICE: 'REMOVE_DEVICE',
  REMOVE_JOB: 'REMOVE_JOB'
};

// Reducer
function appReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: {
          ...state.loading,
          [action.payload.resource]: action.payload.loading
        }
      };
    
    case ActionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload
      };
    
    case ActionTypes.SET_DEVICES:
      return {
        ...state,
        devices: action.payload,
        stats: {
          ...state.stats,
          totalDevices: action.payload.length,
          activeDevices: action.payload.filter(d => d.is_active).length
        }
      };
    
    case ActionTypes.SET_JOBS:
      return {
        ...state,
        jobs: action.payload,
        stats: {
          ...state.stats,
          totalJobs: action.payload.length,
          runningJobs: action.payload.filter(j => j.status === 'running').length,
          completedJobs: action.payload.filter(j => j.status === 'completed').length
        }
      };
    
    case ActionTypes.SET_MODELS:
      return {
        ...state,
        models: action.payload
      };
    
    case ActionTypes.UPDATE_DEVICE:
      return {
        ...state,
        devices: state.devices.map(device =>
          device.id === action.payload.id ? { ...device, ...action.payload } : device
        )
      };
    
    case ActionTypes.UPDATE_JOB:
      return {
        ...state,
        jobs: state.jobs.map(job =>
          job.id === action.payload.id ? { ...job, ...action.payload } : job
        )
      };
    
    case ActionTypes.ADD_DEVICE:
      return {
        ...state,
        devices: [...state.devices, action.payload]
      };
    
    case ActionTypes.ADD_JOB:
      return {
        ...state,
        jobs: [...state.jobs, action.payload]
      };
    
    case ActionTypes.REMOVE_DEVICE:
      return {
        ...state,
        devices: state.devices.filter(device => device.id !== action.payload)
      };
    
    case ActionTypes.REMOVE_JOB:
      return {
        ...state,
        jobs: state.jobs.filter(job => job.id !== action.payload)
      };
    
    default:
      return state;
  }
}

// Provider component
export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds

  // WebSocket connection management
  const connectWebSocket = () => {
    try {
      // Use backend server URL for WebSocket (not React dev server)
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const wsProtocol = apiUrl.startsWith('https') ? 'wss:' : 'ws:';
      const wsHost = apiUrl.replace(/^https?:\/\//, '').split('/')[0];
      const wsUrl = `${wsProtocol}//${wsHost}/ws`;
      
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        reconnectAttempts.current = 0;
        dispatch({ type: ActionTypes.SET_ERROR, payload: null });
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };
      
      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        wsRef.current = null;
        
        // Attempt to reconnect
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          console.log(`🔄 Reconnecting... (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, reconnectDelay);
        } else {
          console.error('❌ Max reconnection attempts reached. Falling back to polling.');
          dispatch({ type: ActionTypes.SET_ERROR, payload: 'WebSocket connection failed. Using polling mode.' });
        }
      };
      
      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      dispatch({ type: ActionTypes.SET_ERROR, payload: 'WebSocket not available. Using polling mode.' });
    }
  };

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'device_registered':
      case 'device_heartbeat':
        // Refresh devices when device status changes
        api.fetchDevices();
        break;
      
      case 'job_created':
        // Add new job to state
        dispatch({ type: ActionTypes.ADD_JOB, payload: message.job });
        // Also refresh to get full job details
        api.fetchJobs();
        break;
      
      case 'job_progress':
        // Update job progress in real-time
        dispatch({
          type: ActionTypes.UPDATE_JOB,
          payload: {
            id: message.job_id,
            progress: message.progress,
            current_epoch: message.current_epoch
          }
        });
        break;
      
      case 'pong':
        // Keep-alive message, no action needed
        break;
      
      default:
        console.log('Unknown WebSocket message type:', message.type);
    }
  };

  // API functions with throttling
  const api = {
    // Devices with retry logic
    fetchDevices: async (retries = 3) => {
      for (let attempt = 0; attempt < retries; attempt++) {
        try {
          dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'devices', loading: true } });
          const response = await axios.get('/devices', { timeout: 10000 });
          dispatch({ type: ActionTypes.SET_DEVICES, payload: response.data });
          dispatch({ type: ActionTypes.SET_ERROR, payload: null });
          return;
        } catch (error) {
          if (attempt === retries - 1) {
            console.warn('Failed to fetch devices after retries:', error.message);
            dispatch({ type: ActionTypes.SET_ERROR, payload: `Failed to load devices: ${error.message}` });
          } else {
            await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1))); // Exponential backoff
          }
        } finally {
          dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'devices', loading: false } });
        }
      }
    },

    // Jobs with retry logic
    fetchJobs: async (retries = 3) => {
      for (let attempt = 0; attempt < retries; attempt++) {
        try {
          dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'jobs', loading: true } });
          const response = await axios.get('/jobs', { timeout: 10000 });
          dispatch({ type: ActionTypes.SET_JOBS, payload: response.data });
          dispatch({ type: ActionTypes.SET_ERROR, payload: null });
          return;
        } catch (error) {
          if (attempt === retries - 1) {
            console.warn('Failed to fetch jobs after retries:', error.message);
            dispatch({ type: ActionTypes.SET_ERROR, payload: `Failed to load jobs: ${error.message}` });
          } else {
            await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
          }
        } finally {
          dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'jobs', loading: false } });
        }
      }
    },

    // Models
    fetchModels: async () => {
      try {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'models', loading: true } });
        const response = await axios.get('/models', { timeout: 10000 });
        dispatch({ type: ActionTypes.SET_MODELS, payload: response.data.models || [] });
        dispatch({ type: ActionTypes.SET_ERROR, payload: null }); // Clear any previous errors
      } catch (error) {
        console.warn('Failed to fetch models:', error.message);
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
      } finally {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'models', loading: false } });
      }
    },

    // Create job with better error handling
    createJob: async (jobData, retries = 2) => {
      for (let attempt = 0; attempt < retries; attempt++) {
        try {
          const response = await axios.post('/jobs', jobData, { timeout: 15000 });
          dispatch({ type: ActionTypes.ADD_JOB, payload: response.data });
          dispatch({ type: ActionTypes.SET_ERROR, payload: null });
          return response.data;
        } catch (error) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to create job';
          if (attempt === retries - 1) {
            dispatch({ type: ActionTypes.SET_ERROR, payload: errorMessage });
            throw new Error(errorMessage);
          } else {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
    },

    // Start job
    startJob: async (jobId) => {
      try {
        await axios.post(`/jobs/${jobId}/start`);
        dispatch({ type: ActionTypes.UPDATE_JOB, payload: { id: jobId, status: 'running' } });
      } catch (error) {
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
        throw error;
      }
    },

    // Update device
    updateDevice: (deviceId, updates) => {
      dispatch({ type: ActionTypes.UPDATE_DEVICE, payload: { id: deviceId, ...updates } });
    },

    // Update job
    updateJob: (jobId, updates) => {
      dispatch({ type: ActionTypes.UPDATE_JOB, payload: { id: jobId, ...updates } });
    },

    // Cleanup devices
    cleanupDevices: async () => {
      try {
        const response = await axios.post('/devices/cleanup');
        // Refresh devices after cleanup
        await api.fetchDevices();
        return response.data;
      } catch (error) {
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
        throw error;
      }
    }
  };

  // Initialize WebSocket and data loading
  useEffect(() => {
    let refreshInterval;
    let isActive = true;

    const initialLoad = async () => {
      try {
        await api.fetchDevices();
        await new Promise(resolve => setTimeout(resolve, 100));
        await api.fetchJobs();
        await new Promise(resolve => setTimeout(resolve, 100));
        await api.fetchModels();
      } catch (error) {
        console.warn('Initial load failed:', error);
      }
    };
    
    initialLoad();
    
    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Fallback polling (slower rate since WebSocket handles real-time updates)
    // Only used if WebSocket fails or for initial data sync
    const setupRefresh = () => {
      const hasRunningJobs = state.jobs.some(job => job.status === 'running');
      const refreshRate = hasRunningJobs ? 60000 : 120000; // 60s if training, 120s otherwise (fallback only)
      
      if (refreshInterval) clearInterval(refreshInterval);
      
      refreshInterval = setInterval(async () => {
        if (!isActive) return;
        // Only refresh if WebSocket is not connected
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          try {
            await api.fetchDevices();
            await new Promise(resolve => setTimeout(resolve, 100));
            await api.fetchJobs();
          } catch (error) {
            console.warn('Fallback polling failed:', error);
          }
        }
      }, refreshRate);
    };

    setupRefresh();

    return () => {
      isActive = false;
      if (refreshInterval) clearInterval(refreshInterval);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // Only run once on mount

  const value = {
    ...state,
    api
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
