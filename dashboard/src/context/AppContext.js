import React, { createContext, useContext, useReducer, useEffect } from 'react';
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

  // API functions with throttling
  const api = {
    // Devices
    fetchDevices: async () => {
      try {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'devices', loading: true } });
        const response = await axios.get('/devices', { timeout: 10000 });
        dispatch({ type: ActionTypes.SET_DEVICES, payload: response.data });
        dispatch({ type: ActionTypes.SET_ERROR, payload: null }); // Clear any previous errors
      } catch (error) {
        console.warn('Failed to fetch devices:', error.message);
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
      } finally {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'devices', loading: false } });
      }
    },

    // Jobs
    fetchJobs: async () => {
      try {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'jobs', loading: true } });
        const response = await axios.get('/jobs', { timeout: 10000 });
        dispatch({ type: ActionTypes.SET_JOBS, payload: response.data });
        dispatch({ type: ActionTypes.SET_ERROR, payload: null }); // Clear any previous errors
      } catch (error) {
        console.warn('Failed to fetch jobs:', error.message);
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
      } finally {
        dispatch({ type: ActionTypes.SET_LOADING, payload: { resource: 'jobs', loading: false } });
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

    // Create job
    createJob: async (jobData) => {
      try {
        const response = await axios.post('/jobs', jobData);
        dispatch({ type: ActionTypes.ADD_JOB, payload: response.data });
        return response.data;
      } catch (error) {
        dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
        throw error;
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
    }
  };

  // Auto-refresh data
  useEffect(() => {
    // Initial load with delay to prevent resource exhaustion
    const initialLoad = () => {
      api.fetchDevices();
      setTimeout(() => api.fetchJobs(), 100);
      setTimeout(() => api.fetchModels(), 200);
    };
    
    initialLoad();

    // Set up auto-refresh with longer interval
    const interval = setInterval(() => {
      api.fetchDevices();
      setTimeout(() => api.fetchJobs(), 100);
    }, 60000); // Refresh every 60 seconds (increased from 30)

    return () => clearInterval(interval);
  }, []); // Removed api from dependency array

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
