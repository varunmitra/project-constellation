import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Devices from './pages/Devices';
import Jobs from './pages/Jobs';
import Models from './pages/Models';
import Settings from './pages/Settings';

// Context
import { AppProvider } from './context/AppContext';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize app
    const initializeApp = async () => {
      try {
        // Test server connection
        await axios.get('/health');
        setIsLoading(false);
      } catch (err) {
        setError('Failed to connect to server. Please check if the server is running.');
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading Constellation Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Connection Error</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <AppProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Header />
          <div className="flex">
            <Sidebar />
            <main className="flex-1 p-6">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/devices" element={<Devices />} />
                <Route path="/jobs" element={<Jobs />} />
                <Route path="/models" element={<Models />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
