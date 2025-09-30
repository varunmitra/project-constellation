import React, { useState } from 'react';
import { Settings as SettingsIcon, Server, Shield, Bell, Save } from 'lucide-react';

function Settings() {
  const [settings, setSettings] = useState({
    server: {
      url: 'http://localhost:8000',
      timeout: 30,
      retries: 3
    },
    notifications: {
      email: true,
      desktop: true,
      trainingComplete: true,
      deviceOffline: true
    },
    security: {
      requireAuth: true,
      sessionTimeout: 60,
      logLevel: 'info'
    },
    training: {
      maxConcurrentJobs: 5,
      defaultEpochs: 10,
      autoStart: false
    }
  });

  const handleSave = () => {
    // Save settings to localStorage or API
    localStorage.setItem('constellation-settings', JSON.stringify(settings));
    alert('Settings saved successfully!');
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset all settings to default?')) {
      setSettings({
        server: {
          url: 'http://localhost:8000',
          timeout: 30,
          retries: 3
        },
        notifications: {
          email: true,
          desktop: true,
          trainingComplete: true,
          deviceOffline: true
        },
        security: {
          requireAuth: true,
          sessionTimeout: 60,
          logLevel: 'info'
        },
        training: {
          maxConcurrentJobs: 5,
          defaultEpochs: 10,
          autoStart: false
        }
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Configure your Constellation deployment</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Server Settings */}
        <div className="card">
          <div className="flex items-center mb-4">
            <Server className="h-5 w-5 text-blue-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Server Configuration</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Server URL
              </label>
              <input
                type="url"
                value={settings.server.url}
                onChange={(e) => setSettings({
                  ...settings,
                  server: { ...settings.server, url: e.target.value }
                })}
                className="input"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Timeout (seconds)
              </label>
              <input
                type="number"
                value={settings.server.timeout}
                onChange={(e) => setSettings({
                  ...settings,
                  server: { ...settings.server, timeout: parseInt(e.target.value) }
                })}
                className="input"
                min="5"
                max="300"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Retry Attempts
              </label>
              <input
                type="number"
                value={settings.server.retries}
                onChange={(e) => setSettings({
                  ...settings,
                  server: { ...settings.server, retries: parseInt(e.target.value) }
                })}
                className="input"
                min="0"
                max="10"
              />
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="card">
          <div className="flex items-center mb-4">
            <Bell className="h-5 w-5 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Email Notifications</span>
              <input
                type="checkbox"
                checked={settings.notifications.email}
                onChange={(e) => setSettings({
                  ...settings,
                  notifications: { ...settings.notifications, email: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Desktop Notifications</span>
              <input
                type="checkbox"
                checked={settings.notifications.desktop}
                onChange={(e) => setSettings({
                  ...settings,
                  notifications: { ...settings.notifications, desktop: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Training Complete</span>
              <input
                type="checkbox"
                checked={settings.notifications.trainingComplete}
                onChange={(e) => setSettings({
                  ...settings,
                  notifications: { ...settings.notifications, trainingComplete: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Device Offline</span>
              <input
                type="checkbox"
                checked={settings.notifications.deviceOffline}
                onChange={(e) => setSettings({
                  ...settings,
                  notifications: { ...settings.notifications, deviceOffline: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
          </div>
        </div>

        {/* Security Settings */}
        <div className="card">
          <div className="flex items-center mb-4">
            <Shield className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Security</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Require Authentication</span>
              <input
                type="checkbox"
                checked={settings.security.requireAuth}
                onChange={(e) => setSettings({
                  ...settings,
                  security: { ...settings.security, requireAuth: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Session Timeout (minutes)
              </label>
              <input
                type="number"
                value={settings.security.sessionTimeout}
                onChange={(e) => setSettings({
                  ...settings,
                  security: { ...settings.security, sessionTimeout: parseInt(e.target.value) }
                })}
                className="input"
                min="5"
                max="480"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Log Level
              </label>
              <select
                value={settings.security.logLevel}
                onChange={(e) => setSettings({
                  ...settings,
                  security: { ...settings.security, logLevel: e.target.value }
                })}
                className="input"
              >
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warn">Warning</option>
                <option value="error">Error</option>
              </select>
            </div>
          </div>
        </div>

        {/* Training Settings */}
        <div className="card">
          <div className="flex items-center mb-4">
            <SettingsIcon className="h-5 w-5 text-purple-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Training</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Concurrent Jobs
              </label>
              <input
                type="number"
                value={settings.training.maxConcurrentJobs}
                onChange={(e) => setSettings({
                  ...settings,
                  training: { ...settings.training, maxConcurrentJobs: parseInt(e.target.value) }
                })}
                className="input"
                min="1"
                max="20"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Default Epochs
              </label>
              <input
                type="number"
                value={settings.training.defaultEpochs}
                onChange={(e) => setSettings({
                  ...settings,
                  training: { ...settings.training, defaultEpochs: parseInt(e.target.value) }
                })}
                className="input"
                min="1"
                max="1000"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Auto-start Jobs</span>
              <input
                type="checkbox"
                checked={settings.training.autoStart}
                onChange={(e) => setSettings({
                  ...settings,
                  training: { ...settings.training, autoStart: e.target.checked }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <button onClick={handleReset} className="btn-secondary">
          Reset to Default
        </button>
        <button onClick={handleSave} className="btn-primary">
          <Save className="h-4 w-4 mr-2" />
          Save Settings
        </button>
      </div>
    </div>
  );
}

export default Settings;
