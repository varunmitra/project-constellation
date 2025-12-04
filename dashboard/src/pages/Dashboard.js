import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { 
  Monitor, 
  Brain, 
  Clock, 
  Activity,
  TrendingUp,
  Users,
  AlertCircle,
  RefreshCw,
  CheckCircle,
  XCircle,
  Play,
  Pause
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

function Dashboard() {
  const { devices, jobs, stats, loading, error, api } = useApp();
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showError, setShowError] = useState(false);
  const [isCleaningUp, setIsCleaningUp] = useState(false);

  // Auto-update last update time
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Handle errors
  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => setShowError(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Manual refresh function
  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await Promise.all([
        api.fetchDevices(),
        api.fetchJobs(),
        api.fetchModels()
      ]);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Manual refresh failed:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleCleanup = async () => {
    setIsCleaningUp(true);
    try {
      const result = await api.cleanupDevices();
      setLastUpdate(new Date());
      console.log('Cleanup result:', result);
    } catch (error) {
      setShowError(true);
      console.error('Cleanup failed:', error);
    } finally {
      setIsCleaningUp(false);
    }
  };

  // Mock data for charts
  const trainingData = [
    { time: '00:00', devices: 12, training: 8 },
    { time: '04:00', devices: 15, training: 12 },
    { time: '08:00', devices: 8, training: 2 },
    { time: '12:00', devices: 6, training: 1 },
    { time: '16:00', devices: 10, training: 5 },
    { time: '20:00', devices: 18, training: 14 },
  ];

  const deviceTypes = [
    { type: 'MacBook Pro', count: 45 },
    { type: 'iMac', count: 23 },
    { type: 'Mac Studio', count: 12 },
  ];

  const StatCard = ({ title, value, icon: Icon, color, subtitle, trend, isLoading }) => (
    <div className="card hover:shadow-lg transition-shadow duration-200">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${color} ${isLoading ? 'animate-pulse' : ''}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">
            {isLoading ? '...' : value}
          </p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
          {trend && (
            <div className={`flex items-center text-xs mt-1 ${
              trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-gray-500'
            }`}>
              {trend > 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : 
               trend < 0 ? <TrendingUp className="h-3 w-3 mr-1 rotate-180" /> : null}
              {trend !== 0 && `${Math.abs(trend)}%`}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  if (loading.devices || loading.jobs) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">Overview of your distributed AI training infrastructure</p>
          <div className="flex items-center mt-2 space-x-4">
            <div className="flex items-center text-sm text-gray-500">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                stats.runningJobs > 0 ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              }`}></div>
              {stats.runningJobs > 0 ? 'Live Training' : 'Idle'}
            </div>
            <div className="text-sm text-gray-500">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCleanup}
            disabled={isCleaningUp}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${isCleaningUp ? 'animate-spin' : ''}`} />
            <span>{isCleaningUp ? 'Cleaning...' : 'Cleanup Devices'}</span>
          </button>
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {showError && error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
          <div className="flex-1">
            <p className="text-sm text-red-800 font-medium">Connection Error</p>
            <p className="text-sm text-red-600">{error}</p>
          </div>
          <button
            onClick={() => setShowError(false)}
            className="text-red-500 hover:text-red-700"
          >
            <XCircle className="h-5 w-5" />
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Devices"
          value={stats.totalDevices}
          icon={Monitor}
          color="bg-blue-500"
          subtitle={`${stats.activeDevices} active`}
          isLoading={loading.devices}
          trend={stats.totalDevices > 0 ? 5 : 0}
        />
        <StatCard
          title="Training Jobs"
          value={stats.totalJobs}
          icon={Brain}
          color="bg-green-500"
          subtitle={`${stats.runningJobs} running`}
          isLoading={loading.jobs}
          trend={stats.runningJobs > 0 ? 12 : 0}
        />
        <StatCard
          title="Completed Jobs"
          value={stats.completedJobs}
          icon={Clock}
          color="bg-purple-500"
          subtitle="All time"
          isLoading={loading.jobs}
        />
        <StatCard
          title="Success Rate"
          value={`${stats.totalJobs > 0 ? Math.round((stats.completedJobs / stats.totalJobs) * 100) : 0}%`}
          icon={Activity}
          color="bg-orange-500"
          subtitle="Job completion"
          isLoading={loading.jobs}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Activity Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Activity (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="devices" 
                stroke="#3B82F6" 
                strokeWidth={2}
                name="Active Devices"
              />
              <Line 
                type="monotone" 
                dataKey="training" 
                stroke="#10B981" 
                strokeWidth={2}
                name="Training"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Device Types Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Device Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={deviceTypes}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Jobs */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Recent Training Jobs</h3>
            <span className="text-sm text-gray-500">{jobs.length} total</span>
          </div>
          <div className="space-y-3">
            {jobs.slice(0, 5).map((job) => (
              <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <p className="font-medium text-gray-900">{job.name}</p>
                    {job.status === 'running' && (
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    )}
                  </div>
                  <p className="text-sm text-gray-500">{job.model_type}</p>
                  {job.status === 'running' && (
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>{Math.round(job.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div 
                          className="bg-green-500 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${job.progress}%` }}
                        ></div>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Epoch {job.current_epoch}/{job.total_epochs}</span>
                        <span>{job.dataset}</span>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`status-badge status-${job.status} flex items-center space-x-1`}>
                    {job.status === 'completed' && <CheckCircle className="h-3 w-3" />}
                    {job.status === 'running' && <Play className="h-3 w-3" />}
                    {job.status === 'pending' && <Pause className="h-3 w-3" />}
                    <span className="capitalize">{job.status}</span>
                  </span>
                </div>
              </div>
            ))}
            {jobs.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <Brain className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>No training jobs yet</p>
                <p className="text-sm">Create your first job to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Device Status */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Active Devices</h3>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">{devices.length} active</span>
              <button
                onClick={handleCleanup}
                disabled={isCleaningUp}
                className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
              >
                {isCleaningUp ? 'Cleaning...' : 'Cleanup'}
              </button>
            </div>
          </div>
          <div className="space-y-3">
            {devices.slice(0, 5).map((device) => {
              const lastSeen = new Date(device.last_seen);
              const timeSinceLastSeen = Date.now() - lastSeen.getTime();
              const isRecentlyActive = timeSinceLastSeen < 5 * 60 * 1000; // 5 minutes
              const isStale = timeSinceLastSeen > 10 * 60 * 1000; // 10 minutes
              
              return (
                <div key={device.id} className={`flex items-center justify-between p-3 rounded-lg hover:bg-gray-100 transition-colors ${
                  isStale ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50'
                }`}>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <p className="font-medium text-gray-900">{device.name}</p>
                      {device.is_active && isRecentlyActive && (
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      )}
                      {isStale && (
                        <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      )}
                    </div>
                    <p className="text-sm text-gray-500">{device.device_type}</p>
                    <div className="flex items-center space-x-4 mt-1">
                      <span className="text-xs text-gray-500">
                        {device.cpu_cores} cores, {device.memory_gb}GB RAM
                      </span>
                      {device.gpu_available && (
                        <span className="text-xs text-blue-600 font-medium">
                          GPU: {device.gpu_memory_gb}GB
                        </span>
                      )}
                    </div>
                    <p className={`text-xs mt-1 ${
                      isRecentlyActive ? 'text-green-600' : 
                      isStale ? 'text-yellow-600' : 'text-gray-400'
                    }`}>
                      Last seen: {isRecentlyActive ? 'Just now' : lastSeen.toLocaleTimeString()}
                      {isStale && ' (Stale)'}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`status-badge ${
                      device.is_active && isRecentlyActive ? 'status-active' : 
                      device.is_active && isStale ? 'status-idle' : 'status-offline'
                    } flex items-center space-x-1`}>
                      {device.is_active && isRecentlyActive ? <CheckCircle className="h-3 w-3" /> : 
                       device.is_active && isStale ? <AlertCircle className="h-3 w-3" /> : 
                       <XCircle className="h-3 w-3" />}
                      <span>
                        {device.is_active && isRecentlyActive ? 'Active' : 
                         device.is_active && isStale ? 'Stale' : 'Offline'}
                      </span>
                    </span>
                  </div>
                </div>
              );
            })}
            {devices.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <Monitor className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>No active devices</p>
                <p className="text-sm">Start the Constellation app to connect devices</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="flex space-x-4">
          <button className="btn-primary">
            <Brain className="h-4 w-4 mr-2" />
            Create Training Job
          </button>
          <button className="btn-secondary">
            <Users className="h-4 w-4 mr-2" />
            View All Devices
          </button>
          <button className="btn-secondary">
            <TrendingUp className="h-4 w-4 mr-2" />
            View Analytics
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
