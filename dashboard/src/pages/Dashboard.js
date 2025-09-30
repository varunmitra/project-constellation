import React from 'react';
import { useApp } from '../context/AppContext';
import { 
  Monitor, 
  Brain, 
  Clock, 
  Activity,
  TrendingUp,
  Users
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

function Dashboard() {
  const { devices, jobs, stats, loading } = useApp();

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

  const StatCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className="card">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
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
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Overview of your distributed AI training infrastructure</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Devices"
          value={stats.totalDevices}
          icon={Monitor}
          color="bg-blue-500"
          subtitle={`${stats.activeDevices} active`}
        />
        <StatCard
          title="Training Jobs"
          value={stats.totalJobs}
          icon={Brain}
          color="bg-green-500"
          subtitle={`${stats.runningJobs} running`}
        />
        <StatCard
          title="Completed Jobs"
          value={stats.completedJobs}
          icon={Clock}
          color="bg-purple-500"
          subtitle="This month"
        />
        <StatCard
          title="Training Time"
          value="1,247h"
          icon={Activity}
          color="bg-orange-500"
          subtitle="Total compute time"
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
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Training Jobs</h3>
          <div className="space-y-3">
            {jobs.slice(0, 5).map((job) => (
              <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{job.name}</p>
                  <p className="text-sm text-gray-500">{job.model_type}</p>
                </div>
                <span className={`status-badge status-${job.status}`}>
                  {job.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Device Status */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Device Status</h3>
          <div className="space-y-3">
            {devices.slice(0, 5).map((device) => (
              <div key={device.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{device.name}</p>
                  <p className="text-sm text-gray-500">{device.device_type}</p>
                </div>
                <span className={`status-badge ${device.is_active ? 'status-active' : 'status-offline'}`}>
                  {device.is_active ? 'Active' : 'Offline'}
                </span>
              </div>
            ))}
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
