import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { Monitor, Cpu, HardDrive, Wifi, WifiOff, Activity } from 'lucide-react';

function Devices() {
  const { devices, loading } = useApp();
  const [filter, setFilter] = useState('all');

  const filteredDevices = devices.filter(device => {
    if (filter === 'active') return device.is_active;
    if (filter === 'offline') return !device.is_active;
    return true;
  });

  const DeviceCard = ({ device }) => (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-lg ${device.is_active ? 'bg-green-100' : 'bg-gray-100'}`}>
            <Monitor className={`h-6 w-6 ${device.is_active ? 'text-green-600' : 'text-gray-400'}`} />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900">{device.name}</h3>
            <p className="text-sm text-gray-500 capitalize">{device.device_type.replace('_', ' ')}</p>
            <p className="text-xs text-gray-400">ID: {device.id}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {device.is_active ? (
            <Wifi className="h-4 w-4 text-green-500" />
          ) : (
            <WifiOff className="h-4 w-4 text-gray-400" />
          )}
          <span className={`status-badge ${device.is_active ? 'status-active' : 'status-offline'}`}>
            {device.is_active ? 'Active' : 'Offline'}
          </span>
        </div>
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="flex items-center space-x-2">
          <Cpu className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-600">{device.cpu_cores} cores</span>
        </div>
        <div className="flex items-center space-x-2">
          <HardDrive className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-600">{device.memory_gb} GB RAM</span>
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-600">
            {device.gpu_available ? 'GPU Available' : 'CPU Only'}
          </span>
        </div>
        <div className="text-sm text-gray-500">
          Last seen: {new Date(device.last_seen).toLocaleString()}
        </div>
      </div>
    </div>
  );

  if (loading.devices) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="card">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2 mb-4"></div>
                <div className="space-y-2">
                  <div className="h-3 bg-gray-200 rounded"></div>
                  <div className="h-3 bg-gray-200 rounded"></div>
                </div>
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Devices</h1>
          <p className="text-gray-600">Manage and monitor connected devices</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="input w-auto"
          >
            <option value="all">All Devices</option>
            <option value="active">Active Only</option>
            <option value="offline">Offline Only</option>
          </select>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-blue-100">
              <Monitor className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Devices</p>
              <p className="text-2xl font-semibold text-gray-900">{devices.length}</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-green-100">
              <Wifi className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active Devices</p>
              <p className="text-2xl font-semibold text-gray-900">
                {devices.filter(d => d.is_active).length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-purple-100">
              <Activity className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">GPU Devices</p>
              <p className="text-2xl font-semibold text-gray-900">
                {devices.filter(d => d.gpu_available).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Device Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredDevices.map((device) => (
          <DeviceCard key={device.id} device={device} />
        ))}
      </div>

      {filteredDevices.length === 0 && (
        <div className="text-center py-12">
          <Monitor className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No devices found</h3>
          <p className="text-gray-500">
            {filter === 'all' 
              ? 'No devices are currently registered.' 
              : `No devices match the "${filter}" filter.`
            }
          </p>
        </div>
      )}
    </div>
  );
}

export default Devices;
