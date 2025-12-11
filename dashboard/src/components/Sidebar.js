import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Monitor, 
  Brain, 
  Database, 
  Settings,
  Activity
} from 'lucide-react';

function Sidebar() {
  const navigation = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Devices', href: '/devices', icon: Monitor },
    { name: 'Training Jobs', href: '/jobs', icon: Brain },
    { name: 'Models', href: '/models', icon: Database },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  return (
    <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
      <nav className="mt-8 px-4">
        <div className="space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  `group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-blue-100 text-blue-700 border-r-2 border-blue-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`
                }
              >
                <Icon
                  className={`mr-3 h-5 w-5 flex-shrink-0 ${
                    window.location.pathname === item.href
                      ? 'text-blue-500'
                      : 'text-gray-400 group-hover:text-gray-500'
                  }`}
                />
                {item.name}
              </NavLink>
            );
          })}
        </div>
      </nav>
      
      {/* System Status */}
      <div className="mt-8 px-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-900 mb-3">System Status</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Server</span>
              <span className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                Online
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Training</span>
              <span className="flex items-center">
                <Activity className="w-4 h-4 text-blue-500 mr-2" />
                Active
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
