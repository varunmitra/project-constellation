import React from 'react';
import { Brain, Wifi } from 'lucide-react';

function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">Project Constellation</h1>
              <p className="text-sm text-gray-500">Decentralized AI Training Infrastructure</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <Wifi className="h-4 w-4 text-green-500" />
              <span className="text-sm text-gray-600">Connected</span>
            </div>
            
            {/* Last Updated */}
            <div className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
