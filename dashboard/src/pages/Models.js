import React from 'react';
import { useApp } from '../context/AppContext';
import { Database, Download, Upload, Trash2, Eye } from 'lucide-react';

function Models() {
  const { models, loading } = useApp();
  
  console.log('🔧 Models component rendered with:', { models: models.length, loading });

  const formatFileSize = (sizeMb) => {
    if (!sizeMb || sizeMb === 0) return '0 MB';
    if (sizeMb < 1) {
      return (sizeMb * 1024).toFixed(2) + ' KB';
    }
    return sizeMb.toFixed(2) + ' MB';
  };

  const handleViewModel = (model) => {
    console.log('🔍 View model clicked:', model);
    alert(`Model Details:\nName: ${model.name}\nType: ${model.model_type}\nStatus: ${model.status}\nPath: ${model.checkpoint_path}`);
  };

  const handleDownloadModel = (model) => {
    console.log('⬇️ Download model clicked:', model);
    // For now, just show an alert. In a real app, this would trigger a download
    alert(`Downloading model: ${model.name}\nPath: ${model.checkpoint_path}`);
  };

  const handleDeleteModel = (model) => {
    console.log('🗑️ Delete model clicked:', model);
    if (window.confirm(`Are you sure you want to delete "${model.name}"?`)) {
      alert(`Model "${model.name}" would be deleted (not implemented yet)`);
    }
  };

  const handleUploadModel = () => {
    console.log('📤 Upload model clicked');
    alert('Upload model functionality not implemented yet');
  };

  const ModelCard = ({ model }) => (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4">
          <div className="p-3 rounded-lg bg-purple-100">
            <Database className="h-6 w-6 text-purple-600" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
            <p className="text-sm text-gray-500">Size: {formatFileSize(model.size_mb)}</p>
            <p className="text-xs text-gray-400">Created: {new Date(model.created_at).toLocaleString()}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button 
            className="p-2 text-gray-400 hover:text-gray-600 cursor-pointer"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              console.log('🔍 Eye button clicked for model:', model.name);
              handleViewModel(model);
            }}
            title="View model details"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button 
            className="p-2 text-gray-400 hover:text-gray-600 cursor-pointer"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              console.log('⬇️ Download button clicked for model:', model.name);
              handleDownloadModel(model);
            }}
            title="Download model"
          >
            <Download className="h-4 w-4" />
          </button>
          <button 
            className="p-2 text-red-400 hover:text-red-600 cursor-pointer"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              console.log('🗑️ Delete button clicked for model:', model.name);
              handleDeleteModel(model);
            }}
            title="Delete model"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  if (loading.models) {
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
          <h1 className="text-2xl font-bold text-gray-900">Models</h1>
          <p className="text-gray-600">Manage trained AI models and checkpoints</p>
        </div>
        <div className="flex items-center space-x-4">
          <button 
            className="btn-secondary cursor-pointer"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              console.log('📤 Upload Model button clicked');
              handleUploadModel();
            }}
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Model
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-purple-100">
              <Database className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Models</p>
              <p className="text-2xl font-semibold text-gray-900">{models.length}</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-blue-100">
              <Download className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Size</p>
              <p className="text-2xl font-semibold text-gray-900">
                {formatFileSize(models.reduce((total, model) => total + (model.size_mb || 0), 0))}
              </p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-green-100">
              <Eye className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Available</p>
              <p className="text-2xl font-semibold text-gray-900">{models.length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelCard key={model.name} model={model} />
        ))}
      </div>

      {models.length === 0 && (
        <div className="text-center py-12">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No models found</h3>
          <p className="text-gray-500">
            No trained models are currently available. Upload a model or wait for training to complete.
          </p>
        </div>
      )}
    </div>
  );
}

export default Models;
