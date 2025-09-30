import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { Brain, Play, Square, Clock, CheckCircle, XCircle } from 'lucide-react';

function Jobs() {
  const { jobs, models, loading, api } = useApp();
  const [filter, setFilter] = useState('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newJob, setNewJob] = useState({
    name: '',
    model_name: '',
    model_type: 'text_classification',
    dataset: 'synthetic',
    total_epochs: 10,
    config: {}
  });

  const filteredJobs = jobs.filter(job => {
    if (filter === 'all') return true;
    return job.status === filter;
  });

  const handleCreateJob = async (e) => {
    e.preventDefault();
    try {
      await api.createJob(newJob);
      setShowCreateModal(false);
      setNewJob({ name: '', model_name: '', model_type: 'text_classification', dataset: 'synthetic', total_epochs: 10, config: {} });
    } catch (error) {
      console.error('Failed to create job:', error);
    }
  };

  const handleStartJob = async (jobId) => {
    try {
      await api.startJob(jobId);
    } catch (error) {
      console.error('Failed to start job:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <Play className="h-4 w-4 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return <Square className="h-4 w-4 text-gray-500" />;
    }
  };

  const JobCard = ({ job }) => (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4">
          <div className="p-3 rounded-lg bg-blue-100">
            <Brain className="h-6 w-6 text-blue-600" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900">{job.name}</h3>
            <p className="text-sm text-gray-500">
              {job.model_name || job.model_type} • Dataset: {job.dataset || 'synthetic'}
            </p>
            <p className="text-xs text-gray-400">ID: {job.id}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {getStatusIcon(job.status)}
          <span className={`status-badge status-${job.status}`}>
            {job.status}
          </span>
        </div>
      </div>
      
      <div className="mt-4 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Progress</span>
          <span className="font-medium">{Math.round(job.progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${job.progress}%` }}
          ></div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Epochs:</span>
            <span className="ml-2 font-medium">{job.current_epoch} / {job.total_epochs}</span>
          </div>
          <div>
            <span className="text-gray-600">Created:</span>
            <span className="ml-2">{new Date(job.created_at).toLocaleDateString()}</span>
          </div>
        </div>
        
        {job.status === 'pending' && (
          <button
            onClick={() => handleStartJob(job.id)}
            className="w-full btn-primary text-sm"
          >
            <Play className="h-4 w-4 mr-2" />
            Start Training
          </button>
        )}
      </div>
    </div>
  );

  if (loading.jobs) {
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
          <h1 className="text-2xl font-bold text-gray-900">Training Jobs</h1>
          <p className="text-gray-600">Manage AI model training jobs</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="input w-auto"
          >
            <option value="all">All Jobs</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary"
          >
            <Brain className="h-4 w-4 mr-2" />
            Create Job
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-blue-100">
              <Brain className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Jobs</p>
              <p className="text-2xl font-semibold text-gray-900">{jobs.length}</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-yellow-100">
              <Clock className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Pending</p>
              <p className="text-2xl font-semibold text-gray-900">
                {jobs.filter(j => j.status === 'pending').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-green-100">
              <Play className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Running</p>
              <p className="text-2xl font-semibold text-gray-900">
                {jobs.filter(j => j.status === 'running').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-purple-100">
              <CheckCircle className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Completed</p>
              <p className="text-2xl font-semibold text-gray-900">
                {jobs.filter(j => j.status === 'completed').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Jobs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredJobs.map((job) => (
          <JobCard key={job.id} job={job} />
        ))}
      </div>

      {filteredJobs.length === 0 && (
        <div className="text-center py-12">
          <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs found</h3>
          <p className="text-gray-500">
            {filter === 'all' 
              ? 'No training jobs have been created yet.' 
              : `No jobs match the "${filter}" status.`
            }
          </p>
        </div>
      )}

      {/* Create Job Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Create Training Job</h3>
            <form onSubmit={handleCreateJob} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Job Name
                </label>
                <input
                  type="text"
                  value={newJob.name}
                  onChange={(e) => setNewJob({...newJob, name: e.target.value})}
                  className="input"
                  required
                />
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm font-medium text-gray-700">
                    Model
                  </label>
                  <button
                    type="button"
                    onClick={() => api.fetchModels()}
                    className="text-xs text-blue-600 hover:text-blue-800"
                    disabled={loading.models}
                  >
                    {loading.models ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
                <select
                  value={newJob.model_name}
                  onChange={(e) => {
                    if (e.target.value === 'create_new') {
                      // Handle creating new model - could open another modal or redirect
                      alert('To create a new model, go to the Models page and train a model first.');
                      return;
                    }
                    const selectedModel = models.find(m => m.name === e.target.value);
                    setNewJob({
                      ...newJob, 
                      model_name: e.target.value,
                      model_type: selectedModel?.model_type || 'text_classification'
                    });
                  }}
                  className="input"
                  required
                  disabled={loading.models}
                >
                  <option value="">Select a model...</option>
                  {models.map((model) => (
                    <option key={model.id} value={model.name}>
                      {model.name} ({model.model_type})
                    </option>
                  ))}
                  <option value="create_new" className="text-blue-600 font-medium">
                    + Create New Model
                  </option>
                </select>
                {models.length === 0 && !loading.models && (
                  <p className="text-xs text-gray-500 mt-1">
                    No models available. Train a model first or check the Models page.
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dataset
                </label>
                <select
                  value={newJob.dataset}
                  onChange={(e) => setNewJob({...newJob, dataset: e.target.value})}
                  className="input"
                  required
                >
                  <option value="synthetic">Synthetic Data (Default)</option>
                  <option value="ag_news">AG News (News Classification)</option>
                  <option value="imdb">IMDB (Movie Reviews)</option>
                  <option value="yelp">Yelp (Restaurant Reviews)</option>
                  <option value="amazon">Amazon (Product Reviews)</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Different devices will use different datasets for more realistic federated learning
                </p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Total Epochs
                </label>
                <input
                  type="number"
                  value={newJob.total_epochs}
                  onChange={(e) => setNewJob({...newJob, total_epochs: parseInt(e.target.value)})}
                  className="input"
                  min="1"
                  max="100"
                />
              </div>
              
              <div className="flex space-x-3 pt-4">
                <button type="submit" className="btn-primary flex-1">
                  Create Job
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn-secondary flex-1"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default Jobs;
