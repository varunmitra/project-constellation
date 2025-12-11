import SwiftUI
import Foundation

// Network-aware version of ConstellationApp that can connect to remote servers
@main
struct ConstellationApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .onAppear {
                    appState.loadConfiguration()
                }
        }
        .windowStyle(DefaultWindowStyle())
        .windowResizability(.contentSize)
    }
}

class AppState: ObservableObject {
    @Published var isConnected = false
    @Published var serverURL = "https://project-constellation.onrender.com"  // Default to Render deployment
    @Published var authToken = "constellation-token"
    @Published var deviceName = "Constellation-Desktop-App"
    @Published var connectionStatus = "Disconnected"
    @Published var lastError: String?
    @Published var deviceId: String?
    @Published var currentJob: TrainingJob?
    
    private let configFileName = "constellation_config.json"
    private let configDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    
    init() {
        // Set default device name with hostname
        if let hostname = Host.current().name {
            deviceName = "Constellation-\(hostname)"
        }
    }
    
    func loadConfiguration() {
        let configURL = configDirectory.appendingPathComponent(configFileName)
        
        if FileManager.default.fileExists(atPath: configURL.path) {
            do {
                let data = try Data(contentsOf: configURL)
                let config = try JSONDecoder().decode(NetworkConfig.self, from: data)
                
                serverURL = config.serverURL
                authToken = config.authToken
                deviceName = config.deviceName ?? deviceName
                
                print("📱 Loaded configuration: \(config)")
                
                // Auto-connect if enabled
                if config.autoConnect {
                    connectToServer()
                }
            } catch {
                print("❌ Failed to load configuration: \(error)")
                lastError = "Failed to load configuration: \(error.localizedDescription)"
            }
        } else {
            print("📱 No configuration file found, using defaults")
            // Try to auto-detect server on local network
            autoDetectServer()
        }
    }
    
    func saveConfiguration() {
        let config = NetworkConfig(
            serverURL: serverURL,
            authToken: authToken,
            deviceName: deviceName,
            autoConnect: true
        )
        
        do {
            let configURL = configDirectory.appendingPathComponent(configFileName)
            try FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            
            let data = try JSONEncoder().encode(config)
            try data.write(to: configURL)
            
            print("💾 Configuration saved: \(configURL)")
        } catch {
            print("❌ Failed to save configuration: \(error)")
            lastError = "Failed to save configuration: \(error.localizedDescription)"
        }
    }
    
    func autoDetectServer() {
        print("🔍 Auto-detecting Constellation server...")
        
        // Try default Render URL first
        let defaultRenderURL = "https://project-constellation.onrender.com"
        testServerConnection(url: defaultRenderURL) { [weak self] success in
            if success {
                DispatchQueue.main.async {
                    self?.serverURL = defaultRenderURL
                    self?.connectToServer()
                }
                return
            }
            
            // Fallback to local network detection
            let commonIPs = [
                "192.168.1.100", "192.168.1.101", "192.168.1.102",
                "192.168.0.100", "192.168.0.101", "192.168.0.102",
                "10.0.0.100", "10.0.0.101", "10.0.0.102"
            ]
            
            for ip in commonIPs {
                let testURL = "http://\(ip):8000"
                self?.testServerConnection(url: testURL) { success in
                    if success {
                        DispatchQueue.main.async {
                            self?.serverURL = testURL
                            self?.connectToServer()
                        }
                        return
                    }
                }
            }
        }
    }
    
    func testServerConnection(url: String, completion: @escaping (Bool) -> Void) {
        guard let serverURL = URL(string: "\(url)/health") else {
            completion(false)
            return
        }
        
        var request = URLRequest(url: serverURL)
        request.timeoutInterval = 10.0  // Increased timeout for Render free tier cold starts
        request.httpMethod = "GET"
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                completion(true)
            } else {
                completion(false)
            }
        }.resume()
    }
    
    func connectToServer() {
        print("🔌 Connecting to server: \(serverURL)")
        connectionStatus = "Connecting..."
        
        guard let url = URL(string: "\(serverURL)/devices/register") else {
            lastError = "Invalid server URL"
            connectionStatus = "Error"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        request.setValue("swift-app", forHTTPHeaderField: "x-constellation-client")  // Required header for device registration
        request.timeoutInterval = 60.0  // Increased timeout for Render free tier cold starts
        
        let deviceConfig = DeviceRegistration(
            name: deviceName,
            deviceType: "macbook",
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            cpuCores: ProcessInfo.processInfo.processorCount,
            memoryGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)),
            gpuAvailable: true,
            gpuMemoryGB: 8,
            status: "online"
        )
        
        do {
            request.httpBody = try JSONEncoder().encode(deviceConfig)
        } catch {
            lastError = "Failed to encode device config: \(error.localizedDescription)"
            connectionStatus = "Error"
            return
        }
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.lastError = "Connection error: \(error.localizedDescription)"
                    self?.connectionStatus = "Error"
                    self?.isConnected = false
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        // Parse device response to get device ID
                        if let data = data {
                            do {
                                let deviceResponse = try JSONDecoder().decode(DeviceResponse.self, from: data)
                                self?.deviceId = deviceResponse.id
                                print("✅ Device registered with ID: \(deviceResponse.id)")
                            } catch {
                                print("⚠️ Could not parse device response: \(error)")
                            }
                        }
                        
                        self?.isConnected = true
                        self?.connectionStatus = "Connected"
                        self?.lastError = nil
                        print("✅ Connected to Constellation server")
                        
                        // Start polling for jobs
                        self?.startJobPolling()
                    } else {
                        // Try to parse error message
                        if let data = data, let errorMsg = String(data: data, encoding: .utf8) {
                            self?.lastError = "Server error \(httpResponse.statusCode): \(errorMsg)"
                        } else {
                            self?.lastError = "Server error: \(httpResponse.statusCode)"
                        }
                        self?.connectionStatus = "Error"
                        self?.isConnected = false
                    }
                } else {
                    self?.lastError = "Invalid response from server"
                    self?.connectionStatus = "Error"
                    self?.isConnected = false
                }
            }
        }.resume()
    }
    
    func disconnect() {
        isConnected = false
        connectionStatus = "Disconnected"
        lastError = nil
        deviceId = nil
        currentJob = nil
        print("🔌 Disconnected from server")
    }
    
    // MARK: - Job Management
    
    func startJobPolling() {
        // Poll for jobs every 30 seconds
        Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] timer in
            guard let self = self, self.isConnected, let deviceId = self.deviceId else {
                timer.invalidate()
                return
            }
            self.fetchNextJob(deviceId: deviceId)
        }
    }
    
    func fetchNextJob(deviceId: String) {
        guard let url = URL(string: "\(serverURL)/devices/\(deviceId)/next-job") else {
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("⚠️ Error fetching next job: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200,
                   let data = data {
                    do {
                        let job = try JSONDecoder().decode(TrainingJob.self, from: data)
                        self?.currentJob = job
                        print("📋 Received training job: \(job.name)")
                    } catch {
                        // No job available or parsing error
                        if httpResponse.statusCode == 404 {
                            // No job available - this is normal
                            return
                        }
                        print("⚠️ Could not parse job response: \(error)")
                    }
                }
            }
        }.resume()
    }
    
    func sendHeartbeat(deviceId: String) {
        guard let url = URL(string: "\(serverURL)/devices/\(deviceId)/heartbeat") else {
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                print("⚠️ Heartbeat error: \(error.localizedDescription)")
            } else if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 200 {
                    print("💓 Heartbeat sent")
                }
            }
        }.resume()
    }
}

struct NetworkConfig: Codable {
    let serverURL: String
    let authToken: String
    let deviceName: String?
    let autoConnect: Bool
    
    enum CodingKeys: String, CodingKey {
        case serverURL = "server_url"
        case authToken = "auth_token"
        case deviceName = "device_name"
        case autoConnect = "auto_connect"
    }
}

struct DeviceRegistration: Codable {
    let name: String
    let deviceType: String
    let osVersion: String
    let cpuCores: Int
    let memoryGB: Int
    let gpuAvailable: Bool
    let gpuMemoryGB: Int
    let status: String
    
    enum CodingKeys: String, CodingKey {
        case name
        case deviceType = "device_type"
        case osVersion = "os_version"
        case cpuCores = "cpu_cores"
        case memoryGB = "memory_gb"
        case gpuAvailable = "gpu_available"
        case gpuMemoryGB = "gpu_memory_gb"
        case status
    }
}

struct DeviceResponse: Codable {
    let id: String
    let name: String
    let deviceType: String
    let osVersion: String
    let cpuCores: Int
    let memoryGB: Int
    let gpuAvailable: Bool
    let gpuMemoryGB: Int
    let isActive: Bool
    let lastSeen: String
    let createdAt: String
    
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case deviceType = "device_type"
        case osVersion = "os_version"
        case cpuCores = "cpu_cores"
        case memoryGB = "memory_gb"
        case gpuAvailable = "gpu_available"
        case gpuMemoryGB = "gpu_memory_gb"
        case isActive = "is_active"
        case lastSeen = "last_seen"
        case createdAt = "created_at"
    }
}

struct TrainingJob: Codable {
    let id: String
    let name: String
    let modelName: String?
    let modelType: String
    let dataset: String?
    let status: String
    let createdAt: String
    let startedAt: String?
    let completedAt: String?
    let totalEpochs: Int
    let currentEpoch: Int
    let progress: Double
    let config: String?
    
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case modelName = "model_name"
        case modelType = "model_type"
        case dataset
        case status
        case createdAt = "created_at"
        case startedAt = "started_at"
        case completedAt = "completed_at"
        case totalEpochs = "total_epochs"
        case currentEpoch = "current_epoch"
        case progress
        case config
    }
}

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var showingSettings = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Image(systemName: "network")
                    .font(.title)
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading) {
                    Text("Constellation")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    Text("Distributed AI Training")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: { showingSettings = true }) {
                    Image(systemName: "gear")
                        .font(.title2)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding()
            
            // Connection Status
            VStack(spacing: 10) {
                HStack {
                    Circle()
                        .fill(appState.isConnected ? .green : .red)
                        .frame(width: 12, height: 12)
                    
                    Text(appState.connectionStatus)
                        .font(.headline)
                    
                    Spacer()
                }
                
                if let error = appState.lastError {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.leading)
                }
                
                HStack {
                    Text("Server: \(appState.serverURL)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    if appState.isConnected {
                        Button("Disconnect") {
                            appState.disconnect()
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button("Connect") {
                            appState.connectToServer()
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            
            // Main Content
            if appState.isConnected {
                VStack(spacing: 15) {
                    Text("🎉 Connected to Constellation Server")
                        .font(.headline)
                        .foregroundColor(.green)
                    
                    Text("This device is now participating in distributed AI training.")
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                    
                    // Training Status
                    VStack(spacing: 10) {
                        Text("Training Status")
                            .font(.headline)
                        
                        if let job = appState.currentJob {
                            VStack(alignment: .leading, spacing: 5) {
                                Text("Current Job: \(job.name)")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                                
                                Text("Status: \(job.status)")
                                    .font(.caption)
                                
                                Text("Progress: \(Int(job.progress))%")
                                    .font(.caption)
                                
                                ProgressView(value: job.progress, total: 100)
                            }
                        } else {
                            Text("Ready to participate in federated learning")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Device Info
                    if let deviceId = appState.deviceId {
                        Text("Device ID: \(deviceId)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            } else {
                VStack(spacing: 15) {
                    Image(systemName: "wifi.slash")
                        .font(.system(size: 50))
                        .foregroundColor(.gray)
                    
                    Text("Not Connected")
                        .font(.headline)
                    
                    Text("Configure server settings to connect to Constellation")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            
            Spacer()
        }
        .padding()
        .frame(minWidth: 400, minHeight: 300)
        .sheet(isPresented: $showingSettings) {
            SettingsView()
                .environmentObject(appState)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var serverURL: String = ""
    @State private var authToken: String = ""
    @State private var deviceName: String = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("Server Configuration") {
                    TextField("Server URL", text: $serverURL)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    TextField("Auth Token", text: $authToken)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                Section("Device Configuration") {
                    TextField("Device Name", text: $deviceName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                Section("Actions") {
                    Button("Test Connection") {
                        testConnection()
                    }
                    
                    Button("Auto-Detect Server") {
                        appState.autoDetectServer()
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveSettings()
                    }
                }
            }
        }
        .onAppear {
            serverURL = appState.serverURL
            authToken = appState.authToken
            deviceName = appState.deviceName
        }
    }
    
    private func testConnection() {
        // Test connection to health endpoint
        guard let url = URL(string: "\(serverURL)/health") else {
            print("❌ Invalid URL: \(serverURL)")
            return
        }
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 10.0
        request.httpMethod = "GET"
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("❌ Connection test failed: \(error.localizedDescription)")
                } else if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        print("✅ Connection test successful!")
                    } else {
                        print("❌ Connection test failed with status: \(httpResponse.statusCode)")
                    }
                }
            }
        }.resume()
    }
    
    private func saveSettings() {
        appState.serverURL = serverURL
        appState.authToken = authToken
        appState.deviceName = deviceName
        appState.saveConfiguration()
        dismiss()
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
}
