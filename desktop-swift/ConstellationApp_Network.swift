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
    @Published var serverURL = "http://localhost:8000"
    @Published var authToken = "constellation-token"
    @Published var deviceName = "Constellation-Desktop-App"
    @Published var connectionStatus = "Disconnected"
    @Published var lastError: String?
    
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
        
        // Try common local network IPs
        let commonIPs = [
            "192.168.1.100", "192.168.1.101", "192.168.1.102",
            "192.168.0.100", "192.168.0.101", "192.168.0.102",
            "10.0.0.100", "10.0.0.101", "10.0.0.102"
        ]
        
        for ip in commonIPs {
            let testURL = "http://\(ip):8000"
            testServerConnection(url: testURL) { [weak self] success in
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
    
    func testServerConnection(url: String, completion: @escaping (Bool) -> Void) {
        guard let serverURL = URL(string: "\(url)/health") else {
            completion(false)
            return
        }
        
        var request = URLRequest(url: serverURL)
        request.timeoutInterval = 2.0
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
                        self?.isConnected = true
                        self?.connectionStatus = "Connected"
                        self?.lastError = nil
                        print("✅ Connected to Constellation server")
                    } else {
                        self?.lastError = "Server error: \(httpResponse.statusCode)"
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
        print("🔌 Disconnected from server")
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
                        
                        Text("Ready to participate in federated learning")
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
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
        // Test connection logic
        print("Testing connection to: \(serverURL)")
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
