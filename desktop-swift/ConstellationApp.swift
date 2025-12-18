#!/usr/bin/env swift

/*
 Project Constellation - Enhanced macOS Application
 A sophisticated macOS app for distributed AI training with improved UI and animations
 */

import Foundation
import AppKit
import Metal

// MARK: - Python Training Executor
class PythonTrainingExecutor {
    private var process: Process?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    
    func executeTraining(job: TrainingJob, assignmentId: String, deviceId: String, serverURL: String) async throws -> Bool {
        // Find Python executable
        let pythonPath = findPythonExecutable()
        guard let pythonPath = pythonPath else {
            throw NSError(domain: "TrainingError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Python not found"])
        }
        
        print("🐍 Using Python at: \(pythonPath)")
        
        // Get the project root directory (assuming ConstellationApp.swift is in desktop-swift/)
        let projectRoot = getProjectRoot()
        let trainingScript = "\(projectRoot)/training/run_job.py"
        
        guard FileManager.default.fileExists(atPath: trainingScript) else {
            throw NSError(domain: "TrainingError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Training script not found: \(trainingScript)"])
        }
        
        // Convert AnyCodable config to [String: Any] for JSON serialization
        var configDict: [String: Any] = [:]
        for (key, anyCodable) in job.config {
            configDict[key] = anyCodable.value
        }
        
        // Create job configuration JSON
        let jobConfig: [String: Any] = [
            "id": job.id,
            "name": job.name,
            "model_type": job.modelType,
            "dataset": configDict["dataset"] as? String ?? "synthetic",
            "total_epochs": job.totalEpochs,
            "config": configDict,
            "assignment_id": assignmentId,
            "device_id": deviceId,
            "server_url": serverURL
        ]
        
        // Write config to temporary file
        let tempConfigPath = NSTemporaryDirectory() + "constellation_job_\(job.id).json"
        
        do {
            guard JSONSerialization.isValidJSONObject(jobConfig) else {
                throw NSError(domain: "TrainingError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON object"])
            }
            let configData = try JSONSerialization.data(withJSONObject: jobConfig)
            guard let configString = String(data: configData, encoding: .utf8) else {
                throw NSError(domain: "TrainingError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert JSON data to string"])
            }
            
            try configString.write(toFile: tempConfigPath, atomically: true, encoding: .utf8)
            print("✅ Config file written to: \(tempConfigPath)")
        } catch {
            print("❌ Failed to serialize job config: \(error)")
            throw NSError(domain: "TrainingError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to serialize job config: \(error.localizedDescription)"])
        }
        
        // Create process
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [trainingScript, tempConfigPath]
        
        print("🐍 Python path: \(pythonPath)")
        print("📝 Training script: \(trainingScript)")
        print("📄 Config file: \(tempConfigPath)")
        print("🔧 Arguments: \(process.arguments?.joined(separator: " ") ?? "none")")
        
        // Set environment variables
        var environment = ProcessInfo.processInfo.environment
        environment["SERVER_URL"] = serverURL
        environment["DEVICE_ID"] = deviceId
        environment["ASSIGNMENT_ID"] = assignmentId
        process.environment = environment
        
        // Set up pipes for output
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        self.process = process
        self.outputPipe = outputPipe
        self.errorPipe = errorPipe
        
        // Start process
        try process.run()
        
        print("🚀 Started Python training process (PID: \(process.processIdentifier))")
        
        // Monitor output for progress updates
        let outputHandle = outputPipe.fileHandleForReading
        let errorHandle = errorPipe.fileHandleForReading
        
        var outputBuffer = Data()
        var errorBuffer = Data()
        
        outputHandle.readabilityHandler = { handle in
            let data = handle.availableData
            if data.count > 0 {
                outputBuffer.append(data)
                if let output = String(data: data, encoding: .utf8) {
                    print("📊 Training output: \(output.trimmingCharacters(in: .whitespacesAndNewlines))")
                }
            } else {
                // End of file
                if let fullOutput = String(data: outputBuffer, encoding: .utf8), !fullOutput.isEmpty {
                    print("📊 Full training output:\n\(fullOutput)")
                }
            }
        }
        
        errorHandle.readabilityHandler = { handle in
            let data = handle.availableData
            if data.count > 0 {
                errorBuffer.append(data)
                if let logOutput = String(data: data, encoding: .utf8) {
                    let trimmedOutput = logOutput.trimmingCharacters(in: .whitespacesAndNewlines)
                    // Parse log level and display appropriately
                    // Python logging format: "INFO:module:message" or "WARNING:module:message" or "ERROR:module:message"
                    if trimmedOutput.contains("ERROR:") || trimmedOutput.contains("CRITICAL:") {
                        print("❌ Training error: \(trimmedOutput)")
                    } else if trimmedOutput.contains("WARNING:") || trimmedOutput.contains("WARN:") {
                        print("⚠️ Training warning: \(trimmedOutput)")
                    } else if trimmedOutput.contains("INFO:") {
                        // INFO logs are normal - use neutral emoji
                        print("ℹ️ Training: \(trimmedOutput)")
                    } else if trimmedOutput.contains("DEBUG:") {
                        // DEBUG logs - can be verbose, use subtle emoji
                        print("🔍 Training debug: \(trimmedOutput)")
                    } else {
                        // Unknown format - treat as info level
                        print("ℹ️ Training: \(trimmedOutput)")
                    }
                }
            } else {
                // End of file
                if let fullError = String(data: errorBuffer, encoding: .utf8), !fullError.isEmpty {
                    // Check if there are actual errors vs just logs
                    if fullError.contains("ERROR:") || fullError.contains("CRITICAL:") || fullError.contains("Traceback") {
                        print("❌ Full training error output:\n\(fullError)")
                    } else {
                        print("ℹ️ Full training log output:\n\(fullError)")
                    }
                }
            }
        }
        
        // Wait for process to complete
        process.waitUntilExit()
        
        // Clean up
        outputHandle.readabilityHandler = nil
        errorHandle.readabilityHandler = nil
        
        // Print final output if any
        if let finalOutput = String(data: outputBuffer, encoding: .utf8), !finalOutput.isEmpty {
            print("📊 Complete training output:\n\(finalOutput)")
        }
        if let finalError = String(data: errorBuffer, encoding: .utf8), !finalError.isEmpty {
            // Check if there are actual errors vs just logs
            if finalError.contains("ERROR:") || finalError.contains("CRITICAL:") || finalError.contains("Traceback") {
                print("❌ Complete training errors:\n\(finalError)")
            } else {
                print("ℹ️ Complete training logs:\n\(finalError)")
            }
        }
        try? FileManager.default.removeItem(atPath: tempConfigPath)
        
        let exitCode = process.terminationStatus
        if exitCode == 0 {
            print("✅ Training completed successfully")
            return true
        } else {
            // Read error output
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            if let errorString = String(data: errorData, encoding: .utf8) {
                print("❌ Training failed: \(errorString)")
            }
            return false
        }
    }
    
    func cancel() {
        process?.terminate()
        process?.waitUntilExit()
    }
    
    private func findPythonExecutable() -> String? {
        let possiblePaths = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python",
            "/usr/local/bin/python"
        ]
        
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                // Verify it's actually Python
                let process = Process()
                process.executableURL = URL(fileURLWithPath: path)
                process.arguments = ["--version"]
                
                do {
                    try process.run()
                    process.waitUntilExit()
                    if process.terminationStatus == 0 {
                        return path
                    }
                } catch {
                    continue
                }
            }
        }
        
        // Try to find Python using which command
        let whichProcess = Process()
        whichProcess.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        whichProcess.arguments = ["python3"]
        
        let pipe = Pipe()
        whichProcess.standardOutput = pipe
        
        do {
            try whichProcess.run()
            whichProcess.waitUntilExit()
            if whichProcess.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !path.isEmpty {
                    return path
                }
            }
        } catch {
            // Ignore
        }
        
        return nil
    }
    
    private func getProjectRoot() -> String {
        let fileManager = FileManager.default
        
        // Strategy 0: Check if training scripts are bundled in app (for distribution)
        if let appBundle = Bundle.main.resourcePath {
            let bundledTrainingScript = "\(appBundle)/training/run_job.py"
            if fileManager.fileExists(atPath: bundledTrainingScript) {
                print("✅ Found project root (bundled in app): \(appBundle)")
                return appBundle
            }
        }
        
        // Strategy 1: Check if we're running from desktop-swift directory
        let currentPath = fileManager.currentDirectoryPath
        
        // If we're in desktop-swift, go up one level
        if currentPath.hasSuffix("desktop-swift") || currentPath.hasSuffix("desktop-swift/build") {
            let parentPath = (currentPath as NSString).deletingLastPathComponent
            let trainingScript = "\(parentPath)/training/run_job.py"
            if fileManager.fileExists(atPath: trainingScript) {
                print("✅ Found project root (from desktop-swift): \(parentPath)")
                return parentPath
            }
        }
        
        // Strategy 2: Check current directory
        let trainingScript1 = "\(currentPath)/training/run_job.py"
        if fileManager.fileExists(atPath: trainingScript1) {
            print("✅ Found project root (current dir): \(currentPath)")
            return currentPath
        }
        
        // Strategy 3: Go up from current directory
        var checkPath = currentPath
        for i in 0..<5 {
            let trainingScript = "\(checkPath)/training/run_job.py"
            if fileManager.fileExists(atPath: trainingScript) {
                print("✅ Found project root (going up \(i) levels): \(checkPath)")
                return checkPath
            }
            let parent = (checkPath as NSString).deletingLastPathComponent
            if parent == checkPath || parent == "/" {
                break
            }
            checkPath = parent
        }
        
        // Strategy 4: Try common locations
        let homeDir = NSHomeDirectory()
        let commonPaths = [
            "\(homeDir)/Documents/GitHub/Constellation",
            "\(homeDir)/Constellation",
            "/Users/\(NSUserName())/Documents/GitHub/Constellation"
        ]
        
        for path in commonPaths {
            let trainingScript = "\(path)/training/run_job.py"
            if fileManager.fileExists(atPath: trainingScript) {
                print("✅ Found project root (common path): \(path)")
                return path
            }
        }
        
        // Fallback: return current path and hope for the best
        print("⚠️ Could not find project root, using: \(currentPath)")
        return currentPath
    }
}

// MARK: - Enums
enum ConnectionStatus {
    case disconnected
    case connecting
    case connected
    case training
    case stopped
}

enum AppStatus {
    case idle
    case running
    case stopped
}

// MARK: - Data Models
struct DeviceRegistration: Codable {
    let name: String
    let deviceType: String
    let osVersion: String
    let cpuCores: Int
    let memoryGB: Int
    let gpuAvailable: Bool
    let gpuMemoryGB: Int
    
    enum CodingKeys: String, CodingKey {
        case name, deviceType = "device_type", osVersion = "os_version"
        case cpuCores = "cpu_cores", memoryGB = "memory_gb"
        case gpuAvailable = "gpu_available", gpuMemoryGB = "gpu_memory_gb"
    }
}

struct DeviceInfo: Codable {
    let id: String
    let name: String
    let deviceType: String
    let osVersion: String
    let cpuCores: Int
    let memoryGB: Int
    let gpuAvailable: Bool
    let gpuMemoryGB: Int
    
    enum CodingKeys: String, CodingKey {
        case id, name, deviceType = "device_type", osVersion = "os_version"
        case cpuCores = "cpu_cores", memoryGB = "memory_gb"
        case gpuAvailable = "gpu_available", gpuMemoryGB = "gpu_memory_gb"
    }
}

struct TrainingJob: Codable {
    let id: String
    let name: String
    let modelType: String
    let status: String
    let totalEpochs: Int
    let currentEpoch: Int
    let progress: Double
    let config: [String: AnyCodable]
    
    enum CodingKeys: String, CodingKey {
        case id, name, modelType = "model_type", status, totalEpochs = "total_epochs"
        case currentEpoch = "current_epoch", progress, config
    }
}

struct AnyCodable: Codable {
    let value: Any
    
    init(_ value: Any) {
        self.value = value
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let intValue = try? container.decode(Int.self) {
            value = intValue
        } else if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
        } else if let stringValue = try? container.decode(String.self) {
            value = stringValue
        } else if let boolValue = try? container.decode(Bool.self) {
            value = boolValue
        } else {
            throw DecodingError.typeMismatch(AnyCodable.self, DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported type"))
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        if let intValue = value as? Int {
            try container.encode(intValue)
        } else if let doubleValue = value as? Double {
            try container.encode(doubleValue)
        } else if let stringValue = value as? String {
            try container.encode(stringValue)
        } else if let boolValue = value as? Bool {
            try container.encode(boolValue)
        } else {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: encoder.codingPath, debugDescription: "Unsupported type"))
        }
    }
}

// MARK: - Server URL Manager
class ServerURLManager {
    private let userDefaults = UserDefaults.standard
    private let serverURLKey = "ConstellationServerURL"
    
    var serverURL: String {
        get {
            // Always default to Render server
            let defaultURL = "https://project-constellation.onrender.com"
            if let savedURL = userDefaults.string(forKey: serverURLKey) {
                // Only use saved URL if it's not localhost
                if savedURL.contains("localhost") || savedURL.contains("127.0.0.1") {
                    print("⚠️  Ignoring localhost setting, using Render default")
                    return defaultURL
                }
                return savedURL
            }
            return defaultURL
        }
        set {
            // Don't save localhost URLs
            if newValue.contains("localhost") || newValue.contains("127.0.0.1") {
                print("⚠️  Cannot set server URL to localhost, using Render default")
                userDefaults.removeObject(forKey: serverURLKey)
            } else {
                userDefaults.set(newValue, forKey: serverURLKey)
            }
        }
    }
    
    func isValidURL(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        return url.scheme == "http" || url.scheme == "https"
    }
}

// MARK: - Network Manager
class NetworkManager: ObservableObject {
    @Published var connectionStatus: ConnectionStatus = .disconnected
    let serverURLManager = ServerURLManager()
    private let session = URLSession.shared
    
    // Cache GPU info to avoid repeated detection
    private var cachedGPUInfo: (available: Bool, cores: Int, memoryGB: Int)?
    private var gpuInfoDetected = false
    
    var baseURL: String {
        return serverURLManager.serverURL
    }
    
    // GPU Detection (cached, only logs on first detection)
    func detectGPU() -> (available: Bool, cores: Int, memoryGB: Int) {
        // Return cached value if available
        if let cached = cachedGPUInfo {
            return cached
        }
        
        // Check for Metal GPU support (macOS)
        let result: (Bool, Int, Int)
        if MTLCreateSystemDefaultDevice() != nil {
            // Estimate GPU cores and memory based on system
            let totalMemory = ProcessInfo.processInfo.physicalMemory
            let memoryGB = Int(totalMemory / (1024 * 1024 * 1024))
            
            // Estimate GPU cores (rough approximation for Apple Silicon)
            let cpuCores = ProcessInfo.processInfo.processorCount
            let estimatedGPUCores = max(8, cpuCores * 2) // Conservative estimate
            
            // Estimate GPU memory (typically 25-50% of system memory)
            let estimatedGPUMemory = max(4, memoryGB / 4)
            
            result = (true, estimatedGPUCores, estimatedGPUMemory)
            
            // Only log on first detection
            if !gpuInfoDetected {
                print("🎮 GPU detected: \(estimatedGPUCores) cores, \(estimatedGPUMemory)GB memory")
                gpuInfoDetected = true
            }
        } else {
            result = (false, 0, 0)
            // Only log on first detection
            if !gpuInfoDetected {
                print("❌ No GPU detected, using CPU only")
                gpuInfoDetected = true
            }
        }
        
        // Cache the result
        cachedGPUInfo = result
        return result
    }
    
    func updateServerURL(_ newURL: String) {
        serverURLManager.serverURL = newURL
        connectionStatus = .disconnected
    }
    
    func testConnection() async -> Bool {
        connectionStatus = .connecting
        
        do {
            let url = URL(string: "\(baseURL)/health")!
            let (_, response) = try await session.data(from: url)
            
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                connectionStatus = .connected
                return true
            } else {
                connectionStatus = .disconnected
                return false
            }
        } catch {
            connectionStatus = .disconnected
            return false
        }
    }
    
    func registerDevice() async throws -> DeviceInfo {
        // Detect GPU availability and cores
        let (gpuAvailable, _, gpuMemoryGB) = detectGPU()
        
        let deviceRegistration = DeviceRegistration(
            name: Host.current().name ?? "Unknown",
            deviceType: "macbook",
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            cpuCores: ProcessInfo.processInfo.processorCount,
            memoryGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)),
            gpuAvailable: gpuAvailable,
            gpuMemoryGB: gpuMemoryGB
        )
        
        var request = URLRequest(url: URL(string: "\(baseURL)/devices/register")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer constellation-token", forHTTPHeaderField: "Authorization")
        request.setValue("Constellation-Swift/1.0", forHTTPHeaderField: "User-Agent")
        request.setValue("swift-app", forHTTPHeaderField: "X-Constellation-Client")
        
        let jsonData = try JSONEncoder().encode(deviceRegistration)
        request.httpBody = jsonData
        
        // Debug logging
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            print("📤 Sending registration data: \(jsonString)")
        }
        print("📤 Registration URL: \(baseURL)/devices/register")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            print("❌ Registration failed with status: \(statusCode)")
            if let responseData = String(data: data, encoding: .utf8) {
                print("❌ Response: \(responseData)")
            }
            throw NSError(domain: "NetworkError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Registration failed"])
        }
        
        return try JSONDecoder().decode(DeviceInfo.self, from: data)
    }
    
    func sendHeartbeat(deviceId: String) async throws {
        var request = URLRequest(url: URL(string: "\(baseURL)/devices/\(deviceId)/heartbeat")!)
        request.httpMethod = "POST"
        request.setValue("Bearer constellation-token", forHTTPHeaderField: "Authorization")
        
        let (_, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "NetworkError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Heartbeat failed"])
        }
    }
    
    func getNextJob(deviceId: String) async throws -> (job: TrainingJob?, assignmentId: String?) {
        var request = URLRequest(url: URL(string: "\(baseURL)/devices/\(deviceId)/next-job")!)
        request.setValue("Bearer constellation-token", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            print("⚠️ getNextJob: HTTP \(statusCode)")
            return (nil, nil)
        }
        
        // Parse JSON response
        guard let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("⚠️ getNextJob: Failed to parse JSON response")
            return (nil, nil)
        }
        
        // Check if job is null/None
        if jsonObject["job"] == nil || jsonObject["job"] is NSNull {
            let message = jsonObject["message"] as? String ?? "No job available"
            print("ℹ️ getNextJob: \(message)")
            return (nil, nil)
        }
        
        // Get job dictionary
        guard let jobDict = jsonObject["job"] as? [String: Any] else {
            print("⚠️ getNextJob: Job is not a dictionary")
            return (nil, nil)
        }
        
        // Get assignment ID from response
        let assignmentId = jsonObject["assignment_id"] as? String
        
        // Decode job using JSONSerialization
        let jobJson = try JSONSerialization.data(withJSONObject: jobDict)
        let job = try JSONDecoder().decode(TrainingJob.self, from: jobJson)
        
        print("✅ getNextJob: Received job '\(job.name)' (ID: \(job.id), assignment: \(assignmentId ?? "none"))")
        
        return (job, assignmentId)
    }
    
    func updateProgress(deviceId: String, assignmentId: String, progress: Double, epoch: Int) async throws {
        var request = URLRequest(url: URL(string: "\(baseURL)/devices/\(deviceId)/training/\(assignmentId)/progress")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer constellation-token", forHTTPHeaderField: "Authorization")
        
        let body: [String: Any] = ["progress": progress, "current_epoch": epoch]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (_, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "NetworkError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Progress update failed"])
        }
    }
    
    func completeTraining(deviceId: String, assignmentId: String) async throws {
        var request = URLRequest(url: URL(string: "\(baseURL)/devices/\(deviceId)/training/\(assignmentId)/complete")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer constellation-token", forHTTPHeaderField: "Authorization")
        
        let body: [String: Any] = ["checkpoint_path": "swift-app-checkpoint-\(assignmentId).pth"]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (_, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "NetworkError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Training completion failed"])
        }
    }
}

// MARK: - Training Manager
class TrainingManager: ObservableObject {
    @Published var isTraining = false
    @Published var currentJob: TrainingJob?
    @Published var progress: Double = 0.0
    @Published var status = "Idle"
    @Published var deviceInfo: DeviceInfo?
    @Published var appStatus: AppStatus = .idle
    @Published var currentEpoch: Int = 0
    @Published var totalEpochs: Int = 0
    @Published var trainingAccuracy: Double = 0.0
    @Published var trainingLoss: Double = 0.0
    @Published var trainingStartTime: Date?
    @Published var estimatedTimeRemaining: TimeInterval = 0
    
    private let networkManager: NetworkManager
    private var heartbeatTimer: Timer?
    private var trainingTimer: Timer?
    private var connectionTimer: Timer?
    
    init(networkManager: NetworkManager) {
        self.networkManager = networkManager
    }
    
    func start() async {
        do {
            print("🔧 Testing connection to server...")
            let connected = await networkManager.testConnection()
            
            if !connected {
                print("❌ Cannot connect to server at \(networkManager.baseURL)")
                status = "Disconnected"
                return
            }
            
            print("✅ Connected to server")
            print("🔧 Registering device...")
            deviceInfo = try await networkManager.registerDevice()
            print("✅ Device registered: \(deviceInfo?.name ?? "Unknown") (ID: \(deviceInfo?.id ?? "unknown"))")
            
            appStatus = .running
            status = "Connected"
            
            print("🔄 Starting background tasks...")
            startHeartbeat()
            startTrainingLoop()
            startConnectionMonitoring()
            print("✅ All background tasks started")
            print("📋 Training loop will check for jobs every 10 seconds")
            
        } catch {
            print("❌ Failed to register device: \(error)")
            status = "Registration Failed"
        }
    }
    
    func stop() {
        appStatus = .stopped
        isTraining = false
        status = "Stopped"
        
        heartbeatTimer?.invalidate()
        trainingTimer?.invalidate()
        connectionTimer?.invalidate()
        
        print("🛑 Training manager stopped")
    }
    
    private func startHeartbeat() {
        print("💓 Starting heartbeat timer (every 30 seconds)...")
        // Invalidate any existing timer first
        heartbeatTimer?.invalidate()
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            Task {
                await self.sendHeartbeat()
            }
        }
        // Ensure timer fires even when app is in background
        RunLoop.current.add(heartbeatTimer!, forMode: .common)
        // Send initial heartbeat immediately
        Task {
            await sendHeartbeat()
        }
    }
    
    private func sendHeartbeat() async {
        guard let deviceId = deviceInfo?.id else { return }
        
        do {
            try await networkManager.sendHeartbeat(deviceId: deviceId)
            print("💓 Heartbeat sent")
        } catch {
            print("❌ Heartbeat failed: \(error)")
            networkManager.connectionStatus = .disconnected
        }
    }
    
    private func startTrainingLoop() {
        print("🔄 Starting training loop (checking every 10 seconds)...")
        // Invalidate any existing timer first
        trainingTimer?.invalidate()
        trainingTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            Task {
                await self.checkForJobs()
            }
        }
        // Ensure timer fires even when app is in background
        RunLoop.current.add(trainingTimer!, forMode: .common)
        // Do an immediate check
        Task {
            print("🔍 Performing initial job check...")
            await checkForJobs()
        }
    }
    
    private func startConnectionMonitoring() {
        connectionTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task {
                await self?.monitorConnection()
            }
        }
    }
    
    private func monitorConnection() async {
        let connected = await networkManager.testConnection()
        if !connected && appStatus == .running {
            status = "Disconnected"
            networkManager.connectionStatus = .disconnected
        }
    }
    
    func checkForJobs() async {
        guard let deviceId = deviceInfo?.id else {
            print("⚠️ checkForJobs: No device ID")
            return
        }
        
        guard !isTraining else {
            print("⚠️ checkForJobs: Already training, skipping")
            return
        }
        
        guard appStatus == .running else {
            print("⚠️ checkForJobs: App status is \(appStatus), not running")
            return
        }
        
        print("🔍 Checking for jobs... (device: \(deviceId), status: \(status))")
        
        do {
            let (job, assignmentId) = try await networkManager.getNextJob(deviceId: deviceId)
            if let job = job {
                print("📝 New training job found: \(job.name) (ID: \(job.id))")
                if let assignmentId = assignmentId {
                    print("✅ Got assignment ID: \(assignmentId)")
                    await startTraining(job: job, assignmentId: assignmentId)
                } else {
                    print("⚠️ No assignment ID from server, generating one")
                    await startTraining(job: job, assignmentId: "assignment-\(job.id)-\(deviceId)")
                }
            } else {
                print("ℹ️ No jobs available at this time (will check again in 10 seconds)")
            }
        } catch {
            print("❌ Failed to get next job: \(error)")
            // Don't stop the timer on error - keep trying
        }
    }
    
    private func startTraining(job: TrainingJob, assignmentId: String) async {
        isTraining = true
        currentJob = job
        status = "Training"
        progress = 0.0
        currentEpoch = 0
        totalEpochs = job.totalEpochs
        trainingAccuracy = 0.0
        trainingLoss = 0.0
        trainingStartTime = Date()
        estimatedTimeRemaining = 0
        networkManager.connectionStatus = .training
        
        print("🚀 Starting training: \(job.name)")
        print("📋 Assignment ID: \(assignmentId)")
        
        // Determine compute resources based on GPU availability
        let (gpuAvailable, gpuCores, gpuMemoryGB) = networkManager.detectGPU()
        let cpuCores = ProcessInfo.processInfo.processorCount
        
        if gpuAvailable {
            print("🎮 Using GPU: \(gpuCores) cores (50% = \(gpuCores / 2))")
            print("💾 GPU Memory: \(gpuMemoryGB)GB")
        } else {
            print("🖥️ Using CPU: \(cpuCores) cores (75% = \(Int(Double(cpuCores) * 0.75)))")
        }
        
        // Get device ID
        guard let deviceId = deviceInfo?.id else {
            print("❌ No device ID available")
            await stopTraining()
            return
        }
        
        // Execute real training based on job type
        await executeTraining(job: job, assignmentId: assignmentId, deviceId: deviceId)
    }
    
    private func executeTraining(job: TrainingJob, assignmentId: String, deviceId: String) async {
        do {
            // Get compute resources
            let (gpuAvailable, gpuCores, _) = networkManager.detectGPU()
            let cpuCores = ProcessInfo.processInfo.processorCount
            
            // Calculate resource allocation
            let computeCores = gpuAvailable ? gpuCores / 2 : Int(Double(cpuCores) * 0.75)
            let computeType = gpuAvailable ? "GPU" : "CPU"
            
            print("🔧 Allocated \(computeCores) \(computeType) cores for training")
            
            // Execute training based on job type
            switch job.modelType {
            case "text_classification":
                await executeTextClassificationTraining(job: job, assignmentId: assignmentId, deviceId: deviceId, cores: computeCores, useGPU: gpuAvailable)
            case "image_classification":
                await executeImageClassificationTraining(job: job, assignmentId: assignmentId, deviceId: deviceId, cores: computeCores, useGPU: gpuAvailable)
            default:
                await executeGenericTraining(job: job, assignmentId: assignmentId, deviceId: deviceId, cores: computeCores, useGPU: gpuAvailable)
            }
        } catch {
            print("❌ Training failed: \(error)")
            await stopTraining()
        }
    }
    
    private func executeTextClassificationTraining(job: TrainingJob, assignmentId: String, deviceId: String, cores: Int, useGPU: Bool) async {
        print("📝 Executing text classification training for: \(job.name)")
        print("🔧 Using \(cores) \(useGPU ? "GPU" : "CPU") cores")
        
        // Execute real Python training
        let executor = PythonTrainingExecutor()
        
        do {
            let success = try await executor.executeTraining(
                job: job,
                assignmentId: assignmentId,
                deviceId: deviceId,
                serverURL: networkManager.baseURL
            )
            
            if success && isTraining {
                await completeTraining(job: job, assignmentId: assignmentId, deviceId: deviceId)
            } else if !success {
                print("❌ Training failed")
                await stopTraining()
            }
        } catch {
            print("❌ Training error: \(error)")
            await stopTraining()
        }
    }
    
    private func executeImageClassificationTraining(job: TrainingJob, assignmentId: String, deviceId: String, cores: Int, useGPU: Bool) async {
        print("🖼️ Executing image classification training for: \(job.name)")
        print("🔧 Using \(cores) \(useGPU ? "GPU" : "CPU") cores")
        
        // Execute real Python training
        let executor = PythonTrainingExecutor()
        
        do {
            let success = try await executor.executeTraining(
                job: job,
                assignmentId: assignmentId,
                deviceId: deviceId,
                serverURL: networkManager.baseURL
            )
            
            if success && isTraining {
                await completeTraining(job: job, assignmentId: assignmentId, deviceId: deviceId)
            } else if !success {
                print("❌ Training failed")
                await stopTraining()
            }
        } catch {
            print("❌ Training error: \(error)")
            await stopTraining()
        }
    }
    
    private func executeGenericTraining(job: TrainingJob, assignmentId: String, deviceId: String, cores: Int, useGPU: Bool) async {
        print("🔧 Executing generic training for: \(job.name)")
        print("🔧 Using \(cores) \(useGPU ? "GPU" : "CPU") cores")
        
        // Execute real Python training
        let executor = PythonTrainingExecutor()
        
        do {
            let success = try await executor.executeTraining(
                job: job,
                assignmentId: assignmentId,
                deviceId: deviceId,
                serverURL: networkManager.baseURL
            )
            
            if success && isTraining {
                await completeTraining(job: job, assignmentId: assignmentId, deviceId: deviceId)
            } else if !success {
                print("❌ Training failed")
                await stopTraining()
            }
        } catch {
            print("❌ Training error: \(error)")
            await stopTraining()
        }
    }
    
    private func calculateTrainingTime(epoch: Int, totalEpochs: Int, modelType: String) -> Double {
        // Get current compute resources
        let (gpuAvailable, gpuCores, _) = networkManager.detectGPU()
        let cpuCores = ProcessInfo.processInfo.processorCount
        let computeCores = gpuAvailable ? gpuCores / 2 : Int(Double(cpuCores) * 0.75)
        
        // Simulate realistic training times based on compute resources
        let baseTime: Double
        switch modelType {
        case "text_classification":
            baseTime = gpuAvailable ? 1.0 : 2.5 // GPU is faster
        case "image_classification":
            baseTime = gpuAvailable ? 2.0 : 5.0 // Image training benefits more from GPU
        default:
            baseTime = gpuAvailable ? 1.5 : 3.0
        }
        
        // Adjust based on available cores (more cores = faster)
        let coreMultiplier = 1.0 - (Double(computeCores) / 100.0)
        let adjustedTime = baseTime * max(0.3, coreMultiplier) // Minimum 30% of base time
        
        // Add some variation based on epoch (later epochs might be faster due to optimization)
        let variation = 1.0 + (Double(epoch) / Double(totalEpochs)) * 0.3
        return adjustedTime * variation
    }
    
    private func simulateAccuracy(epoch: Int, totalEpochs: Int) -> Double {
        // Simulate improving accuracy over time
        let baseAccuracy = 60.0
        let improvement = (Double(epoch) / Double(totalEpochs)) * 35.0
        let noise = Double.random(in: -2.0...2.0)
        return max(0, min(100, baseAccuracy + improvement + noise))
    }
    
    private func simulateLoss(epoch: Int, totalEpochs: Int) -> Double {
        // Simulate decreasing loss over time
        let baseLoss = 2.0
        let improvement = (Double(epoch) / Double(totalEpochs)) * 1.5
        let noise = Double.random(in: -0.1...0.1)
        return max(0.01, baseLoss - improvement + noise)
    }
    
    private func completeTraining(job: TrainingJob, assignmentId: String, deviceId: String) async {
        print("✅ Training completed: \(job.name)")
        
        // Mark training as completed on server
        try? await networkManager.completeTraining(
            deviceId: deviceId,
            assignmentId: assignmentId
        )
        
        isTraining = false
        status = "Connected - Ready for next job"
        currentJob = nil
        progress = 0.0
        currentEpoch = 0
        totalEpochs = 0
        trainingAccuracy = 0.0
        trainingLoss = 0.0
        trainingStartTime = nil
        estimatedTimeRemaining = 0
        networkManager.connectionStatus = .connected
        
        // Ensure training loop continues checking for new jobs
        print("🔄 Training loop will continue checking for new jobs...")
        // Trigger an immediate check for new jobs after a short delay
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 second delay
            if !isTraining && appStatus == .running {
                print("🔍 Checking for new jobs after training completion...")
                await checkForJobs()
            }
        }
    }
    
    func stopTraining() {
        isTraining = false
        status = "Stopped"
        networkManager.connectionStatus = .stopped
        print("🛑 Training stopped")
    }
}

// MARK: - Server URL Dialog
class ServerURLDialog: NSWindowController {
    private let networkManager = NetworkManager()
    private var completionHandler: ((String) -> Void)?
    private var urlTextField: NSTextField!
    private var testButton: NSButton!
    private var statusLabel: NSTextField!
    private var saveButton: NSButton!
    private var cancelButton: NSButton!
    
    convenience init(currentURL: String, completion: @escaping (String) -> Void) {
        self.init(window: nil)
        self.completionHandler = completion
        setupWindow()
        urlTextField.stringValue = currentURL
    }
    
    private func setupWindow() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 200),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Configure Server URL"
        window.center()
        window.isReleasedWhenClosed = false
        
        self.window = window
        
        let contentView = window.contentView!
        
        // URL Label
        let urlLabel = NSTextField(labelWithString: "Server URL:")
        urlLabel.frame = NSRect(x: 20, y: 150, width: 80, height: 20)
        contentView.addSubview(urlLabel)
        
        // URL Text Field
        urlTextField = NSTextField()
        urlTextField.frame = NSRect(x: 110, y: 150, width: 270, height: 24)
        urlTextField.placeholderString = "http://localhost:8000"
        contentView.addSubview(urlTextField)
        
        // Status Label
        statusLabel = NSTextField(labelWithString: "")
        statusLabel.frame = NSRect(x: 20, y: 120, width: 360, height: 20)
        statusLabel.isEditable = false
        statusLabel.isBordered = false
        statusLabel.backgroundColor = .clear
        contentView.addSubview(statusLabel)
        
        // Test Button
        testButton = NSButton(title: "Test Connection", target: self, action: #selector(testConnection))
        testButton.frame = NSRect(x: 20, y: 80, width: 120, height: 32)
        contentView.addSubview(testButton)
        
        // Save Button
        saveButton = NSButton(title: "Save", target: self, action: #selector(saveURL))
        saveButton.frame = NSRect(x: 220, y: 20, width: 80, height: 32)
        saveButton.keyEquivalent = "\r" // Enter key
        contentView.addSubview(saveButton)
        
        // Cancel Button
        cancelButton = NSButton(title: "Cancel", target: self, action: #selector(cancel))
        cancelButton.frame = NSRect(x: 310, y: 20, width: 80, height: 32)
        cancelButton.keyEquivalent = "\u{1b}" // Escape key
        contentView.addSubview(cancelButton)
    }
    
    @objc private func testConnection() {
        let url = urlTextField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard !url.isEmpty else {
            statusLabel.stringValue = "Please enter a URL"
            statusLabel.textColor = .systemRed
            return
        }
        
        guard networkManager.serverURLManager.isValidURL(url) else {
            statusLabel.stringValue = "Invalid URL format"
            statusLabel.textColor = .systemRed
            return
        }
        
        statusLabel.stringValue = "Testing connection..."
        statusLabel.textColor = .systemBlue
        testButton.isEnabled = false
        
        Task {
            networkManager.updateServerURL(url)
            let connected = await networkManager.testConnection()
            
            await MainActor.run {
                if connected {
                    statusLabel.stringValue = "✅ Connection successful!"
                    statusLabel.textColor = .systemGreen
                } else {
                    statusLabel.stringValue = "❌ Connection failed"
                    statusLabel.textColor = .systemRed
                }
                testButton.isEnabled = true
            }
        }
    }
    
    @objc private func saveURL() {
        let url = urlTextField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        completionHandler?(url)
        window?.close()
    }
    
    @objc private func cancel() {
        window?.close()
    }
}

// MARK: - Status Icon Manager
class StatusIconManager {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private var animationTimer: Timer?
    private var currentFrame = 0
    private let animationFrames = ["brain.head.profile", "brain.head.profile.fill", "brain.head.profile"]
    
    init() {
        setupStatusItem()
    }
    
    private func setupStatusItem() {
        guard let button = statusItem.button else { return }
        button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
        // Don't set action here - let MenuBarApp handle it
    }
    
    func updateStatus(_ connectionStatus: ConnectionStatus, _ appStatus: AppStatus) {
        guard let button = statusItem.button else { return }
        
        // Stop any existing animation
        animationTimer?.invalidate()
        animationTimer = nil
        
        switch (connectionStatus, appStatus) {
        case (.connecting, _):
            startConnectingAnimation()
        case (.connected, .running):
            button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
            button.image?.isTemplate = false
            button.image?.lockFocus()
            NSColor.systemOrange.set()
            button.image?.unlockFocus()
        case (.training, _):
            button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
            button.image?.isTemplate = false
            button.image?.lockFocus()
            NSColor.systemGreen.set()
            button.image?.unlockFocus()
        case (.stopped, _), (_, .stopped):
            button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
            button.image?.isTemplate = false
            button.image?.lockFocus()
            NSColor.systemRed.set()
            button.image?.unlockFocus()
        case (.disconnected, _):
            button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
            button.image?.isTemplate = true
        default:
            button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
            button.image?.isTemplate = true
        }
    }
    
    private func startConnectingAnimation() {
        guard let button = statusItem.button else { return }
        
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            let frameName = self.animationFrames[self.currentFrame]
            button.image = NSImage(systemSymbolName: frameName, accessibilityDescription: "Constellation")
            button.image?.isTemplate = false
            button.image?.lockFocus()
            NSColor.systemBlue.set()
            button.image?.unlockFocus()
            
            self.currentFrame = (self.currentFrame + 1) % self.animationFrames.count
        }
    }
    
    func getStatusItem() -> NSStatusItem {
        return statusItem
    }
}

// MARK: - App Instance Manager
class AppInstanceManager {
    static let shared = AppInstanceManager()
    private let lockFilePath: String
    private var lockFileHandle: FileHandle?
    
    private init() {
        let tempDir = NSTemporaryDirectory()
        lockFilePath = "\(tempDir)constellation_app.lock"
    }
    
    func tryAcquireLock() -> Bool {
        // Check if another instance is already running
        if FileManager.default.fileExists(atPath: lockFilePath) {
            // Try to read the PID from the lock file
            if let data = FileManager.default.contents(atPath: lockFilePath),
               let pidString = String(data: data, encoding: .utf8),
               let pid = Int32(pidString) {
                
                // Check if the process is still running
                if isProcessRunning(pid: pid) {
                    print("❌ Another instance of Constellation is already running (PID: \(pid))")
                    return false
                } else {
                    // Process is dead, remove stale lock file
                    try? FileManager.default.removeItem(atPath: lockFilePath)
                }
            }
        }
        
        // Create lock file with current process PID
        let pid = ProcessInfo.processInfo.processIdentifier
        let pidData = String(pid).data(using: .utf8)
        
        do {
            try pidData?.write(to: URL(fileURLWithPath: lockFilePath))
            print("✅ Acquired app lock (PID: \(pid))")
            return true
        } catch {
            print("❌ Failed to create lock file: \(error)")
            return false
        }
    }
    
    func releaseLock() {
        try? FileManager.default.removeItem(atPath: lockFilePath)
        print("🔓 Released app lock")
    }
    
    private func isProcessRunning(pid: Int32) -> Bool {
        let result = kill(pid, 0)
        return result == 0 || errno == EPERM
    }
}

// MARK: - Menu Bar App
class MenuBarApp: NSObject {
    static let shared = MenuBarApp()
    
    private let statusIconManager = StatusIconManager()
    private let networkManager = NetworkManager()
    private let trainingManager: TrainingManager
    private var statusUpdateTimer: Timer?
    
    private override init() {
        print("🚀 DEBUG: MenuBarApp init() called")
        self.trainingManager = TrainingManager(networkManager: networkManager)
        super.init()
        print("🚀 DEBUG: Calling setupMenuBar()")
        setupMenuBar()
        print("🚀 DEBUG: Calling startStatusUpdates()")
        startStatusUpdates()
        print("📡 Server URL: \(networkManager.baseURL)")
        print("🌐 Dashboard: https://constellation-dashboard.onrender.com")
        print("🚀 DEBUG: MenuBarApp initialization complete")
    }
    
    static func createInstance() -> MenuBarApp? {
        return shared
    }
    
    private func setupMenuBar() {
        guard let button = statusIconManager.getStatusItem().button else { 
            print("❌ DEBUG: Failed to get status item button")
            return 
        }
        
        print("✅ DEBUG: Setting up menu bar button")
        button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Constellation")
        
        print("✅ DEBUG: Creating menu immediately")
        createMenu()
        
        print("✅ DEBUG: Button setup complete")
    }
    
    private func startStatusUpdates() {
        statusUpdateTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateStatusIcon()
        }
    }
    
    private func updateStatusIcon() {
        statusIconManager.updateStatus(networkManager.connectionStatus, trainingManager.appStatus)
        // Refresh menu when status changes
        createMenu()
    }
    
    private func createMenu() {
        let menu = NSMenu()
        
        // Server Configuration
        let serverURLDisplay = networkManager.baseURL.replacingOccurrences(of: "https://", with: "").replacingOccurrences(of: "http://", with: "")
        let serverItem = NSMenuItem(title: "Server: \(serverURLDisplay)", action: #selector(configureServer), keyEquivalent: "")
        serverItem.target = self
        serverItem.isEnabled = true
        serverItem.toolTip = "Click to change server URL"
        menu.addItem(serverItem)
        
        menu.addItem(NSMenuItem.separator())
        
        // Status
        let statusItem = NSMenuItem(title: "Status: \(trainingManager.status)", action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        menu.addItem(statusItem)
        
        // Connection Status
        let connectionStatusText = getConnectionStatusText()
        let connectionItem = NSMenuItem(title: "Connection: \(connectionStatusText)", action: nil, keyEquivalent: "")
        connectionItem.isEnabled = false
        menu.addItem(connectionItem)
        
        if let job = trainingManager.currentJob {
            let jobItem = NSMenuItem(title: "Job: \(job.name)", action: nil, keyEquivalent: "")
            jobItem.isEnabled = false
            menu.addItem(jobItem)
            
            let progressItem = NSMenuItem(title: "Progress: \(Int(trainingManager.progress))%", action: nil, keyEquivalent: "")
            progressItem.isEnabled = false
            menu.addItem(progressItem)
            
            // Show detailed training information
            if trainingManager.isTraining {
                let epochItem = NSMenuItem(title: "Epoch: \(trainingManager.currentEpoch)/\(trainingManager.totalEpochs)", action: nil, keyEquivalent: "")
                epochItem.isEnabled = false
                menu.addItem(epochItem)
                
                if trainingManager.trainingAccuracy > 0 {
                    let accuracyItem = NSMenuItem(title: "Accuracy: \(String(format: "%.1f", trainingManager.trainingAccuracy))%", action: nil, keyEquivalent: "")
                    accuracyItem.isEnabled = false
                    menu.addItem(accuracyItem)
                }
                
                if trainingManager.trainingLoss > 0 {
                    let lossItem = NSMenuItem(title: "Loss: \(String(format: "%.4f", trainingManager.trainingLoss))", action: nil, keyEquivalent: "")
                    lossItem.isEnabled = false
                    menu.addItem(lossItem)
                }
                
                if trainingManager.estimatedTimeRemaining > 0 {
                    let timeRemaining = formatTimeInterval(trainingManager.estimatedTimeRemaining)
                    let timeItem = NSMenuItem(title: "ETA: \(timeRemaining)", action: nil, keyEquivalent: "")
                    timeItem.isEnabled = false
                    menu.addItem(timeItem)
                }
                
                if let startTime = trainingManager.trainingStartTime {
                    let elapsed = Date().timeIntervalSince(startTime)
                    let elapsedFormatted = formatTimeInterval(elapsed)
                    let elapsedItem = NSMenuItem(title: "Elapsed: \(elapsedFormatted)", action: nil, keyEquivalent: "")
                    elapsedItem.isEnabled = false
                    menu.addItem(elapsedItem)
                }
            }
        }
        
        menu.addItem(NSMenuItem.separator())
        
        // Controls (only show if connected)
        if networkManager.connectionStatus == .connected {
            if trainingManager.isTraining {
                let stopItem = NSMenuItem(title: "Stop Training", action: #selector(stopTraining), keyEquivalent: "")
                stopItem.target = self
                stopItem.isEnabled = true
                menu.addItem(stopItem)
            } else {
                let startItem = NSMenuItem(title: "Start Training", action: #selector(startTraining), keyEquivalent: "")
                startItem.target = self
                startItem.isEnabled = true
                menu.addItem(startItem)
            }
        } else {
            let connectItem = NSMenuItem(title: "Connect to Server", action: #selector(connectToServer), keyEquivalent: "")
            connectItem.target = self
            connectItem.isEnabled = true
            menu.addItem(connectItem)
        }
        
        menu.addItem(NSMenuItem.separator())
        
        // Info
        if let device = trainingManager.deviceInfo {
            let deviceItem = NSMenuItem(title: "Device: \(device.name)", action: nil, keyEquivalent: "")
            deviceItem.isEnabled = false
            menu.addItem(deviceItem)
            
            // Show compute resources
            let (gpuAvailable, gpuCores, gpuMemoryGB) = networkManager.detectGPU()
            let cpuCores = ProcessInfo.processInfo.processorCount
            
            if gpuAvailable {
                let allocatedCores = gpuCores / 2
                let computeItem = NSMenuItem(title: "Compute: \(allocatedCores) GPU cores (50% of \(gpuCores))", action: nil, keyEquivalent: "")
                computeItem.isEnabled = false
                menu.addItem(computeItem)
                
                let memoryItem = NSMenuItem(title: "GPU Memory: \(gpuMemoryGB)GB", action: nil, keyEquivalent: "")
                memoryItem.isEnabled = false
                menu.addItem(memoryItem)
            } else {
                let allocatedCores = Int(Double(cpuCores) * 0.75)
                let computeItem = NSMenuItem(title: "Compute: \(allocatedCores) CPU cores (75% of \(cpuCores))", action: nil, keyEquivalent: "")
                computeItem.isEnabled = false
                menu.addItem(computeItem)
            }
        }
        
        menu.addItem(NSMenuItem.separator())
        
        // Quit
        let quitItem = NSMenuItem(title: "Quit Constellation", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        quitItem.isEnabled = true
        menu.addItem(quitItem)
        
        statusIconManager.getStatusItem().menu = menu
    }
    
    private func getConnectionStatusText() -> String {
        switch networkManager.connectionStatus {
        case .disconnected:
            return "Disconnected"
        case .connecting:
            return "Connecting..."
        case .connected:
            return "Connected"
        case .training:
            return "Training"
        case .stopped:
            return "Stopped"
        }
    }
    
    private func formatTimeInterval(_ interval: TimeInterval) -> String {
        let hours = Int(interval) / 3600
        let minutes = Int(interval) % 3600 / 60
        let seconds = Int(interval) % 60
        
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        } else if minutes > 0 {
            return String(format: "%d:%02d", minutes, seconds)
        } else {
            return String(format: "%ds", seconds)
        }
    }
    
    @objc private func configureServer() {
        print("🔧 [BUTTON] Configure Server clicked!")
        let dialog = ServerURLDialog(currentURL: networkManager.baseURL) { [weak self] newURL in
            guard let self = self else { return }
            print("🔧 Updating server URL to: \(newURL)")
            self.networkManager.updateServerURL(newURL)
            Task {
                await self.trainingManager.start()
            }
        }
        
        dialog.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func connectToServer() {
        print("🔌 [BUTTON] Connect to Server clicked!")
        print("🔌 Connecting to: \(networkManager.baseURL)")
        trainingManager.status = "Connecting..."
        Task {
            await trainingManager.start()
        }
    }
    
    @objc private func startTraining() {
        print("🚀 [BUTTON] Start Training clicked!")
        Task {
            print("🚀 Manually checking for jobs...")
            await trainingManager.checkForJobs()
        }
    }
    
    @objc private func stopTraining() {
        print("🛑 [BUTTON] Stop Training clicked!")
        trainingManager.stopTraining()
    }
    
    @objc private func quitApp() {
        print("🛑 [BUTTON] Quit App clicked!")
        trainingManager.stop()
        NSApplication.shared.terminate(nil)
    }
}

// MARK: - Main Application
print("🧠 Project Constellation - Enhanced Distributed AI Training")
print(String(repeating: "=", count: 60))
print("🚀 DEBUG: Starting application...")
print("🚀 DEBUG: Creating MenuBarApp instance...")
print("🌐 Dashboard: https://constellation-dashboard.onrender.com")
print("🛑 Press Ctrl+C to quit")
print("")

// Check if another instance is already running
let instanceManager = AppInstanceManager.shared
guard instanceManager.tryAcquireLock() else {
    print("❌ Cannot start: Another instance is already running")
    print("💡 If you're sure no other instance is running, delete the lock file:")
    print("   rm \(NSTemporaryDirectory())constellation_app.lock")
    exit(1)
}

// MARK: - App Delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        AppInstanceManager.shared.releaseLock()
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return false // Keep running even if no windows are open (menu bar app)
    }
}

// Set up cleanup on app termination
let app = NSApplication.shared
let appDelegate = AppDelegate() // Keep strong reference
app.delegate = appDelegate

// Create the singleton menu bar app
let menuBarApp = MenuBarApp.shared

// Keep the app running
app.run()