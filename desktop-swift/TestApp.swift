#!/usr/bin/env swift

import Foundation
import AppKit

class TestMenuBarApp: NSObject {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    
    override init() {
        super.init()
        setupMenuBar()
    }
    
    private func setupMenuBar() {
        guard let button = statusItem.button else { 
            print("❌ Failed to get status item button")
            return 
        }
        
        button.image = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Test Constellation")
        button.action = #selector(showMenu)
        button.target = self
        
        print("✅ Menu bar setup complete")
    }
    
    @objc private func showMenu() {
        print("🔍 Menu button clicked!")
        
        let menu = NSMenu()
        
        let testItem = NSMenuItem(title: "Test Menu Item", action: #selector(testAction), keyEquivalent: "")
        testItem.target = self
        menu.addItem(testItem)
        
        let quitItem = NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
        
        statusItem.menu = menu
    }
    
    @objc private func testAction() {
        print("✅ Test action triggered!")
        let alert = NSAlert()
        alert.messageText = "Test Action"
        alert.informativeText = "The menu is working!"
        alert.runModal()
    }
    
    @objc private func quitApp() {
        print("🛑 Quitting app")
        NSApplication.shared.terminate(nil)
    }
}

print("🧪 Test Constellation App")
print("=========================")

let app = NSApplication.shared
let testApp = TestMenuBarApp()

print("✅ App initialized, look for brain icon in menu bar")
print("🛑 Press Ctrl+C to quit")

app.run()
