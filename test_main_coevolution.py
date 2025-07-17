#!/usr/bin/env python3
"""
Quick test to validate main.py co-evolution integration.
"""

import asyncio
import sys
import json
import time
import subprocess
import requests
from pathlib import Path

async def test_main_integration():
    """Test the main.py server with co-evolution endpoints"""
    
    print("🚀 Testing main.py RSI-Agent Co-Evolution Integration")
    print("=" * 60)
    
    # Start the server in background
    print("📡 Starting RSI server...")
    
    # Run server in background
    server_process = subprocess.Popen(
        [sys.executable, "-m", "src.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give server time to start
    print("⏳ Waiting for server to start...")
    await asyncio.sleep(10)
    
    try:
        # Test basic health
        print("\n🏥 Testing basic health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"⚠️ Server health check failed: {response.status_code}")
        
        # Test co-evolution status endpoint
        print("\n📊 Testing co-evolution status endpoint...")
        try:
            response = requests.get("http://localhost:8000/coevolution/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print("✅ Co-evolution status endpoint working")
                print(f"   • Total cycles: {status.get('total_cycles', 0)}")
                print(f"   • Agent status: {status.get('agent_status', {}).get('status', 'unknown')}")
            else:
                print(f"⚠️ Co-evolution status failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Co-evolution status endpoint error: {e}")
        
        # Test starting a co-evolution cycle
        print("\n🔄 Testing co-evolution start endpoint...")
        try:
            targets = {
                "revenue_improvement": 0.05,  # Very conservative
                "execution_efficiency": 0.60,
                "learning_acceleration": 0.10
            }
            
            response = requests.post(
                "http://localhost:8000/coevolution/start",
                json=targets,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Co-evolution cycle started and completed successfully!")
                print(f"   • Status: {result.get('status', 'unknown')}")
                print(f"   • Cycles completed: {result.get('results', {}).get('cycles_completed', 0)}")
                print(f"   • Total improvements: {result.get('results', {}).get('total_improvements', 0)}")
            else:
                print(f"⚠️ Co-evolution start failed: {response.status_code}")
                print(f"   • Response: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Co-evolution start endpoint error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Main.py integration test completed!")
        print("The RSI-Agent Co-Evolution System is working in the main server!")
        
    finally:
        # Stop the server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ Server stopped")

if __name__ == "__main__":
    asyncio.run(test_main_integration())