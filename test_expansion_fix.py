#!/usr/bin/env python3
"""
Test the fixed expansion counting logic.
"""

import asyncio
import json
from continuous_rsi_runner import ContinuousRSIRunner

async def test_expansion_fix():
    """Test that expansions are now counted correctly."""
    print("🧪 Testing Fixed Expansion Counting")
    print("=" * 50)
    
    # Create runner
    runner = ContinuousRSIRunner()
    
    # Initialize systems
    print("1. Initializing systems...")
    success = await runner.initialize_systems()
    if not success:
        print("❌ Failed to initialize systems")
        return
    print("✅ Systems initialized")
    
    # Test single expansion cycle
    print("\n2. Testing single expansion cycle...")
    initial_successful = runner.metrics['successful_expansions']
    initial_failed = runner.metrics['failed_expansions']
    
    print(f"   Initial - Successful: {initial_successful}, Failed: {initial_failed}")
    
    # Run expansion cycle
    result = await runner.autonomous_expansion_cycle()
    
    final_successful = runner.metrics['successful_expansions']
    final_failed = runner.metrics['failed_expansions']
    
    print(f"   Final - Successful: {final_successful}, Failed: {final_failed}")
    print(f"   Expansion result: {result}")
    
    # Analyze result
    if result.get('success', True) and result.get('success_rate', 0) > 0.5:
        expected_successful = initial_successful + 1
        expected_failed = initial_failed
        print(f"   Expected successful expansion count increase")
    else:
        expected_successful = initial_successful
        expected_failed = initial_failed + 1
        print(f"   Expected failed expansion count increase")
    
    # Verify counting
    if (final_successful == expected_successful and 
        final_failed == expected_failed):
        print("✅ Expansion counting is working correctly!")
    else:
        print("❌ Expansion counting is still incorrect")
        print(f"   Expected: successful={expected_successful}, failed={expected_failed}")
        print(f"   Actual: successful={final_successful}, failed={final_failed}")
    
    # Save final state
    await runner._save_state()
    
    # Cleanup
    print("\n3. Cleanup...")
    await runner.shutdown()
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_expansion_fix())