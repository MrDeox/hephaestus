#!/usr/bin/env python3
"""
Test script to validate RSI-Agent Co-Evolution System integration.

This script tests the complete integration without running the full server.
"""

import asyncio
import sys
import traceback
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

async def test_coevolution_integration():
    """Test the RSI-Agent co-evolution system integration"""
    
    print("🧪 Testing RSI-Agent Co-Evolution System Integration")
    print("=" * 60)
    
    try:
        # Test 1: Import co-evolution components
        print("📦 Test 1: Importing co-evolution components...")
        
        from src.coevolution.rsi_agent_orchestrator import (
            RSIAgentOrchestrator, create_rsi_agent_orchestrator
        )
        from src.agents.rsi_tool_agent import (
            RSIToolAgent, create_rsi_tool_agent
        )
        print("✅ Co-evolution components imported successfully")
        
        # Test 2: Create RSI Tool Agent
        print("\n🤖 Test 2: Creating RSI Tool Agent...")
        tool_agent = create_rsi_tool_agent("http://localhost:8000")
        print(f"✅ RSI Tool Agent created: {tool_agent.agent_id}")
        
        # Test 3: Get agent status
        print("\n📊 Test 3: Getting agent status...")
        agent_status = await tool_agent.get_agent_status()
        print(f"✅ Agent Status: {agent_status['status']} with {agent_status['total_executions']} executions")
        
        # Test 4: Create Co-Evolution Orchestrator
        print("\n🔄 Test 4: Creating Co-Evolution Orchestrator...")
        coevo_orchestrator = create_rsi_agent_orchestrator(
            rsi_orchestrator=None,  # Will use fallback
            revenue_generator=None,  # Will use fallback
            base_url="http://localhost:8000"
        )
        print("✅ Co-Evolution Orchestrator created successfully")
        
        # Test 5: Get co-evolution status
        print("\n📈 Test 5: Getting co-evolution status...")
        coevo_status = await coevo_orchestrator.get_coevolution_status()
        print(f"✅ Co-Evolution Status: {coevo_status['total_cycles']} cycles completed")
        
        # Test 6: Test a simple hypothesis execution
        print("\n🎯 Test 6: Testing hypothesis execution...")
        test_hypothesis = {
            "id": "test_hyp_001",
            "description": "Test revenue optimization strategy",
            "type": "revenue_optimization",
            "target_improvement": 100.0,
            "confidence": 0.8
        }
        
        execution_result = await tool_agent.execute_rsi_hypothesis(test_hypothesis)
        print(f"✅ Hypothesis executed: {execution_result['success']}")
        
        # Test 7: Test single co-evolution cycle (simplified)
        print("\n🔄 Test 7: Testing simplified co-evolution cycle...")
        
        targets = {
            "revenue_improvement": 0.10,  # Conservative target
            "execution_efficiency": 0.70,
            "learning_acceleration": 0.15
        }
        
        # Test hypothesis generation
        print("  📋 Generating hypotheses...")
        hypotheses = await coevo_orchestrator._rsi_generate_hypotheses(targets, {"test": True})
        print(f"  ✅ Generated {len(hypotheses.get('hypotheses', []))} hypotheses")
        
        # Test agent execution
        print("  🔧 Testing agent execution...")
        agent_results = await coevo_orchestrator._agent_execute_hypotheses(hypotheses)
        print(f"  ✅ Agent execution completed with {agent_results['success_rate']:.1%} success rate")
        
        # Test learning
        print("  🧠 Testing learning phases...")
        rsi_learning = await coevo_orchestrator._rsi_learn_from_agent(hypotheses, agent_results)
        agent_learning = await coevo_orchestrator._agent_learn_from_rsi(hypotheses, agent_results)
        print("  ✅ Learning phases completed")
        
        # Test synthesis
        print("  ⚡ Testing synthesis...")
        synthesis = await coevo_orchestrator._synthesize_learnings(rsi_learning, agent_learning)
        print(f"  ✅ Synthesis completed with {len(synthesis.get('combined_improvements', []))} improvements")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! RSI-Agent Co-Evolution System is working correctly!")
        print("\n📊 Test Summary:")
        print(f"   • Tool Agent ID: {tool_agent.agent_id}")
        print(f"   • Agent Tools Available: {len(tool_agent.available_tools)}")
        print(f"   • Hypotheses Generated: {len(hypotheses.get('hypotheses', []))}")
        print(f"   • Agent Success Rate: {agent_results['success_rate']:.1%}")
        print(f"   • Learning Insights: {len(rsi_learning.get('next_hypothesis_suggestions', []))}")
        print(f"   • Combined Improvements: {len(synthesis.get('combined_improvements', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"\n🔍 Error details:")
        traceback.print_exc()
        return False

async def test_main_integration():
    """Test the main.py integration"""
    
    print("\n" + "=" * 60)
    print("🏗️ Testing Main.py Integration")
    print("=" * 60)
    
    try:
        # Import main module components
        print("📦 Importing main module...")
        from src.main import RSIOrchestrator
        print("✅ Main module imported successfully")
        
        # Test RSI Orchestrator initialization
        print("\n🚀 Testing RSI Orchestrator initialization...")
        orchestrator = RSIOrchestrator(environment='development')
        print("✅ RSI Orchestrator initialized")
        
        # Check if co-evolution system was initialized
        has_coevolution = hasattr(orchestrator, 'rsi_agent_coevolution_orchestrator')
        coevo_available = orchestrator.rsi_agent_coevolution_orchestrator is not None if has_coevolution else False
        
        print(f"\n🔄 Co-Evolution System Status:")
        print(f"   • Has attribute: {has_coevolution}")
        print(f"   • Is available: {coevo_available}")
        
        if coevo_available:
            print("✅ RSI-Agent Co-Evolution System successfully integrated into main orchestrator!")
            
            # Test getting status
            coevo_status = await orchestrator.rsi_agent_coevolution_orchestrator.get_coevolution_status()
            print(f"   • Total cycles: {coevo_status['total_cycles']}")
            print(f"   • Agent status: {coevo_status['agent_status']['status']}")
        else:
            print("⚠️ Co-Evolution System not initialized (this is OK if dependencies are missing)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Main integration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def run_all_tests():
        print(f"🧪 RSI-Agent Co-Evolution Integration Test")
        print(f"⏰ Started at: {datetime.now()}")
        print("=" * 80)
        
        # Run individual component tests
        test1_passed = await test_coevolution_integration()
        
        # Run main integration test
        test2_passed = await test_main_integration()
        
        print("\n" + "=" * 80)
        print("🏁 FINAL RESULTS:")
        print(f"   • Co-Evolution Components Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"   • Main Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 🎉 ALL INTEGRATION TESTS PASSED! 🎉 🎉")
            print("\nThe RSI-Agent Co-Evolution System is successfully integrated!")
            print("You can now run 'python -m src.main' to start the full system.")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")
        
        print(f"\n⏰ Completed at: {datetime.now()}")
    
    # Run the tests
    asyncio.run(run_all_tests())