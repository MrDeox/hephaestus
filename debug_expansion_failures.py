#!/usr/bin/env python3
"""
Debug script to identify expansion failures.
"""

import asyncio
import logging
from src.main import RSIOrchestrator
from src.memory import RSIMemoryHierarchy, RSIMemoryConfig
from autonomous_expansion_strategy import AutonomousExpansionEngine

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_expansion_system():
    """Debug the expansion system to find failure points."""
    print("🔍 Debugging Expansion System")
    print("=" * 50)
    
    try:
        # Initialize systems
        print("\n1. Initializing orchestrator...")
        orchestrator = RSIOrchestrator(environment='development')
        await orchestrator.start()
        print("✅ Orchestrator initialized")
        
        print("\n2. Initializing memory system...")
        memory_config = RSIMemoryConfig(
            working_memory_capacity=1000,
            max_memory_usage_gb=4,
            monitoring_enabled=True
        )
        memory_system = RSIMemoryHierarchy(memory_config)
        print("✅ Memory system initialized")
        
        # Connect memory to orchestrator
        orchestrator.memory_system = memory_system
        
        print("\n3. Initializing expansion engine...")
        expansion_engine = AutonomousExpansionEngine(orchestrator, memory_system)
        print("✅ Expansion engine initialized")
        
        print("\n4. Testing performance analysis...")
        try:
            performance_analysis = await expansion_engine.analyze_system_performance()
            print(f"✅ Performance analysis completed")
            print(f"   - Memory efficiency: {performance_analysis['memory_efficiency']:.3f}")
            print(f"   - Usage patterns: {performance_analysis['usage_patterns']}")
            print(f"   - Bottlenecks: {len(performance_analysis['bottlenecks'])}")
            print(f"   - Improvements: {len(performance_analysis['improvement_opportunities'])}")
        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n5. Testing expansion proposals...")
        try:
            proposed_expansions = await expansion_engine.propose_expansions(performance_analysis)
            print(f"✅ Expansion proposals completed")
            print(f"   - Proposed expansions: {len(proposed_expansions)}")
            for i, expansion in enumerate(proposed_expansions, 1):
                print(f"      {i}. {expansion['type']}: {expansion['description']} (priority: {expansion['priority']})")
        except Exception as e:
            print(f"❌ Expansion proposals failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n6. Testing expansion implementation...")
        implemented_count = 0
        for expansion in proposed_expansions:
            try:
                print(f"   Attempting: {expansion['description']}")
                success = await expansion_engine.implement_expansion(expansion)
                if success:
                    implemented_count += 1
                    print(f"   ✅ Success: {expansion['description']}")
                else:
                    print(f"   ⚠️ Failed: {expansion['description']}")
            except Exception as e:
                print(f"   ❌ Error implementing {expansion['description']}: {e}")
        
        print(f"\n   Implementation results: {implemented_count}/{len(proposed_expansions)} successful")
        
        print("\n7. Testing full expansion cycle...")
        try:
            cycle_result = await expansion_engine.autonomous_expansion_cycle()
            print(f"✅ Full expansion cycle completed")
            print(f"   - Proposed: {cycle_result['proposed_expansions']}")
            print(f"   - Implemented: {cycle_result['implemented_expansions']}")
            print(f"   - Success rate: {cycle_result['success_rate']:.3f}")
            
            # Check success rate threshold
            if cycle_result['success_rate'] > 0.5:
                print("   ✅ Would be counted as successful expansion")
            else:
                print("   ❌ Would be counted as failed expansion")
                print("   🔍 Likely cause: Low success rate due to implementation failures")
                
        except Exception as e:
            print(f"❌ Full expansion cycle failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        print("\n8. Cleanup...")
        await memory_system.shutdown()
        await orchestrator.stop()
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Critical error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_expansion_system())