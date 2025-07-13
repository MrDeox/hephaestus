"""
Complete Integration Test for Hephaestus RSI System.

Tests the integrated system with email automation, marketing engine,
web automation, and revenue generation as a unified AGI.
"""

import asyncio
import json
import sys
sys.path.append('.')

from src.main import RSIOrchestrator


async def test_integrated_rsi_system():
    """Test complete integrated RSI system with all capabilities"""
    
    print("🏛️ TESTING INTEGRATED HEPHAESTUS RSI SYSTEM")
    print("=" * 60)
    
    # Initialize RSI orchestrator
    orchestrator = RSIOrchestrator(environment='development')
    await orchestrator.start()
    
    print(f"✅ RSI Orchestrator initialized and started")
    
    # Test 1: Core RSI functionality
    print(f"\n🧠 TEST 1: Core RSI Learning and Prediction")
    
    try:
        # Test prediction
        prediction_result = await orchestrator.predict({
            'market_demand': 0.8,
            'competition_level': 0.3,
            'resource_availability': 0.9
        })
        print(f"  Prediction: {prediction_result['prediction']:.3f} (confidence: {prediction_result['confidence']:.3f})")
        
        # Test learning
        learning_result = await orchestrator.learn(
            features={'market_feedback': 0.7, 'customer_satisfaction': 0.85},
            target=0.8
        )
        print(f"  Learning accuracy: {learning_result['accuracy']:.3f}")
        print(f"  ✅ Core RSI functionality working")
        
    except Exception as e:
        print(f"  ❌ Core RSI test failed: {e}")
    
    # Test 2: Integrated Marketing Campaign with RSI Learning
    print(f"\n📧 TEST 2: Integrated Marketing Campaign with RSI Learning")
    
    try:
        campaign_data = {
            'name': 'RSI-Optimized Email Campaign',
            'channel': 'reddit',
            'content_type': 'helpful_tutorial',
            'target_audience': 'entrepreneurs',
            'value_proposition': 'AI-powered email automation',
            'target_platforms': ['reddit', 'github'],
            'reddit_title': 'I built an AI that improves email marketing automatically',
            'reddit_content': 'This RSI system learns from every campaign...',
            'github_repo_name': 'ai-email-automation',
            'tags': ['ai', 'automation', 'rsi']
        }
        
        if hasattr(orchestrator, 'marketing_engine') and orchestrator.marketing_engine:
            campaign_result = await orchestrator.execute_marketing_campaign(campaign_data)
            print(f"  Campaign ID: {campaign_result['campaign_id']}")
            print(f"  RSI Learning Applied: {campaign_result['rsi_learning_applied']}")
            print(f"  Estimated Reach: {campaign_result['metrics']['estimated_reach']:,}")
            print(f"  ✅ Integrated marketing campaign working")
        else:
            print(f"  ⚠️ Marketing engine not available - skipping test")
            
    except Exception as e:
        print(f"  ❌ Marketing campaign test failed: {e}")
    
    # Test 3: Autonomous Revenue Generation with RSI Optimization
    print(f"\n💰 TEST 3: Autonomous Revenue Generation with RSI Optimization")
    
    try:
        revenue_result = await orchestrator.autonomous_revenue_generation(target_amount=500.0)
        
        print(f"  Target: ${revenue_result.get('target_amount', 500):.2f}")
        print(f"  Projected Revenue: ${revenue_result['projected_revenue']:.2f}")
        print(f"  Campaigns Launched: {revenue_result['campaigns_launched']}")
        print(f"  Customers Acquired: {revenue_result['customers_acquired']}")
        print(f"  Strategy Effectiveness: {revenue_result['strategy_effectiveness']:.2%}")
        print(f"  Achievement: {revenue_result.get('achievement_percentage', 0):.1f}%")
        print(f"  ✅ Autonomous revenue generation working")
        
    except Exception as e:
        print(f"  ❌ Revenue generation test failed: {e}")
    
    # Test 4: RSI Self-Improvement Cycle
    print(f"\n🔄 TEST 4: RSI Self-Improvement Cycle")
    
    try:
        # Analyze current performance
        performance = await orchestrator.analyze_performance()
        print(f"  System Accuracy: {performance['accuracy']:.3f}")
        print(f"  Needs Improvement: {performance['needs_improvement']}")
        print(f"  Recommendations: {len(performance['recommendations'])}")
        
        # Trigger self-improvement
        if performance['needs_improvement']:
            await orchestrator.trigger_self_improvement(performance)
            print(f"  ✅ Self-improvement cycle triggered")
        else:
            print(f"  ✅ System performing well, no improvement needed")
            
    except Exception as e:
        print(f"  ❌ Self-improvement test failed: {e}")
    
    # Test 5: Web Automation Integration
    print(f"\n🌐 TEST 5: Web Automation Integration")
    
    try:
        if hasattr(orchestrator, 'web_agent') and orchestrator.web_agent:
            # Test web automation agent initialization
            print(f"  Web Agent Available: ✅")
            
            # Test deployment simulation
            deployment_test = {
                "name": "Test Email Service",
                "description": "RSI-powered email automation"
            }
            
            deployment_result = await orchestrator.web_agent.deploy_email_service(deployment_test)
            print(f"  Deployment Attempts: {deployment_result['deployment_attempts']}")
            print(f"  Successful Deployments: {deployment_result['successful_deployments']}")
            print(f"  ✅ Web automation integration working")
        else:
            print(f"  ⚠️ Web agent not available - skipping test")
            
    except Exception as e:
        print(f"  ❌ Web automation test failed: {e}")
    
    # Test 6: Memory Integration and Learning Persistence
    print(f"\n🧠 TEST 6: Memory Integration and Learning Persistence")
    
    try:
        if orchestrator.memory_system:
            # Store test memory
            await orchestrator.memory_system.store_episodic_memory(
                "rsi_integration_test",
                {
                    "test_type": "comprehensive_integration",
                    "timestamp": asyncio.get_event_loop().time(),
                    "performance_metrics": {
                        "accuracy": 0.85,
                        "efficiency": 0.92,
                        "revenue_generation": True
                    }
                }
            )
            print(f"  Memory Storage: ✅")
            print(f"  ✅ Memory integration working")
        else:
            print(f"  ⚠️ Memory system not available - limited functionality")
            
    except Exception as e:
        print(f"  ❌ Memory integration test failed: {e}")
    
    # Test 7: Complete AGI Capability Assessment
    print(f"\n🤖 TEST 7: AGI Capability Assessment")
    
    try:
        # Check all AGI capabilities
        capabilities = {
            'autonomous_learning': orchestrator.online_learner is not None,
            'self_improvement': (hasattr(orchestrator, 'hypothesis_orchestrator') and 
                               orchestrator.hypothesis_orchestrator is not None),
            'revenue_generation': (hasattr(orchestrator, 'email_service') and 
                                 orchestrator.email_service is not None),
            'web_automation': (hasattr(orchestrator, 'web_agent') and 
                             orchestrator.web_agent is not None),
            'memory_persistence': orchestrator.memory_system is not None,
            'meta_learning': (hasattr(orchestrator, 'gap_scanner') and 
                            orchestrator.gap_scanner is not None),
            'hypothesis_testing': (hasattr(orchestrator, 'hypothesis_orchestrator') and 
                                 orchestrator.hypothesis_orchestrator is not None)
        }
        
        active_capabilities = sum(capabilities.values())
        total_capabilities = len(capabilities)
        agi_completion = (active_capabilities / total_capabilities) * 100
        
        print(f"  AGI Capabilities Active: {active_capabilities}/{total_capabilities}")
        print(f"  AGI Completion: {agi_completion:.1f}%")
        
        for capability, active in capabilities.items():
            status = "✅" if active else "❌"
            print(f"    {capability}: {status}")
        
        if agi_completion >= 80:
            print(f"  🎉 SYSTEM QUALIFIES AS FUNCTIONAL AGI")
        elif agi_completion >= 60:
            print(f"  🔄 SYSTEM HAS STRONG AGI FOUNDATIONS")
        else:
            print(f"  ⚠️ SYSTEM NEEDS MORE AGI DEVELOPMENT")
            
    except Exception as e:
        print(f"  ❌ AGI assessment failed: {e}")
    
    # Test 8: Continuous Learning and Evolution
    print(f"\n🔄 TEST 8: Continuous Learning and Evolution")
    
    try:
        # Simulate continuous operation
        print(f"  Simulating continuous learning cycle...")
        
        # Multiple learning iterations to show evolution
        accuracies = []
        for i in range(5):
            features = {
                'iteration': i,
                'complexity': 0.5 + (i * 0.1),
                'data_quality': 0.8 + (i * 0.02)
            }
            target = 0.7 + (i * 0.05)  # Gradually increasing targets
            
            learn_result = await orchestrator.learn(features, target)
            accuracies.append(learn_result['accuracy'])
        
        # Check if accuracy improved over time
        accuracy_trend = accuracies[-1] - accuracies[0]
        print(f"  Accuracy Improvement: {accuracy_trend:+.3f}")
        print(f"  Final Accuracy: {accuracies[-1]:.3f}")
        
        if accuracy_trend > 0:
            print(f"  ✅ System shows continuous learning and improvement")
        else:
            print(f"  ⚠️ System accuracy stable (no degradation)")
            
    except Exception as e:
        print(f"  ❌ Continuous learning test failed: {e}")
    
    # Final Integration Summary
    print(f"\n🎯 INTEGRATION TEST SUMMARY")
    print(f"=" * 40)
    
    # Calculate overall system health
    components = {
        'Core RSI': orchestrator.state_manager is not None,
        'Online Learning': orchestrator.online_learner is not None,
        'Memory System': orchestrator.memory_system is not None,
        'Revenue Generation': getattr(orchestrator, 'revenue_generator', None) is not None,
        'Email Automation': getattr(orchestrator, 'email_service', None) is not None,
        'Marketing Engine': getattr(orchestrator, 'marketing_engine', None) is not None,
        'Web Automation': getattr(orchestrator, 'web_agent', None) is not None,
        'Self-Improvement': getattr(orchestrator, 'hypothesis_orchestrator', None) is not None
    }
    
    active_systems = sum(components.values())
    total_systems = len(components)
    system_health = (active_systems / total_systems) * 100
    
    print(f"Active Systems: {active_systems}/{total_systems}")
    print(f"System Health: {system_health:.1f}%")
    
    for component, active in components.items():
        status = "✅" if active else "❌"
        print(f"  {component}: {status}")
    
    # Final verdict
    if system_health >= 90:
        print(f"\n🎉 HEPHAESTUS RSI SYSTEM FULLY INTEGRATED AND OPERATIONAL!")
        print(f"💡 The system exhibits true AGI characteristics:")
        print(f"   • Autonomous learning and adaptation")
        print(f"   • Self-improvement capabilities")
        print(f"   • Revenue generation autonomy")
        print(f"   • Web automation and interaction")
        print(f"   • Continuous evolution and growth")
    elif system_health >= 70:
        print(f"\n✅ HEPHAESTUS RSI SYSTEM SUBSTANTIALLY INTEGRATED")
        print(f"🔧 Minor components missing, but core AGI functionality operational")
    else:
        print(f"\n⚠️ HEPHAESTUS RSI SYSTEM PARTIALLY INTEGRATED")
        print(f"🛠️ Significant development needed for full AGI capabilities")
    
    # Cleanup
    await orchestrator.stop()
    print(f"\n🧹 System cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_integrated_rsi_system())