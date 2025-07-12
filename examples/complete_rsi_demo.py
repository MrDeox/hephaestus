#!/usr/bin/env python3
"""
🏛️ Hephaestus Complete RSI System Demonstration
Demonstrates the full recursive self-improvement cycle with hypothesis testing.
"""

import asyncio
import json
import time
from typing import Dict, Any

import requests
import numpy as np
from loguru import logger

# Configure logger
logger.add("rsi_demo.log", rotation="10 MB")

class HephaestusRSIDemo:
    """Comprehensive demonstration of the Hephaestus RSI system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_system_status(self) -> Dict[str, Any]:
        """Check if the system is running and hypothesis testing is available"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to connect to Hephaestus system: {}", str(e))
            raise
    
    def run_prediction_demo(self) -> Dict[str, Any]:
        """Demonstrate prediction capabilities with uncertainty quantification"""
        logger.info("🔮 Running prediction demonstration...")
        
        # Create sample features
        features = {
            "feature_1": 0.5,
            "feature_2": 0.8,
            "feature_3": 0.3,
            "feature_4": 0.9,
            "feature_5": 0.1
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={
                    "features": features,
                    "user_id": "demo_user",
                    "uncertainty_estimation": True
                }
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("✅ Prediction successful:")
            logger.info("   - Prediction: {:.4f}", result["prediction"])
            logger.info("   - Confidence: {:.4f}", result["confidence"])
            if "uncertainty" in result:
                uncertainty = result["uncertainty"]
                logger.info("   - Total Uncertainty: {:.4f}", uncertainty.get("total_uncertainty", 0))
                logger.info("   - Confidence Interval: {}", uncertainty.get("confidence_interval", []))
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed: {}", str(e))
            raise
    
    def run_learning_demo(self) -> Dict[str, Any]:
        """Demonstrate online learning capabilities"""
        logger.info("🧠 Running learning demonstration...")
        
        # Create sample training data
        features = {
            "feature_1": 0.7,
            "feature_2": 0.4,
            "feature_3": 0.6,
            "feature_4": 0.2,
            "feature_5": 0.9
        }
        target = 0.65  # Target value
        
        try:
            response = self.session.post(
                f"{self.base_url}/learn",
                json={
                    "features": features,
                    "target": target,
                    "user_id": "demo_user",
                    "safety_level": "medium"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("✅ Learning successful:")
            logger.info("   - Accuracy: {:.4f}", result.get("accuracy", 0))
            logger.info("   - Samples Processed: {}", result.get("samples_processed", 0))
            logger.info("   - Concept Drift: {}", result.get("concept_drift_detected", False))
            logger.info("   - Memory Stored: {}", result.get("memory_stored", False))
            
            return result
            
        except Exception as e:
            logger.error("Learning failed: {}", str(e))
            raise
    
    def run_hypothesis_generation_demo(self) -> Dict[str, Any]:
        """Demonstrate the complete hypothesis generation and testing cycle"""
        logger.info("🧪 Running hypothesis generation demonstration...")
        
        # Define improvement targets
        improvement_targets = {
            "accuracy": 0.05,      # Improve accuracy by 5%
            "efficiency": 0.1,     # Improve efficiency by 10%
            "robustness": 0.03     # Improve robustness by 3%
        }
        
        # Additional context
        context = {
            "current_performance": {
                "accuracy": 0.82,
                "efficiency": 0.75,
                "robustness": 0.70
            },
            "constraints": {
                "max_computational_cost": 100,
                "max_risk_level": 0.6
            },
            "environment": "demo"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/hypothesis/generate",
                json={
                    "improvement_targets": improvement_targets,
                    "context": context,
                    "max_hypotheses": 8,
                    "user_id": "demo_user"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("✅ Hypothesis generation successful:")
            logger.info("   - Hypotheses Generated: {}", result["hypotheses_generated"])
            logger.info("   - Deployment Ready: {}", result["deployment_ready_count"])
            
            # Display details of generated hypotheses
            for i, hypothesis in enumerate(result["results"][:3], 1):  # Show first 3
                logger.info(f"   📋 Hypothesis {i}:")
                logger.info(f"      - Type: {hypothesis['hypothesis_type']}")
                logger.info(f"      - Description: {hypothesis['description']}")
                logger.info(f"      - Status: {hypothesis['status']}")
                logger.info(f"      - Priority: {hypothesis['priority']}")
                logger.info(f"      - Risk Level: {hypothesis['risk_level']:.3f}")
                logger.info(f"      - Confidence: {hypothesis.get('confidence_score', 'N/A')}")
                logger.info(f"      - Deployment Ready: {hypothesis['deployment_ready']}")
                logger.info(f"      - Expected Improvement: {hypothesis['expected_improvement']}")
                
                if hypothesis['deployment_ready']:
                    logger.info(f"      ✅ Ready for deployment!")
                logger.info("")
            
            return result
            
        except Exception as e:
            logger.error("Hypothesis generation failed: {}", str(e))
            raise
    
    def monitor_hypothesis_status(self, hypothesis_id: str) -> Dict[str, Any]:
        """Monitor the status of a specific hypothesis"""
        logger.info("📊 Monitoring hypothesis status: {}", hypothesis_id)
        
        try:
            response = self.session.get(f"{self.base_url}/hypothesis/status/{hypothesis_id}")
            response.raise_for_status()
            result = response.json()
            
            logger.info("📈 Hypothesis Status:")
            logger.info("   - Status: {}", result["status"])
            logger.info("   - Current Phase: {}", result["current_phase"])
            logger.info("   - Confidence Score: {}", result.get("confidence_score", "N/A"))
            logger.info("   - Deployment Ready: {}", result["deployment_ready"])
            logger.info("   - Total Duration: {:.2f}s", result.get("total_duration_seconds", 0))
            
            if result.get("phase_durations"):
                logger.info("   - Phase Durations:")
                for phase, duration in result["phase_durations"].items():
                    logger.info(f"     • {phase}: {duration:.2f}s")
            
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("Hypothesis not found: {}", hypothesis_id)
                return {"error": "not_found"}
            else:
                logger.error("Error getting hypothesis status: {}", str(e))
                raise
        except Exception as e:
            logger.error("Error monitoring hypothesis: {}", str(e))
            raise
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        logger.info("📊 Getting system statistics...")
        
        try:
            # Get various system metrics
            health_response = self.session.get(f"{self.base_url}/health")
            health_response.raise_for_status()
            health_data = health_response.json()
            
            # Get hypothesis statistics if available
            hypothesis_stats = {}
            try:
                hyp_response = self.session.get(f"{self.base_url}/hypothesis/statistics")
                if hyp_response.status_code == 200:
                    hypothesis_stats = hyp_response.json()
            except:
                pass
            
            # Get metacognitive status
            metacog_response = self.session.get(f"{self.base_url}/metacognitive-status")
            metacog_response.raise_for_status()
            metacog_data = metacog_response.json()
            
            logger.info("📈 System Statistics:")
            logger.info("   - System Health: {}", health_data.get("status", "unknown"))
            logger.info("   - Uptime: {:.2f} hours", health_data.get("uptime_seconds", 0) / 3600)
            
            if "metacognitive_status" in health_data:
                meta_status = health_data["metacognitive_status"]
                logger.info("   - Metacognitive Awareness: {:.3f}", meta_status.get("metacognitive_awareness", 0))
                logger.info("   - Learning Efficiency: {:.3f}", meta_status.get("learning_efficiency", 0))
                logger.info("   - Safety Score: {:.3f}", meta_status.get("safety_score", 0))
            
            if hypothesis_stats:
                logger.info("   🧪 Hypothesis System:")
                logger.info("      - Total Orchestrations: {}", hypothesis_stats.get("total_orchestrations", 0))
                logger.info("      - Deployment Rate: {:.1%}", hypothesis_stats.get("deployment_rate", 0))
                logger.info("      - Avg Confidence: {:.3f}", hypothesis_stats.get("avg_confidence_score", 0))
            
            return {
                "health": health_data,
                "hypothesis_stats": hypothesis_stats,
                "metacognitive": metacog_data
            }
            
        except Exception as e:
            logger.error("Error getting system statistics: {}", str(e))
            raise
    
    def demonstrate_complete_cycle(self):
        """Demonstrate the complete RSI cycle"""
        logger.info("🎯 Starting Complete RSI Cycle Demonstration")
        logger.info("=" * 60)
        
        try:
            # 1. Check system status
            logger.info("🔍 Phase 1: System Status Check")
            status = self.check_system_status()
            logger.info("✅ System Status: {}", status["message"])
            logger.info("🔧 Features Available: {}", len(status["features"]))
            logger.info("🧪 Hypothesis System: {}", status.get("hypothesis_system_available", False))
            logger.info("")
            
            # 2. Run basic prediction
            logger.info("🔍 Phase 2: Basic System Operation")
            prediction_result = self.run_prediction_demo()
            time.sleep(1)
            
            # 3. Run learning
            learning_result = self.run_learning_demo()
            time.sleep(1)
            logger.info("")
            
            # 4. Generate hypotheses for improvement
            logger.info("🔍 Phase 3: RSI Hypothesis Generation & Testing")
            hypothesis_result = self.run_hypothesis_generation_demo()
            time.sleep(2)
            
            # 5. Monitor hypothesis status (if any were generated)
            if hypothesis_result and hypothesis_result["results"]:
                first_hypothesis = hypothesis_result["results"][0]
                hypothesis_id = first_hypothesis["hypothesis_id"]
                
                logger.info("🔍 Phase 4: Hypothesis Monitoring")
                status_result = self.monitor_hypothesis_status(hypothesis_id)
                time.sleep(1)
                logger.info("")
            
            # 6. Get comprehensive system statistics
            logger.info("🔍 Phase 5: System Analysis")
            stats = self.get_system_statistics()
            logger.info("")
            
            # 7. Summary
            logger.info("🎉 RSI Cycle Demonstration Complete!")
            logger.info("=" * 60)
            logger.info("📊 Summary:")
            logger.info("   ✅ Prediction with uncertainty quantification")
            logger.info("   ✅ Online learning with concept drift detection")
            logger.info("   ✅ Hypothesis generation and validation")
            logger.info("   ✅ Safety verification and testing")
            logger.info("   ✅ Comprehensive system monitoring")
            logger.info("")
            logger.info("🚀 The RSI cycle is now complete!")
            logger.info("   The system can continuously improve itself through:")
            logger.info("   • Intelligent hypothesis generation")
            logger.info("   • Rigorous validation and testing")  
            logger.info("   • Safe deployment with human oversight")
            logger.info("   • Continuous monitoring and feedback")
            logger.info("")
            logger.info("🎯 O ciclo está fechado! The RSI loop is closed!")
            
            return {
                "cycle_completed": True,
                "phases_executed": 5,
                "system_status": status,
                "prediction_demo": prediction_result,
                "learning_demo": learning_result,
                "hypothesis_demo": hypothesis_result,
                "system_stats": stats
            }
            
        except Exception as e:
            logger.error("❌ RSI Cycle demonstration failed: {}", str(e))
            raise

def main():
    """Main demonstration function"""
    logger.info("🏛️ Hephaestus RSI System - Complete Demonstration")
    logger.info("=" * 80)
    
    # Initialize demo
    demo = HephaestusRSIDemo()
    
    try:
        # Run complete cycle demonstration
        result = demo.demonstrate_complete_cycle()
        
        logger.info("✅ Demonstration completed successfully!")
        logger.info("📋 Full results available in rsi_demo.log")
        
        return result
        
    except Exception as e:
        logger.error("❌ Demonstration failed: {}", str(e))
        logger.info("💡 Make sure the Hephaestus system is running on http://localhost:8000")
        logger.info("   Start it with: python -m src.main")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("🎉 RSI Demonstration completed successfully!")
        print("📊 Check rsi_demo.log for detailed results")
    else:
        print("❌ Demonstration failed - check logs for details")