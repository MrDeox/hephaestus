"""
RSI Hypothesis Testing Orchestrator.
Comprehensive end-to-end orchestration of hypothesis generation, validation, approval, execution, and simulation.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from loguru import logger

from .hypothesis_generator import (
    RSIHypothesisGenerator, RSIHypothesis, HypothesisGenerationConfig,
    HypothesisType, HypothesisPriority
)
from .hypothesis_validator import (
    RSIHypothesisValidator, HypothesisValidationResult, ValidationLevel
)
from .safety_verifier import (
    RSISafetyVerifier, ExecutionResult, SafetyConstraints, 
    create_development_safety_constraints, create_strict_safety_constraints
)
from .human_in_loop import (
    HumanInLoopManager, ReviewRequest, ReviewDecision, ReviewStatus
)
from .hypothesis_simulator import (
    RSIHypothesisSimulator, SimulationResult, SimulationConfig, SimulationEnvironment
)
from ..core.state import RSIStateManager
from ..validation.validators import RSIValidator
from ..safety.circuits import RSICircuitBreaker
from ..monitoring.audit_logger import AuditLogger


class OrchestrationPhase(str, Enum):
    GENERATION = "generation"
    VALIDATION = "validation"
    APPROVAL = "approval"
    EXECUTION = "execution"
    SIMULATION = "simulation"
    ASSESSMENT = "assessment"
    DEPLOYMENT = "deployment"


class HypothesisStatus(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXECUTION_FAILED = "execution_failed"
    SIMULATED = "simulated"
    SIMULATION_FAILED = "simulation_failed"
    RECOMMENDED = "recommended"
    NOT_RECOMMENDED = "not_recommended"
    DEPLOYED = "deployed"
    DEPLOYMENT_FAILED = "deployment_failed"


@dataclass
class HypothesisOrchestrationResult:
    """Complete orchestration result for RSI hypothesis"""
    hypothesis: RSIHypothesis
    status: HypothesisStatus
    current_phase: OrchestrationPhase
    
    # Results from each phase
    validation_result: Optional[HypothesisValidationResult] = None
    review_request: Optional[ReviewRequest] = None
    review_decision: Optional[ReviewDecision] = None
    execution_result: Optional[ExecutionResult] = None
    simulation_result: Optional[SimulationResult] = None
    
    # Orchestration metadata
    start_time: float = None
    end_time: Optional[float] = None
    total_duration_seconds: Optional[float] = None
    phase_durations: Dict[str, float] = None
    
    # Final assessment
    final_recommendation: Optional[str] = None
    confidence_score: Optional[float] = None
    deployment_ready: bool = False
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.phase_durations is None:
            self.phase_durations = {}


class RSIHypothesisOrchestrator:
    """
    Comprehensive orchestrator for RSI hypothesis testing pipeline.
    Coordinates all components from generation through deployment.
    """
    
    def __init__(self,
                 state_manager: Optional[RSIStateManager] = None,
                 validator: Optional[RSIValidator] = None,
                 circuit_breaker: Optional[RSICircuitBreaker] = None,
                 audit_logger: Optional[AuditLogger] = None,
                 environment: str = "development"):
        
        self.state_manager = state_manager
        self.validator = validator
        self.circuit_breaker = circuit_breaker
        self.audit_logger = audit_logger
        self.environment = environment
        
        # Initialize components
        self._initialize_components()
        
        # Orchestration tracking
        self.active_orchestrations: Dict[str, HypothesisOrchestrationResult] = {}
        self.completed_orchestrations: Dict[str, HypothesisOrchestrationResult] = {}
        self.orchestration_history: List[HypothesisOrchestrationResult] = []
        
        # Configuration
        self.auto_approve_safety_enhancements = True
        self.require_simulation_for_architecture_changes = True
        self.max_concurrent_orchestrations = 5
        
        logger.info("RSI Hypothesis Orchestrator initialized for {} environment", environment)
    
    def _initialize_components(self):
        """Initialize all orchestration components"""
        
        # Generator
        generation_config = HypothesisGenerationConfig(
            max_hypotheses_per_iteration=50,
            safety_threshold=0.8,
            computational_budget=600.0
        )
        self.generator = RSIHypothesisGenerator(
            config=generation_config,
            state_manager=self.state_manager,
            validator=self.validator,
            circuit_breaker=self.circuit_breaker
        )
        
        # Validator
        self.hypothesis_validator = RSIHypothesisValidator(
            validator=self.validator,
            circuit_breaker=self.circuit_breaker
        )
        
        # Safety verifier
        safety_constraints = (
            create_development_safety_constraints() 
            if self.environment == "development" 
            else create_strict_safety_constraints()
        )
        self.safety_verifier = RSISafetyVerifier(
            default_constraints=safety_constraints,
            circuit_breaker=self.circuit_breaker
        )
        
        # Human-in-the-loop manager
        self.human_loop_manager = HumanInLoopManager(
            audit_logger=self.audit_logger,
            auto_approve_safety_score=0.9
        )
        
        # Simulator
        simulation_config = SimulationConfig(
            environment=SimulationEnvironment.SANDBOX if self.environment == "development" else SimulationEnvironment.STAGING,
            simulation_duration_minutes=30,
            num_test_scenarios=50
        )
        self.simulator = RSIHypothesisSimulator(
            audit_logger=self.audit_logger,
            default_config=simulation_config
        )
    
    async def orchestrate_hypothesis_lifecycle(self,
                                             improvement_targets: Dict[str, float],
                                             context: Optional[Dict[str, Any]] = None,
                                             max_hypotheses: int = 10) -> List[HypothesisOrchestrationResult]:
        """
        Orchestrate complete hypothesis lifecycle from generation to deployment recommendation.
        
        Args:
            improvement_targets: Target improvements (e.g., {"accuracy": 0.05, "efficiency": 0.1})
            context: Additional context for hypothesis generation
            max_hypotheses: Maximum number of hypotheses to process
            
        Returns:
            List of orchestration results for all processed hypotheses
        """
        logger.info("Starting hypothesis lifecycle orchestration with targets: {}", improvement_targets)
        
        orchestration_results = []
        
        try:
            # Phase 1: Generate hypotheses
            logger.info("Phase 1: Generating hypotheses")
            hypotheses = await self.generator.generate_hypotheses_batch(
                improvement_targets, context or {}
            )
            
            # Limit number of hypotheses
            hypotheses = hypotheses[:max_hypotheses]
            logger.info("Generated {} hypotheses for orchestration", len(hypotheses))
            
            # Process each hypothesis through the pipeline
            semaphore = asyncio.Semaphore(self.max_concurrent_orchestrations)
            
            async def process_hypothesis(hypothesis: RSIHypothesis) -> HypothesisOrchestrationResult:
                async with semaphore:
                    return await self._orchestrate_single_hypothesis(hypothesis, context)
            
            # Process hypotheses concurrently
            tasks = [process_hypothesis(h) for h in hypotheses]
            orchestration_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in orchestration_results if isinstance(r, HypothesisOrchestrationResult)]
            
            # Sort by confidence score and recommendation
            valid_results.sort(key=lambda r: (
                r.confidence_score or 0,
                1 if r.deployment_ready else 0
            ), reverse=True)
            
            logger.info("Completed orchestration for {} hypotheses", len(valid_results))
            
            # Log orchestration summary
            if self.audit_logger:
                await self.audit_logger.log_event(
                    "hypothesis_orchestration_completed",
                    {
                        "total_hypotheses": len(hypotheses),
                        "successful_orchestrations": len(valid_results),
                        "deployment_ready": len([r for r in valid_results if r.deployment_ready]),
                        "average_confidence": sum(r.confidence_score or 0 for r in valid_results) / max(1, len(valid_results))
                    }
                )
            
            return valid_results
            
        except Exception as e:
            logger.error("Error in hypothesis lifecycle orchestration: {}", str(e))
            return []
    
    async def _orchestrate_single_hypothesis(self,
                                           hypothesis: RSIHypothesis,
                                           context: Optional[Dict[str, Any]]) -> HypothesisOrchestrationResult:
        """Orchestrate complete lifecycle for a single hypothesis"""
        
        result = HypothesisOrchestrationResult(
            hypothesis=hypothesis,
            status=HypothesisStatus.GENERATED,
            current_phase=OrchestrationPhase.GENERATION
        )
        
        self.active_orchestrations[hypothesis.hypothesis_id] = result
        
        try:
            # Phase 2: Validation
            phase_start = time.time()
            result.current_phase = OrchestrationPhase.VALIDATION
            
            logger.debug("Validating hypothesis {}", hypothesis.hypothesis_id)
            validation_result = await self.hypothesis_validator.validate_hypothesis_comprehensive(
                hypothesis, ValidationLevel.COMPREHENSIVE
            )
            result.validation_result = validation_result
            result.phase_durations[OrchestrationPhase.VALIDATION.value] = time.time() - phase_start
            
            if not validation_result.is_valid:
                result.status = HypothesisStatus.VALIDATION_FAILED
                logger.warning("Hypothesis {} failed validation", hypothesis.hypothesis_id)
                return result
            
            result.status = HypothesisStatus.VALIDATED
            
            # Phase 3: Approval (if needed)
            phase_start = time.time()
            result.current_phase = OrchestrationPhase.APPROVAL
            
            if validation_result.requires_human_review:
                logger.debug("Requesting approval for hypothesis {}", hypothesis.hypothesis_id)
                review_request = await self.human_loop_manager.request_approval(
                    hypothesis, validation_result, self.safety_verifier.default_constraints, context
                )
                result.review_request = review_request
                result.status = HypothesisStatus.PENDING_APPROVAL
                
                # Wait for approval (with timeout)
                review_status, review_decision = await self.human_loop_manager.wait_for_approval(
                    review_request.request_id, timeout_seconds=300  # 5 minutes for testing
                )
                result.review_decision = review_decision
                
                if review_status != ReviewStatus.APPROVED:
                    result.status = HypothesisStatus.REJECTED
                    logger.info("Hypothesis {} rejected during review", hypothesis.hypothesis_id)
                    return result
            
            result.status = HypothesisStatus.APPROVED
            result.phase_durations[OrchestrationPhase.APPROVAL.value] = time.time() - phase_start
            
            # Phase 4: Execution
            phase_start = time.time()
            result.current_phase = OrchestrationPhase.EXECUTION
            
            logger.debug("Executing hypothesis {}", hypothesis.hypothesis_id)
            execution_result = await self.safety_verifier.execute_hypothesis_safely(
                hypothesis, validation_result, context=context
            )
            result.execution_result = execution_result
            result.phase_durations[OrchestrationPhase.EXECUTION.value] = time.time() - phase_start
            
            if execution_result.status.value != "completed":
                result.status = HypothesisStatus.EXECUTION_FAILED
                logger.warning("Hypothesis {} execution failed", hypothesis.hypothesis_id)
                return result
            
            result.status = HypothesisStatus.EXECUTED
            
            # Phase 5: Simulation (for certain hypothesis types)
            if self._requires_simulation(hypothesis):
                phase_start = time.time()
                result.current_phase = OrchestrationPhase.SIMULATION
                
                logger.debug("Simulating hypothesis {}", hypothesis.hypothesis_id)
                simulation_result = await self.simulator.simulate_hypothesis(
                    hypothesis, validation_result, execution_result
                )
                result.simulation_result = simulation_result
                result.phase_durations[OrchestrationPhase.SIMULATION.value] = time.time() - phase_start
                
                if simulation_result.status.value != "completed":
                    result.status = HypothesisStatus.SIMULATION_FAILED
                    logger.warning("Hypothesis {} simulation failed", hypothesis.hypothesis_id)
                    return result
                
                result.status = HypothesisStatus.SIMULATED
            
            # Phase 6: Final assessment
            phase_start = time.time()
            result.current_phase = OrchestrationPhase.ASSESSMENT
            
            assessment = await self._generate_final_assessment(result)
            result.final_recommendation = assessment['recommendation']
            result.confidence_score = assessment['confidence_score']
            result.deployment_ready = assessment['deployment_ready']
            result.phase_durations[OrchestrationPhase.ASSESSMENT.value] = time.time() - phase_start
            
            if result.deployment_ready:
                result.status = HypothesisStatus.RECOMMENDED
            else:
                result.status = HypothesisStatus.NOT_RECOMMENDED
            
            logger.info("Hypothesis {} orchestration completed: {}", 
                       hypothesis.hypothesis_id, result.status.value)
            
        except Exception as e:
            logger.error("Error orchestrating hypothesis {}: {}", hypothesis.hypothesis_id, str(e))
            result.status = HypothesisStatus.EXECUTION_FAILED
            
        finally:
            # Finalize result
            result.end_time = time.time()
            result.total_duration_seconds = result.end_time - result.start_time
            
            # Move to completed
            if hypothesis.hypothesis_id in self.active_orchestrations:
                del self.active_orchestrations[hypothesis.hypothesis_id]
            
            self.completed_orchestrations[hypothesis.hypothesis_id] = result
            self.orchestration_history.append(result)
        
        return result
    
    def _requires_simulation(self, hypothesis: RSIHypothesis) -> bool:
        """Determine if hypothesis requires simulation testing"""
        
        # Always simulate architecture changes
        if (self.require_simulation_for_architecture_changes and 
            hypothesis.hypothesis_type == HypothesisType.ARCHITECTURE_CHANGE):
            return True
        
        # Simulate high-risk hypotheses
        if hypothesis.risk_level > 0.6:
            return True
        
        # Simulate hypotheses with significant expected impact
        if sum(hypothesis.expected_improvement.values()) > 0.05:
            return True
        
        # Always simulate in production environment
        if self.environment == "production":
            return True
        
        return False
    
    async def _generate_final_assessment(self, result: HypothesisOrchestrationResult) -> Dict[str, Any]:
        """Generate final assessment and deployment recommendation"""
        
        confidence_factors = []
        risk_factors = []
        
        # Validation confidence
        if result.validation_result:
            confidence_factors.append(result.validation_result.overall_score)
            if result.validation_result.safety_score < 0.7:
                risk_factors.append("Low safety score")
        
        # Execution success
        if result.execution_result:
            if result.execution_result.status.value == "completed":
                confidence_factors.append(0.8)
                if result.execution_result.resource_violations:
                    risk_factors.append("Resource violations during execution")
                if result.execution_result.security_issues:
                    risk_factors.append("Security issues detected")
            else:
                confidence_factors.append(0.2)
        
        # Simulation results
        if result.simulation_result:
            confidence_factors.append(result.simulation_result.confidence_score)
            if result.simulation_result.success_rate < 0.7:
                risk_factors.append("Low simulation success rate")
            if result.simulation_result.risk_assessment.get('risk_level') == 'high':
                risk_factors.append("High risk assessment from simulation")
        
        # Calculate overall confidence
        confidence_score = sum(confidence_factors) / max(1, len(confidence_factors))
        
        # Determine deployment readiness
        deployment_ready = (
            confidence_score > 0.7 and
            len(risk_factors) == 0 and
            result.status in [HypothesisStatus.EXECUTED, HypothesisStatus.SIMULATED]
        )
        
        # Generate recommendation
        if deployment_ready:
            recommendation = "DEPLOY: High confidence, minimal risk factors"
        elif confidence_score > 0.6 and len(risk_factors) <= 1:
            recommendation = "CONDITIONAL DEPLOY: Good confidence with manageable risks"
        elif confidence_score > 0.5:
            recommendation = "INVESTIGATE: Moderate confidence, requires risk mitigation"
        else:
            recommendation = "DO NOT DEPLOY: Low confidence or significant risks"
        
        return {
            'confidence_score': confidence_score,
            'deployment_ready': deployment_ready,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'confidence_factors_count': len(confidence_factors)
        }
    
    async def get_orchestration_status(self, hypothesis_id: str) -> Optional[HypothesisOrchestrationResult]:
        """Get current orchestration status for a hypothesis"""
        
        if hypothesis_id in self.active_orchestrations:
            return self.active_orchestrations[hypothesis_id]
        elif hypothesis_id in self.completed_orchestrations:
            return self.completed_orchestrations[hypothesis_id]
        else:
            return None
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics"""
        
        total_orchestrations = len(self.completed_orchestrations)
        active_count = len(self.active_orchestrations)
        
        if total_orchestrations == 0:
            return {"status": "no_orchestrations_completed"}
        
        # Status distribution
        status_distribution = {}
        deployment_ready_count = 0
        
        for result in self.completed_orchestrations.values():
            status = result.status.value
            status_distribution[status] = status_distribution.get(status, 0) + 1
            if result.deployment_ready:
                deployment_ready_count += 1
        
        # Phase duration analysis
        phase_durations = {}
        for result in self.completed_orchestrations.values():
            for phase, duration in result.phase_durations.items():
                if phase not in phase_durations:
                    phase_durations[phase] = []
                phase_durations[phase].append(duration)
        
        avg_phase_durations = {
            phase: sum(durations) / len(durations)
            for phase, durations in phase_durations.items()
        }
        
        # Success metrics
        confidence_scores = [
            result.confidence_score for result in self.completed_orchestrations.values()
            if result.confidence_score is not None
        ]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        deployment_rate = deployment_ready_count / total_orchestrations
        
        return {
            'total_orchestrations': total_orchestrations,
            'active_orchestrations': active_count,
            'deployment_ready_count': deployment_ready_count,
            'deployment_rate': deployment_rate,
            'avg_confidence_score': avg_confidence,
            'status_distribution': status_distribution,
            'avg_phase_durations_seconds': avg_phase_durations,
            'avg_total_duration_seconds': sum(
                result.total_duration_seconds for result in self.completed_orchestrations.values()
                if result.total_duration_seconds
            ) / max(1, total_orchestrations),
            'environment': self.environment
        }
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        
        logger.info("Cleaning up RSI Hypothesis Orchestrator")
        
        # Cleanup individual components
        if hasattr(self.generator, 'cleanup'):
            await self.generator.cleanup()
        
        if hasattr(self.safety_verifier, 'cleanup'):
            await self.safety_verifier.cleanup()
        
        # Cancel any active orchestrations
        for hypothesis_id in list(self.active_orchestrations.keys()):
            logger.warning("Cancelling active orchestration: {}", hypothesis_id)
        
        logger.info("RSI Hypothesis Orchestrator cleanup completed")