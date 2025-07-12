"""
Human-in-the-Loop Approval System for RSI Hypothesis Testing.
Implements comprehensive approval workflow with review tracking and decision logging.
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from .hypothesis_generator import RSIHypothesis, HypothesisPriority
from .hypothesis_validator import HypothesisValidationResult
from .safety_verifier import ExecutionResult, SafetyConstraints
from ..monitoring.audit_logger import AuditLogger


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MODIFICATION = "needs_modification"
    ESCALATED = "escalated"


class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ReviewRequest:
    """Human review request for RSI hypothesis"""
    request_id: str
    hypothesis: RSIHypothesis
    validation_result: HypothesisValidationResult
    safety_constraints: SafetyConstraints
    review_priority: ReviewPriority
    reviewer_notes: str
    context: Dict[str, Any]
    
    # Metadata
    created_timestamp: float
    requested_by: str
    review_deadline: Optional[float] = None
    
    # Review process
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_reviewer: Optional[str] = None
    review_start_time: Optional[float] = None
    review_completion_time: Optional[float] = None
    
    # Decision
    approved: Optional[bool] = None
    reviewer_comments: Optional[str] = None
    modification_requests: List[str] = None
    escalation_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.modification_requests is None:
            self.modification_requests = []


@dataclass
class ReviewDecision:
    """Final review decision with full audit trail"""
    request_id: str
    hypothesis_id: str
    reviewer_id: str
    decision: ReviewStatus
    confidence_level: float
    reasoning: str
    modification_suggestions: List[str]
    risk_assessment: Dict[str, Any]
    approval_conditions: List[str]
    timestamp: float
    
    def __post_init__(self):
        if self.modification_suggestions is None:
            self.modification_suggestions = []
        if self.approval_conditions is None:
            self.approval_conditions = []


class HumanInLoopManager:
    """
    Comprehensive human-in-the-loop approval system for RSI hypothesis testing.
    Manages review workflows, approval processes, and decision tracking.
    """
    
    def __init__(self, 
                 audit_logger: Optional[AuditLogger] = None,
                 review_timeout_hours: int = 24,
                 approval_threshold: float = 0.7,
                 auto_approve_safety_score: float = 0.9):
        
        self.audit_logger = audit_logger
        self.review_timeout_hours = review_timeout_hours
        self.approval_threshold = approval_threshold
        self.auto_approve_safety_score = auto_approve_safety_score
        
        # Review tracking
        self.pending_reviews: Dict[str, ReviewRequest] = {}
        self.completed_reviews: Dict[str, ReviewRequest] = {}
        self.review_history: List[ReviewDecision] = []
        
        # Review queues by priority
        self.review_queues = {
            ReviewPriority.URGENT: [],
            ReviewPriority.HIGH: [],
            ReviewPriority.MEDIUM: [],
            ReviewPriority.LOW: []
        }
        
        # Reviewer management
        self.available_reviewers: List[str] = ["human_reviewer_1", "safety_expert", "ml_engineer"]
        self.reviewer_workload: Dict[str, int] = {r: 0 for r in self.available_reviewers}
        
        # Auto-approval rules
        self.auto_approval_rules = {
            "safety_enhancement": True,  # Always auto-approve safety improvements
            "low_risk_hyperparameter": True,  # Auto-approve low-risk hyperparameter changes
            "ensemble_strategy": False,  # Always require human review
            "algorithm_modification": False  # Always require human review
        }
        
        logger.info("Human-in-the-Loop Manager initialized with {} reviewers", 
                   len(self.available_reviewers))
    
    async def request_approval(self, 
                             hypothesis: RSIHypothesis,
                             validation_result: HypothesisValidationResult,
                             safety_constraints: SafetyConstraints,
                             context: Optional[Dict[str, Any]] = None) -> ReviewRequest:
        """
        Request human approval for hypothesis execution.
        
        Args:
            hypothesis: The hypothesis requiring approval
            validation_result: Validation results
            safety_constraints: Safety constraints for execution
            context: Additional context for review
            
        Returns:
            Review request with tracking information
        """
        request_id = f"review_{uuid.uuid4().hex[:8]}"
        
        # Determine review priority
        review_priority = self._determine_review_priority(hypothesis, validation_result)
        
        # Check for auto-approval eligibility
        if await self._check_auto_approval_eligibility(hypothesis, validation_result):
            logger.info("Hypothesis {} auto-approved based on safety criteria", 
                       hypothesis.hypothesis_id)
            
            # Create pre-approved review request
            review_request = ReviewRequest(
                request_id=request_id,
                hypothesis=hypothesis,
                validation_result=validation_result,
                safety_constraints=safety_constraints,
                review_priority=review_priority,
                reviewer_notes="Auto-approved based on safety criteria",
                context=context or {},
                created_timestamp=time.time(),
                requested_by="system",
                status=ReviewStatus.APPROVED,
                approved=True,
                reviewer_comments="Automatically approved - meets safety criteria",
                review_completion_time=time.time()
            )
            
            self.completed_reviews[request_id] = review_request
            
            # Log approval
            if self.audit_logger:
                await self.audit_logger.log_event(
                    "hypothesis_auto_approved",
                    {
                        "request_id": request_id,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "safety_score": validation_result.safety_score,
                        "auto_approval_criteria": "safety_score_threshold"
                    }
                )
            
            return review_request
        
        # Create review request
        review_request = ReviewRequest(
            request_id=request_id,
            hypothesis=hypothesis,
            validation_result=validation_result,
            safety_constraints=safety_constraints,
            review_priority=review_priority,
            reviewer_notes=self._generate_reviewer_notes(hypothesis, validation_result),
            context=context or {},
            created_timestamp=time.time(),
            requested_by="system",
            review_deadline=time.time() + (self.review_timeout_hours * 3600)
        )
        
        # Add to appropriate queue
        self.pending_reviews[request_id] = review_request
        self.review_queues[review_priority].append(request_id)
        
        # Assign reviewer
        assigned_reviewer = self._assign_reviewer(review_priority)
        if assigned_reviewer:
            review_request.assigned_reviewer = assigned_reviewer
            self.reviewer_workload[assigned_reviewer] += 1
            
            logger.info("Review request {} assigned to reviewer {}", 
                       request_id, assigned_reviewer)
        
        # Log review request
        if self.audit_logger:
            await self.audit_logger.log_event(
                "hypothesis_review_requested",
                {
                    "request_id": request_id,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_type": hypothesis.hypothesis_type.value,
                    "priority": review_priority.value,
                    "assigned_reviewer": assigned_reviewer,
                    "safety_score": validation_result.safety_score,
                    "requires_approval_reason": validation_result.requires_human_review
                }
            )
        
        logger.info("Review requested for hypothesis {} with priority {}", 
                   hypothesis.hypothesis_id, review_priority.value)
        
        return review_request
    
    async def submit_review_decision(self, 
                                   request_id: str,
                                   reviewer_id: str,
                                   decision: ReviewStatus,
                                   reasoning: str,
                                   confidence_level: float = 0.8,
                                   modification_suggestions: Optional[List[str]] = None,
                                   approval_conditions: Optional[List[str]] = None) -> ReviewDecision:
        """
        Submit a review decision for a pending hypothesis.
        
        Args:
            request_id: ID of the review request
            reviewer_id: ID of the reviewing expert
            decision: Review decision (approved/rejected/needs_modification)
            reasoning: Detailed reasoning for the decision
            confidence_level: Reviewer's confidence in the decision (0.0-1.0)
            modification_suggestions: Suggested modifications if needed
            approval_conditions: Conditions that must be met for approval
            
        Returns:
            Complete review decision with audit trail
        """
        if request_id not in self.pending_reviews:
            raise ValueError(f"Review request {request_id} not found or already completed")
        
        review_request = self.pending_reviews[request_id]
        
        # Validate reviewer authority
        if review_request.assigned_reviewer and review_request.assigned_reviewer != reviewer_id:
            logger.warning("Reviewer {} attempting to review request assigned to {}", 
                          reviewer_id, review_request.assigned_reviewer)
        
        # Create review decision
        review_decision = ReviewDecision(
            request_id=request_id,
            hypothesis_id=review_request.hypothesis.hypothesis_id,
            reviewer_id=reviewer_id,
            decision=decision,
            confidence_level=confidence_level,
            reasoning=reasoning,
            modification_suggestions=modification_suggestions or [],
            risk_assessment=self._assess_decision_risk(review_request, decision),
            approval_conditions=approval_conditions or [],
            timestamp=time.time()
        )
        
        # Update review request
        review_request.status = decision
        review_request.approved = decision == ReviewStatus.APPROVED
        review_request.reviewer_comments = reasoning
        review_request.modification_requests = modification_suggestions or []
        review_request.review_completion_time = time.time()
        
        if decision == ReviewStatus.ESCALATED:
            review_request.escalation_reason = reasoning
            # Keep in pending for escalation handling
        else:
            # Move to completed reviews
            self.completed_reviews[request_id] = review_request
            del self.pending_reviews[request_id]
            
            # Remove from queue
            for priority_queue in self.review_queues.values():
                if request_id in priority_queue:
                    priority_queue.remove(request_id)
            
            # Update reviewer workload
            if review_request.assigned_reviewer in self.reviewer_workload:
                self.reviewer_workload[review_request.assigned_reviewer] -= 1
        
        # Add to review history
        self.review_history.append(review_decision)
        
        # Log decision
        if self.audit_logger:
            await self.audit_logger.log_event(
                "hypothesis_review_decision",
                {
                    "request_id": request_id,
                    "hypothesis_id": review_request.hypothesis.hypothesis_id,
                    "reviewer_id": reviewer_id,
                    "decision": decision.value,
                    "confidence_level": confidence_level,
                    "reasoning": reasoning,
                    "review_duration_minutes": (
                        (review_request.review_completion_time - review_request.created_timestamp) / 60
                        if review_request.review_completion_time else None
                    )
                }
            )
        
        logger.info("Review decision submitted: {} for hypothesis {} by reviewer {}", 
                   decision.value, review_request.hypothesis.hypothesis_id, reviewer_id)
        
        return review_decision
    
    async def get_pending_reviews(self, 
                                reviewer_id: Optional[str] = None,
                                priority: Optional[ReviewPriority] = None) -> List[ReviewRequest]:
        """Get pending reviews, optionally filtered by reviewer or priority"""
        
        pending = list(self.pending_reviews.values())
        
        if reviewer_id:
            pending = [r for r in pending if r.assigned_reviewer == reviewer_id]
        
        if priority:
            pending = [r for r in pending if r.review_priority == priority]
        
        # Sort by priority and creation time
        priority_order = {
            ReviewPriority.URGENT: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3
        }
        
        pending.sort(key=lambda r: (priority_order[r.review_priority], r.created_timestamp))
        
        return pending
    
    async def check_approval_status(self, request_id: str) -> Optional[ReviewStatus]:
        """Check the current approval status of a review request"""
        
        if request_id in self.pending_reviews:
            return self.pending_reviews[request_id].status
        elif request_id in self.completed_reviews:
            return self.completed_reviews[request_id].status
        else:
            return None
    
    async def wait_for_approval(self, 
                              request_id: str, 
                              timeout_seconds: Optional[int] = None) -> Tuple[ReviewStatus, Optional[ReviewDecision]]:
        """
        Wait for approval decision with optional timeout.
        
        Args:
            request_id: ID of the review request
            timeout_seconds: Maximum time to wait (defaults to review deadline)
            
        Returns:
            Tuple of (final_status, review_decision)
        """
        if request_id not in self.pending_reviews and request_id not in self.completed_reviews:
            raise ValueError(f"Review request {request_id} not found")
        
        # If already completed, return immediately
        if request_id in self.completed_reviews:
            review_request = self.completed_reviews[request_id]
            decision = next((d for d in self.review_history if d.request_id == request_id), None)
            return review_request.status, decision
        
        # Calculate timeout
        review_request = self.pending_reviews[request_id]
        if timeout_seconds is None:
            timeout_seconds = int(review_request.review_deadline - time.time()) if review_request.review_deadline else 3600
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check if completed
            if request_id in self.completed_reviews:
                review_request = self.completed_reviews[request_id]
                decision = next((d for d in self.review_history if d.request_id == request_id), None)
                return review_request.status, decision
            
            # Check for timeout
            if review_request.review_deadline and time.time() > review_request.review_deadline:
                logger.warning("Review request {} timed out", request_id)
                
                # Auto-reject on timeout for safety
                await self.submit_review_decision(
                    request_id=request_id,
                    reviewer_id="system",
                    decision=ReviewStatus.REJECTED,
                    reasoning="Review timed out - automatically rejected for safety",
                    confidence_level=1.0
                )
                
                return ReviewStatus.REJECTED, None
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        # Timeout without completion
        logger.warning("Timeout waiting for approval of request {}", request_id)
        return ReviewStatus.PENDING, None
    
    def _determine_review_priority(self, 
                                 hypothesis: RSIHypothesis, 
                                 validation_result: HypothesisValidationResult) -> ReviewPriority:
        """Determine review priority based on hypothesis characteristics"""
        
        # Critical safety issues get urgent priority
        if validation_result.safety_score < 0.5:
            return ReviewPriority.URGENT
        
        # High-risk or high-impact hypotheses get high priority
        if (hypothesis.risk_level > 0.7 or 
            hypothesis.priority == HypothesisPriority.CRITICAL or
            sum(hypothesis.expected_improvement.values()) > 0.1):
            return ReviewPriority.HIGH
        
        # Safety enhancements get medium priority
        if hypothesis.hypothesis_type.value == "safety_enhancement":
            return ReviewPriority.MEDIUM
        
        # Low-risk changes get low priority
        if hypothesis.risk_level < 0.3 and validation_result.safety_score > 0.8:
            return ReviewPriority.LOW
        
        # Default to medium priority
        return ReviewPriority.MEDIUM
    
    async def _check_auto_approval_eligibility(self, 
                                             hypothesis: RSIHypothesis, 
                                             validation_result: HypothesisValidationResult) -> bool:
        """Check if hypothesis is eligible for automatic approval"""
        
        # Safety score threshold
        if validation_result.safety_score >= self.auto_approve_safety_score:
            return True
        
        # Check auto-approval rules
        hypothesis_type = hypothesis.hypothesis_type.value
        
        if hypothesis_type == "safety_enhancement":
            return self.auto_approval_rules.get("safety_enhancement", False)
        
        if (hypothesis_type == "hyperparameter_optimization" and 
            hypothesis.risk_level < 0.3 and 
            validation_result.safety_score > 0.8):
            return self.auto_approval_rules.get("low_risk_hyperparameter", False)
        
        return False
    
    def _generate_reviewer_notes(self, 
                               hypothesis: RSIHypothesis, 
                               validation_result: HypothesisValidationResult) -> str:
        """Generate helpful notes for human reviewers"""
        
        notes = []
        
        notes.append(f"Hypothesis Type: {hypothesis.hypothesis_type.value}")
        notes.append(f"Risk Level: {hypothesis.risk_level:.2f}")
        notes.append(f"Expected Improvement: {hypothesis.expected_improvement}")
        notes.append(f"Safety Score: {validation_result.safety_score:.2f}")
        notes.append(f"Performance Score: {validation_result.performance_score:.2f}")
        
        if validation_result.safety_score < 0.7:
            notes.append("⚠️ LOW SAFETY SCORE - Requires careful review")
        
        if hypothesis.risk_level > 0.7:
            notes.append("⚠️ HIGH RISK HYPOTHESIS - Consider additional safeguards")
        
        if validation_result.security_score < 0.8:
            notes.append("⚠️ SECURITY CONCERNS - Review security implications")
        
        return "\n".join(notes)
    
    def _assign_reviewer(self, priority: ReviewPriority) -> Optional[str]:
        """Assign an available reviewer based on priority and workload"""
        
        # Sort reviewers by current workload
        available = sorted(self.available_reviewers, 
                         key=lambda r: self.reviewer_workload[r])
        
        # For urgent/high priority, assign to least loaded reviewer
        if priority in [ReviewPriority.URGENT, ReviewPriority.HIGH]:
            return available[0] if available else None
        
        # For medium/low priority, assign to any available reviewer
        for reviewer in available:
            if self.reviewer_workload[reviewer] < 5:  # Max 5 concurrent reviews
                return reviewer
        
        return None
    
    def _assess_decision_risk(self, 
                            review_request: ReviewRequest, 
                            decision: ReviewStatus) -> Dict[str, Any]:
        """Assess the risk level of a review decision"""
        
        hypothesis = review_request.hypothesis
        validation_result = review_request.validation_result
        
        risk_factors = []
        risk_score = 0.0
        
        if decision == ReviewStatus.APPROVED:
            if validation_result.safety_score < 0.7:
                risk_factors.append("Low safety score approved")
                risk_score += 0.3
            
            if hypothesis.risk_level > 0.7:
                risk_factors.append("High-risk hypothesis approved")
                risk_score += 0.2
            
            if validation_result.security_score < 0.8:
                risk_factors.append("Security concerns approved")
                risk_score += 0.2
        
        elif decision == ReviewStatus.REJECTED:
            if validation_result.safety_score > 0.9:
                risk_factors.append("High safety score rejected")
                risk_score += 0.1
        
        return {
            "risk_score": min(1.0, risk_score),
            "risk_factors": risk_factors,
            "decision_confidence": "high" if risk_score < 0.2 else "medium" if risk_score < 0.5 else "low"
        }
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review system statistics"""
        
        total_requests = len(self.completed_reviews) + len(self.pending_reviews)
        completed_count = len(self.completed_reviews)
        pending_count = len(self.pending_reviews)
        
        if completed_count == 0:
            return {"status": "no_reviews_completed"}
        
        # Decision distribution
        decision_distribution = {}
        approval_rate = 0
        
        for review in self.completed_reviews.values():
            status = review.status.value
            decision_distribution[status] = decision_distribution.get(status, 0) + 1
            if review.approved:
                approval_rate += 1
        
        approval_rate = approval_rate / completed_count if completed_count > 0 else 0
        
        # Review times
        review_times = []
        for review in self.completed_reviews.values():
            if review.review_completion_time and review.created_timestamp:
                review_times.append(review.review_completion_time - review.created_timestamp)
        
        avg_review_time = sum(review_times) / len(review_times) if review_times else 0
        
        return {
            "total_requests": total_requests,
            "completed_reviews": completed_count,
            "pending_reviews": pending_count,
            "approval_rate": approval_rate,
            "decision_distribution": decision_distribution,
            "avg_review_time_hours": avg_review_time / 3600,
            "reviewer_workload": self.reviewer_workload.copy(),
            "queue_lengths": {p.value: len(q) for p, q in self.review_queues.items()},
            "auto_approval_rate": len([r for r in self.completed_reviews.values() 
                                     if r.reviewer_comments and "auto" in r.reviewer_comments.lower()]) / max(1, completed_count)
        }