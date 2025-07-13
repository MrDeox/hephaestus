"""
Memory Consolidation System for RSI AI.
Implements intelligent routing and consolidation of information between memory systems.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory systems."""
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class ConsolidationRule:
    """Rule for memory consolidation."""
    name: str
    source_type: MemoryType
    target_type: MemoryType
    condition: str
    priority: float
    action: str
    parameters: Dict[str, Any]


class MemoryConsolidation:
    """
    Memory Consolidation System for RSI AI.
    
    Manages the flow of information between different memory systems:
    - Routes information to appropriate memory systems
    - Consolidates working memory to long-term storage
    - Manages memory lifecycle and cleanup
    - Implements sleep-like consolidation processes
    """
    
    def __init__(self, working_memory, semantic_memory, episodic_memory, 
                 procedural_memory, consolidation_threshold: float = 0.8):
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.consolidation_threshold = consolidation_threshold
        
        # Consolidation rules
        self.consolidation_rules = self._initialize_consolidation_rules()
        
        # Statistics
        self.stats = {
            'total_consolidations': 0,
            'working_to_semantic': 0,
            'working_to_episodic': 0,
            'working_to_procedural': 0,
            'semantic_to_episodic': 0,
            'consolidation_failures': 0,
            'last_consolidation': None
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Memory Consolidation system initialized")
    
    def _initialize_consolidation_rules(self) -> List[ConsolidationRule]:
        """Initialize consolidation rules."""
        return [
            ConsolidationRule(
                name="concept_to_semantic",
                source_type=MemoryType.WORKING,
                target_type=MemoryType.SEMANTIC,
                condition="has_concept_structure",
                priority=0.9,
                action="store_concept",
                parameters={"min_confidence": 0.7}
            ),
            ConsolidationRule(
                name="experience_to_episodic",
                source_type=MemoryType.WORKING,
                target_type=MemoryType.EPISODIC,
                condition="has_temporal_sequence",
                priority=0.8,
                action="store_episode",
                parameters={"min_importance": 0.5}
            ),
            ConsolidationRule(
                name="skill_to_procedural",
                source_type=MemoryType.WORKING,
                target_type=MemoryType.PROCEDURAL,
                condition="has_skill_structure",
                priority=0.7,
                action="store_skill",
                parameters={"min_success_rate": 0.6}
            ),
            ConsolidationRule(
                name="semantic_to_episodic",
                source_type=MemoryType.SEMANTIC,
                target_type=MemoryType.EPISODIC,
                condition="has_usage_context",
                priority=0.6,
                action="create_usage_episode",
                parameters={"min_usage_count": 5}
            ),
            ConsolidationRule(
                name="high_priority_immediate",
                source_type=MemoryType.WORKING,
                target_type=MemoryType.SEMANTIC,
                condition="high_priority",
                priority=1.0,
                action="immediate_consolidation",
                parameters={"priority_threshold": 0.9}
            )
        ]
    
    async def route_information(self, information: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route information to appropriate memory systems.
        
        Args:
            information: Information to route
            
        Returns:
            Routing results
        """
        try:
            with self.lock:
                routing_results = {
                    'routed_to': [],
                    'consolidation_actions': [],
                    'success': True
                }
                
                # Analyze information characteristics
                characteristics = self._analyze_information_characteristics(information)
                
                # Apply consolidation rules
                for rule in sorted(self.consolidation_rules, key=lambda r: r.priority, reverse=True):
                    if self._evaluate_consolidation_condition(information, characteristics, rule):
                        action_result = await self._execute_consolidation_action(information, rule)
                        
                        if action_result['success']:
                            routing_results['routed_to'].append(rule.target_type.value)
                            routing_results['consolidation_actions'].append({
                                'rule': rule.name,
                                'target': rule.target_type.value,
                                'result': action_result
                            })
                        else:
                            self.stats['consolidation_failures'] += 1
                
                return routing_results
                
        except Exception as e:
            logger.error(f"Failed to route information: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_information_characteristics(self, information: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze information to determine its characteristics."""
        characteristics = {
            'has_concept_structure': False,
            'has_temporal_sequence': False,
            'has_skill_structure': False,
            'has_usage_context': False,
            'high_priority': False,
            'complexity_score': 0.0,
            'temporal_markers': 0,
            'semantic_markers': 0,
            'procedural_markers': 0
        }
        
        try:
            # Check for concept structure
            if any(key in information for key in ['concept', 'knowledge', 'fact', 'definition']):
                characteristics['has_concept_structure'] = True
                characteristics['semantic_markers'] += 1
            
            # Check for temporal sequence
            if any(key in information for key in ['timestamp', 'sequence', 'episode', 'event']):
                characteristics['has_temporal_sequence'] = True
                characteristics['temporal_markers'] += 1
            
            # Check for skill structure
            if any(key in information for key in ['skill', 'action', 'procedure', 'method']):
                characteristics['has_skill_structure'] = True
                characteristics['procedural_markers'] += 1
            
            # Check for usage context
            if any(key in information for key in ['usage', 'context', 'application']):
                characteristics['has_usage_context'] = True
            
            # Check for high priority
            priority = information.get('priority', 0.0)
            if priority > 0.8:
                characteristics['high_priority'] = True
            
            # Calculate complexity score
            complexity_factors = [
                len(str(information)),  # Size
                len(information.keys()) if isinstance(information, dict) else 1,  # Structure
                characteristics['semantic_markers'],
                characteristics['temporal_markers'],
                characteristics['procedural_markers']
            ]
            
            characteristics['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Failed to analyze information characteristics: {e}")
            return characteristics
    
    def _evaluate_consolidation_condition(self, information: Dict[str, Any], 
                                        characteristics: Dict[str, Any], 
                                        rule: ConsolidationRule) -> bool:
        """Evaluate if consolidation condition is met."""
        try:
            condition = rule.condition
            
            if condition == "has_concept_structure":
                return characteristics['has_concept_structure']
            elif condition == "has_temporal_sequence":
                return characteristics['has_temporal_sequence']
            elif condition == "has_skill_structure":
                return characteristics['has_skill_structure']
            elif condition == "has_usage_context":
                return characteristics['has_usage_context']
            elif condition == "high_priority":
                return characteristics['high_priority']
            else:
                # Custom condition evaluation
                return self._evaluate_custom_condition(information, characteristics, condition)
                
        except Exception as e:
            logger.error(f"Failed to evaluate consolidation condition: {e}")
            return False
    
    def _evaluate_custom_condition(self, information: Dict[str, Any], 
                                 characteristics: Dict[str, Any], 
                                 condition: str) -> bool:
        """Evaluate custom consolidation conditions."""
        try:
            # This would implement more sophisticated condition evaluation
            # For now, return False for unknown conditions
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate custom condition: {e}")
            return False
    
    async def _execute_consolidation_action(self, information: Dict[str, Any], 
                                          rule: ConsolidationRule) -> Dict[str, Any]:
        """Execute consolidation action."""
        try:
            action = rule.action
            parameters = rule.parameters
            
            if action == "store_concept":
                return await self._store_concept(information, parameters)
            elif action == "store_episode":
                return await self._store_episode(information, parameters)
            elif action == "store_skill":
                return await self._store_skill(information, parameters)
            elif action == "create_usage_episode":
                return await self._create_usage_episode(information, parameters)
            elif action == "immediate_consolidation":
                return await self._immediate_consolidation(information, parameters)
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Failed to execute consolidation action: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _store_concept(self, information: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Store information as a concept in semantic memory."""
        try:
            min_confidence = parameters.get('min_confidence', 0.7)
            confidence = information.get('confidence', 1.0)
            
            if confidence < min_confidence:
                return {'success': False, 'reason': 'Confidence below threshold'}
            
            # Extract concept data
            concept_data = {
                'name': information.get('name', information.get('concept', 'Unknown')),
                'description': information.get('description', ''),
                'type': information.get('type', 'concept'),
                'confidence': confidence,
                'source': 'memory_consolidation',
                'consolidated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add any additional properties
            for key, value in information.items():
                if key not in concept_data:
                    concept_data[key] = value
            
            concept_id = await self.semantic_memory.store_concept(concept_data)
            
            self.stats['working_to_semantic'] += 1
            self.stats['total_consolidations'] += 1
            
            return {'success': True, 'concept_id': concept_id}
            
        except Exception as e:
            logger.error(f"Failed to store concept: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _store_episode(self, information: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Store information as an episode in episodic memory."""
        try:
            min_importance = parameters.get('min_importance', 0.5)
            importance = information.get('importance', 1.0)
            
            if importance < min_importance:
                return {'success': False, 'reason': 'Importance below threshold'}
            
            # Extract episode data
            episode_data = {
                'content': information,
                'tags': information.get('tags', []),
                'context': information.get('context', {}),
                'importance': importance,
                'emotions': information.get('emotions', {}),
                'source': 'memory_consolidation'
            }
            
            episode_id = await self.episodic_memory.record_episode(
                episode_data['content'],
                episode_data['tags'],
                episode_data['context'],
                episode_data['importance'],
                episode_data['emotions']
            )
            
            self.stats['working_to_episodic'] += 1
            self.stats['total_consolidations'] += 1
            
            return {'success': True, 'episode_id': episode_id}
            
        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _store_skill(self, information: Dict[str, Any], 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Store information as a skill in procedural memory."""
        try:
            min_success_rate = parameters.get('min_success_rate', 0.6)
            success_rate = information.get('success_rate', 1.0)
            
            if success_rate < min_success_rate:
                return {'success': False, 'reason': 'Success rate below threshold'}
            
            # Extract skill data
            skill_data = {
                'name': information.get('name', information.get('skill', 'Unknown')),
                'description': information.get('description', ''),
                'skill_type': information.get('skill_type', 'action'),
                'parameters': information.get('parameters', {}),
                'implementation': information.get('implementation', {}),
                'preconditions': information.get('preconditions', []),
                'postconditions': information.get('postconditions', []),
                'tags': information.get('tags', []),
                'complexity': information.get('complexity', 1.0),
                'dependencies': information.get('dependencies', []),
                'source': 'memory_consolidation'
            }
            
            skill_id = await self.procedural_memory.store_skill(skill_data)
            
            self.stats['working_to_procedural'] += 1
            self.stats['total_consolidations'] += 1
            
            return {'success': True, 'skill_id': skill_id}
            
        except Exception as e:
            logger.error(f"Failed to store skill: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _create_usage_episode(self, information: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create usage episode from semantic memory."""
        try:
            min_usage_count = parameters.get('min_usage_count', 5)
            usage_count = information.get('usage_count', 0)
            
            if usage_count < min_usage_count:
                return {'success': False, 'reason': 'Usage count below threshold'}
            
            # Create episode about concept usage
            episode_data = {
                'content': {
                    'type': 'concept_usage',
                    'concept_id': information.get('id', ''),
                    'usage_count': usage_count,
                    'usage_context': information.get('usage_context', {}),
                    'performance_metrics': information.get('performance_metrics', {})
                },
                'tags': ['concept_usage', 'semantic_to_episodic'],
                'context': information.get('context', {}),
                'importance': min(1.0, usage_count / 10.0),
                'emotions': {},
                'source': 'memory_consolidation'
            }
            
            episode_id = await self.episodic_memory.record_episode(
                episode_data['content'],
                episode_data['tags'],
                episode_data['context'],
                episode_data['importance'],
                episode_data['emotions']
            )
            
            self.stats['semantic_to_episodic'] += 1
            self.stats['total_consolidations'] += 1
            
            return {'success': True, 'episode_id': episode_id}
            
        except Exception as e:
            logger.error(f"Failed to create usage episode: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _immediate_consolidation(self, information: Dict[str, Any], 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Immediate consolidation for high-priority information."""
        try:
            priority_threshold = parameters.get('priority_threshold', 0.9)
            priority = information.get('priority', 0.0)
            
            if priority < priority_threshold:
                return {'success': False, 'reason': 'Priority below threshold'}
            
            # Store in multiple memory systems for immediate availability
            results = []
            
            # Store in semantic memory
            concept_result = await self._store_concept(information, {'min_confidence': 0.5})
            if concept_result['success']:
                results.append(concept_result)
            
            # Store in episodic memory
            episode_result = await self._store_episode(information, {'min_importance': 0.3})
            if episode_result['success']:
                results.append(episode_result)
            
            return {'success': True, 'consolidation_results': results}
            
        except Exception as e:
            logger.error(f"Failed immediate consolidation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def consolidate_batch(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Consolidate a batch of items from working memory.
        
        Args:
            batch_size: Number of items to consolidate
            
        Returns:
            Consolidation results
        """
        try:
            with self.lock:
                # Get items from working memory
                working_items = await self.working_memory.retrieve({'limit': batch_size})
                
                consolidation_results = {
                    'total_items': len(working_items),
                    'successful_consolidations': 0,
                    'failed_consolidations': 0,
                    'consolidation_details': []
                }
                
                for item in working_items:
                    try:
                        # Route item to appropriate memory systems
                        routing_result = await self.route_information(item['content'])
                        
                        if routing_result['success']:
                            consolidation_results['successful_consolidations'] += 1
                            
                            # Remove from working memory if successfully consolidated
                            if routing_result['routed_to']:
                                await self.working_memory.remove(item['id'])
                        else:
                            consolidation_results['failed_consolidations'] += 1
                        
                        consolidation_results['consolidation_details'].append({
                            'item_id': item['id'],
                            'routing_result': routing_result
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to consolidate item {item.get('id', 'unknown')}: {e}")
                        consolidation_results['failed_consolidations'] += 1
                
                self.stats['last_consolidation'] = datetime.now(timezone.utc).isoformat()
                
                return consolidation_results
                
        except Exception as e:
            logger.error(f"Failed batch consolidation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def sleep_consolidation(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """
        Perform sleep-like consolidation process.
        
        Args:
            duration_minutes: Duration of consolidation process
            
        Returns:
            Consolidation results
        """
        try:
            logger.info(f"Starting sleep consolidation for {duration_minutes} minutes")
            
            start_time = datetime.now(timezone.utc)
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            total_consolidations = 0
            consolidation_cycles = 0
            
            while datetime.now(timezone.utc) < end_time:
                # Consolidate batch
                batch_result = await self.consolidate_batch(batch_size=50)
                total_consolidations += batch_result.get('successful_consolidations', 0)
                consolidation_cycles += 1
                
                # Sleep between consolidation cycles
                await asyncio.sleep(60)  # 1 minute between cycles
            
            # Final optimization
            await self._optimize_memory_connections()
            
            consolidation_results = {
                'duration_minutes': duration_minutes,
                'consolidation_cycles': consolidation_cycles,
                'total_consolidations': total_consolidations,
                'average_per_cycle': total_consolidations / max(1, consolidation_cycles)
            }
            
            logger.info(f"Sleep consolidation completed: {consolidation_results}")
            
            return consolidation_results
            
        except Exception as e:
            logger.error(f"Failed sleep consolidation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _optimize_memory_connections(self):
        """Optimize connections between memory systems."""
        try:
            # This would implement sophisticated optimization
            # For now, just a placeholder
            logger.info("Optimizing memory connections...")
            
            # Could include:
            # - Strengthening frequently used connections
            # - Weakening unused connections
            # - Creating new associative links
            # - Reorganizing memory hierarchies
            
        except Exception as e:
            logger.error(f"Failed to optimize memory connections: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        with self.lock:
            return self.stats.copy()
    
    async def shutdown(self):
        """Shutdown memory consolidation system."""
        try:
            # Final consolidation
            await self.consolidate_batch(batch_size=1000)
            
            logger.info("Memory consolidation system shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown memory consolidation: {e}")