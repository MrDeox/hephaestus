"""
Procedural Memory Implementation for RSI AI.
Implements skills, actions, and procedures storage and retrieval.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
import hashlib
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A skill stored in procedural memory."""
    id: str
    name: str
    description: str
    skill_type: str  # 'action', 'procedure', 'heuristic', 'algorithm'
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    implementation: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    complexity: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'skill_type': self.skill_type,
            'parameters': self.parameters,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'implementation': self.implementation,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'complexity': self.complexity,
            'dependencies': self.dependencies
        }
    
    def update_performance(self, success: bool):
        """Update skill performance metrics."""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)


class ProceduralMemory:
    """
    Procedural Memory System for RSI AI.
    
    Stores and manages skills, actions, and procedures with:
    - Skill acquisition and refinement
    - Performance tracking
    - Dependency management
    - Hierarchical skill organization
    """
    
    def __init__(self, max_skills: int = 10000):
        self.max_skills = max_skills
        self.skills: Dict[str, Skill] = {}
        self.skill_index = defaultdict(set)  # Index by type, tags, etc.
        self.dependency_graph = defaultdict(set)
        self.execution_history = []
        
        # Persistent storage
        self.storage_path = Path("./procedural_memory.pkl")
        self._load_from_storage()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_skills': 0,
            'skill_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'skills_created': 0,
            'skills_updated': 0
        }
        
        logger.info("Procedural Memory initialized")
    
    def _load_from_storage(self):
        """Load skills from persistent storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.skills = data.get('skills', {})
                    self.skill_index = data.get('skill_index', defaultdict(set))
                    self.dependency_graph = data.get('dependency_graph', defaultdict(set))
                    self.stats = data.get('stats', self.stats)
                    
                logger.info(f"Loaded {len(self.skills)} skills from storage")
        except Exception as e:
            logger.warning(f"Failed to load procedural memory: {e}")
    
    def _save_to_storage(self):
        """Save skills to persistent storage."""
        try:
            data = {
                'skills': self.skills,
                'skill_index': dict(self.skill_index),
                'dependency_graph': dict(self.dependency_graph),
                'stats': self.stats
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save procedural memory: {e}")
    
    def _generate_skill_id(self, skill_data: Dict[str, Any]) -> str:
        """Generate unique ID for a skill."""
        skill_str = json.dumps({
            'name': skill_data.get('name', ''),
            'skill_type': skill_data.get('skill_type', ''),
            'parameters': skill_data.get('parameters', {})
        }, sort_keys=True)
        return hashlib.md5(skill_str.encode()).hexdigest()
    
    async def store_skill(self, skill_data: Dict[str, Any]) -> str:
        """
        Store a new skill in procedural memory.
        
        Args:
            skill_data: Skill information
            
        Returns:
            Skill ID
        """
        try:
            with self.lock:
                skill_id = self._generate_skill_id(skill_data)
                
                # Check if skill already exists
                if skill_id in self.skills:
                    # Update existing skill
                    existing_skill = self.skills[skill_id]
                    existing_skill.updated_at = datetime.now(timezone.utc)
                    existing_skill.description = skill_data.get('description', existing_skill.description)
                    existing_skill.implementation = skill_data.get('implementation', existing_skill.implementation)
                    existing_skill.tags = skill_data.get('tags', existing_skill.tags)
                    
                    self.stats['skills_updated'] += 1
                    self._save_to_storage()
                    return skill_id
                
                # Create new skill
                skill = Skill(
                    id=skill_id,
                    name=skill_data.get('name', ''),
                    description=skill_data.get('description', ''),
                    skill_type=skill_data.get('skill_type', 'action'),
                    parameters=skill_data.get('parameters', {}),
                    preconditions=skill_data.get('preconditions', []),
                    postconditions=skill_data.get('postconditions', []),
                    implementation=skill_data.get('implementation', {}),
                    tags=skill_data.get('tags', []),
                    complexity=skill_data.get('complexity', 1.0),
                    dependencies=skill_data.get('dependencies', [])
                )
                
                # Store skill
                self.skills[skill_id] = skill
                
                # Update indices
                self.skill_index['type'].add(skill.skill_type)
                for tag in skill.tags:
                    self.skill_index['tag'].add(tag)
                
                # Update dependency graph
                for dep in skill.dependencies:
                    self.dependency_graph[skill_id].add(dep)
                
                # Manage capacity
                if len(self.skills) > self.max_skills:
                    await self._cleanup_old_skills()
                
                self.stats['total_skills'] += 1
                self.stats['skills_created'] += 1
                
                self._save_to_storage()
                return skill_id
                
        except Exception as e:
            logger.error(f"Failed to store skill: {e}")
            raise
    
    async def retrieve_skills(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve skills from procedural memory.
        
        Args:
            query: Search query
            
        Returns:
            List of matching skills
        """
        try:
            with self.lock:
                results = []
                
                if 'id' in query:
                    # Direct ID lookup
                    skill_id = query['id']
                    if skill_id in self.skills:
                        skill = self.skills[skill_id]
                        results.append(skill.to_dict())
                
                elif 'name' in query:
                    # Name-based search
                    name = query['name'].lower()
                    for skill in self.skills.values():
                        if name in skill.name.lower():
                            results.append(skill.to_dict())
                
                elif 'skill_type' in query:
                    # Type-based search
                    skill_type = query['skill_type']
                    for skill in self.skills.values():
                        if skill.skill_type == skill_type:
                            results.append(skill.to_dict())
                
                elif 'tags' in query:
                    # Tag-based search
                    search_tags = query['tags']
                    if isinstance(search_tags, str):
                        search_tags = [search_tags]
                    
                    for skill in self.skills.values():
                        if any(tag in skill.tags for tag in search_tags):
                            results.append(skill.to_dict())
                
                elif 'preconditions' in query:
                    # Precondition-based search
                    preconditions = query['preconditions']
                    if isinstance(preconditions, str):
                        preconditions = [preconditions]
                    
                    for skill in self.skills.values():
                        if all(pre in skill.preconditions for pre in preconditions):
                            results.append(skill.to_dict())
                
                elif 'complexity_range' in query:
                    # Complexity-based search
                    min_complexity = query['complexity_range'].get('min', 0.0)
                    max_complexity = query['complexity_range'].get('max', 10.0)
                    
                    for skill in self.skills.values():
                        if min_complexity <= skill.complexity <= max_complexity:
                            results.append(skill.to_dict())
                
                elif 'best_performers' in query:
                    # Get best performing skills
                    limit = query.get('limit', 10)
                    sorted_skills = sorted(
                        self.skills.values(),
                        key=lambda s: s.success_rate,
                        reverse=True
                    )
                    results = [skill.to_dict() for skill in sorted_skills[:limit]]
                
                elif 'most_used' in query:
                    # Get most used skills
                    limit = query.get('limit', 10)
                    sorted_skills = sorted(
                        self.skills.values(),
                        key=lambda s: s.usage_count,
                        reverse=True
                    )
                    results = [skill.to_dict() for skill in sorted_skills[:limit]]
                
                else:
                    # Return all skills (limited)
                    limit = query.get('limit', 20)
                    results = [skill.to_dict() for skill in list(self.skills.values())[:limit]]
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve skills: {e}")
            return []
    
    async def execute_skill(self, skill_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a skill and track performance.
        
        Args:
            skill_id: Skill ID
            parameters: Execution parameters
            
        Returns:
            Execution result
        """
        try:
            with self.lock:
                if skill_id not in self.skills:
                    return {'success': False, 'error': 'Skill not found'}
                
                skill = self.skills[skill_id]
                execution_params = parameters or {}
                
                # Check preconditions
                if not self._check_preconditions(skill, execution_params):
                    return {'success': False, 'error': 'Preconditions not met'}
                
                # Record execution start
                execution_start = datetime.now(timezone.utc)
                
                # Execute skill (placeholder - would contain actual execution logic)
                try:
                    # This is where the actual skill execution would happen
                    result = await self._execute_skill_implementation(skill, execution_params)
                    success = result.get('success', True)
                    
                    # Update performance metrics
                    skill.update_performance(success)
                    
                    # Record execution
                    execution_record = {
                        'skill_id': skill_id,
                        'parameters': execution_params,
                        'result': result,
                        'success': success,
                        'timestamp': execution_start.isoformat(),
                        'duration': (datetime.now(timezone.utc) - execution_start).total_seconds()
                    }
                    
                    self.execution_history.append(execution_record)
                    
                    # Update stats
                    self.stats['skill_executions'] += 1
                    if success:
                        self.stats['successful_executions'] += 1
                    else:
                        self.stats['failed_executions'] += 1
                    
                    self._save_to_storage()
                    
                    return result
                    
                except Exception as e:
                    # Handle execution failure
                    skill.update_performance(False)
                    self.stats['skill_executions'] += 1
                    self.stats['failed_executions'] += 1
                    
                    return {'success': False, 'error': str(e)}
                    
        except Exception as e:
            logger.error(f"Failed to execute skill: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_preconditions(self, skill: Skill, parameters: Dict[str, Any]) -> bool:
        """Check if skill preconditions are met."""
        try:
            # Simple precondition checking (would be more sophisticated in practice)
            for precondition in skill.preconditions:
                if precondition.startswith('param:'):
                    param_name = precondition[6:]  # Remove 'param:' prefix
                    if param_name not in parameters:
                        return False
                
                # Add more precondition types as needed
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to check preconditions: {e}")
            return False
    
    async def _execute_skill_implementation(self, skill: Skill, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual skill implementation."""
        try:
            # This is a placeholder for actual skill execution
            # In practice, this would dispatch to different execution engines
            # based on the skill type and implementation
            
            implementation = skill.implementation
            skill_type = skill.skill_type
            
            if skill_type == 'action':
                # Execute action
                result = await self._execute_action(implementation, parameters)
            elif skill_type == 'procedure':
                # Execute procedure
                result = await self._execute_procedure(implementation, parameters)
            elif skill_type == 'heuristic':
                # Execute heuristic
                result = await self._execute_heuristic(implementation, parameters)
            elif skill_type == 'algorithm':
                # Execute algorithm
                result = await self._execute_algorithm(implementation, parameters)
            else:
                result = {'success': False, 'error': f'Unknown skill type: {skill_type}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute skill implementation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_action(self, implementation: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action skill."""
        # Placeholder for action execution
        return {'success': True, 'result': 'Action executed successfully'}
    
    async def _execute_procedure(self, implementation: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a procedure skill."""
        # Placeholder for procedure execution
        return {'success': True, 'result': 'Procedure executed successfully'}
    
    async def _execute_heuristic(self, implementation: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a heuristic skill."""
        # Placeholder for heuristic execution
        return {'success': True, 'result': 'Heuristic executed successfully'}
    
    async def _execute_algorithm(self, implementation: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an algorithm skill."""
        # Placeholder for algorithm execution
        return {'success': True, 'result': 'Algorithm executed successfully'}
    
    async def get_skill_dependencies(self, skill_id: str) -> List[str]:
        """Get dependencies for a skill."""
        try:
            with self.lock:
                if skill_id not in self.skills:
                    return []
                
                return list(self.dependency_graph.get(skill_id, set()))
                
        except Exception as e:
            logger.error(f"Failed to get skill dependencies: {e}")
            return []
    
    async def get_skills_count(self) -> int:
        """Get total number of skills."""
        return len(self.skills)
    
    async def _cleanup_old_skills(self):
        """Clean up old, unused skills."""
        try:
            with self.lock:
                # Remove skills that haven't been used recently and have low success rates
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                
                skills_to_remove = []
                for skill_id, skill in self.skills.items():
                    if (skill.last_used < cutoff_date and 
                        skill.success_rate < 0.3 and 
                        skill.usage_count < 5):
                        skills_to_remove.append(skill_id)
                
                for skill_id in skills_to_remove:
                    await self.remove_skill(skill_id)
                
                logger.info(f"Cleaned up {len(skills_to_remove)} old skills")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old skills: {e}")
    
    async def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill from procedural memory."""
        try:
            with self.lock:
                if skill_id not in self.skills:
                    return False
                
                skill = self.skills[skill_id]
                
                # Remove from main storage
                del self.skills[skill_id]
                
                # Remove from indices
                for tag in skill.tags:
                    self.skill_index['tag'].discard(tag)
                
                # Remove from dependency graph
                if skill_id in self.dependency_graph:
                    del self.dependency_graph[skill_id]
                
                # Remove references to this skill in other dependencies
                for deps in self.dependency_graph.values():
                    deps.discard(skill_id)
                
                self.stats['total_skills'] -= 1
                self._save_to_storage()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove skill: {e}")
            return False
    
    async def cleanup(self):
        """Clean up procedural memory."""
        try:
            await self._cleanup_old_skills()
            
            # Clean up execution history
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
            self.execution_history = [
                record for record in self.execution_history
                if datetime.fromisoformat(record['timestamp']) > cutoff_date
            ]
            
            self._save_to_storage()
            
        except Exception as e:
            logger.error(f"Failed to cleanup procedural memory: {e}")
    
    async def optimize(self):
        """Optimize procedural memory performance."""
        try:
            # Rebuild indices
            with self.lock:
                self.skill_index.clear()
                
                for skill in self.skills.values():
                    self.skill_index['type'].add(skill.skill_type)
                    for tag in skill.tags:
                        self.skill_index['tag'].add(tag)
                
                # Optimize dependency graph
                valid_skills = set(self.skills.keys())
                for skill_id in list(self.dependency_graph.keys()):
                    if skill_id not in valid_skills:
                        del self.dependency_graph[skill_id]
                    else:
                        # Remove invalid dependencies
                        self.dependency_graph[skill_id] = {
                            dep for dep in self.dependency_graph[skill_id]
                            if dep in valid_skills
                        }
                
                self._save_to_storage()
                
        except Exception as e:
            logger.error(f"Failed to optimize procedural memory: {e}")
    
    def get_size(self) -> int:
        """Get current size of procedural memory."""
        return len(self.skills)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get procedural memory statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'current_skills': len(self.skills),
                'skill_types': len(self.skill_index['type']),
                'total_tags': len(self.skill_index['tag']),
                'execution_history_size': len(self.execution_history),
                'average_success_rate': sum(skill.success_rate for skill in self.skills.values()) / len(self.skills) if self.skills else 0
            })
            return stats
    
    async def shutdown(self):
        """Shutdown procedural memory."""
        try:
            # Final save
            self._save_to_storage()
            
            # Clear memory
            with self.lock:
                self.skills.clear()
                self.skill_index.clear()
                self.dependency_graph.clear()
                self.execution_history.clear()
            
            logger.info("Procedural memory shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown procedural memory: {e}")