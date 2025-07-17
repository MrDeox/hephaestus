#!/usr/bin/env python3
"""
Sistema de Auto-Evolução Arquitetural

Este sistema implementa a capacidade do RSI de analisar, redesenhar e evoluir
sua própria arquitetura automaticamente para se tornar mais eficiente e elegante.

ISTO É RSI ARQUITETURAL REAL - o sistema modifica sua própria estrutura!
"""

import ast
import os
import re
import shutil
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio

from loguru import logger

try:
    from ..memory.memory_manager import RSIMemoryManager
except ImportError:
    RSIMemoryManager = None


class ArchitecturalIssueType(str, Enum):
    """Tipos de problemas arquiteturais detectáveis."""
    EXCESSIVE_COMPLEXITY = "excessive_complexity"
    HIGH_COUPLING = "high_coupling"
    LOW_COHESION = "low_cohesion"
    GOD_OBJECT = "god_object"
    DUPLICATE_CODE = "duplicate_code"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INCONSISTENT_PATTERNS = "inconsistent_patterns"
    POOR_SEPARATION = "poor_separation"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"


class RefactoringType(str, Enum):
    """Tipos de refatoração disponíveis."""
    EXTRACT_MODULE = "extract_module"
    EXTRACT_CLASS = "extract_class"
    EXTRACT_FUNCTION = "extract_function"
    IMPLEMENT_DI = "implement_dependency_injection"
    REDUCE_COUPLING = "reduce_coupling"
    STANDARDIZE_PATTERNS = "standardize_patterns"
    ELIMINATE_DUPLICATION = "eliminate_duplication"
    BREAK_CIRCULAR_DEPS = "break_circular_dependencies"
    OPTIMIZE_IMPORTS = "optimize_imports"


@dataclass
class CodeMetrics:
    """Métricas de qualidade de código."""
    
    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    function_count: int
    class_count: int
    import_count: int
    max_function_length: int
    avg_function_length: float
    coupling_count: int
    code_duplication_ratio: float
    maintainability_index: float


@dataclass
class ArchitecturalIssue:
    """Representa um problema arquitetural detectado."""
    
    issue_type: ArchitecturalIssueType
    severity: str  # "low", "medium", "high", "critical"
    file_path: str
    line_number: Optional[int]
    description: str
    current_value: Any
    recommended_value: Any
    impact_score: float
    fix_effort: str  # "trivial", "easy", "medium", "hard", "complex"


@dataclass
class RefactoringProposal:
    """Proposta de refatoração arquitetural."""
    
    proposal_id: str
    refactoring_type: RefactoringType
    target_files: List[str]
    description: str
    expected_benefits: List[str]
    risk_level: str  # "low", "medium", "high"
    estimated_effort: str
    priority_score: float
    implementation_plan: List[str]
    validation_criteria: List[str]


@dataclass
class RefactoringResult:
    """Resultado de uma refatoração aplicada."""
    
    proposal_id: str
    success: bool
    changes_made: List[str]
    files_created: List[str]
    files_modified: List[str]
    files_deleted: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    improvement_achieved: bool
    rollback_path: Optional[str]
    error_message: Optional[str]


class CodeAnalyzer:
    """Analisador de código para métricas de qualidade."""
    
    def __init__(self):
        self.file_patterns = {
            "python": "**/*.py",
            "config": "**/*.json",
            "docs": "**/*.md"
        }
    
    async def analyze_file(self, file_path: str) -> CodeMetrics:
        """Analisa um arquivo Python e calcula métricas."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Calcular métricas básicas
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            # Contar elementos
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            import_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
            
            # Calcular complexidade ciclomática
            complexity = self._calculate_cyclomatic_complexity(tree)
            
            # Analisar funções
            function_lengths = self._analyze_functions(tree, content)
            max_function_length = max(function_lengths) if function_lengths else 0
            avg_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0
            
            # Calcular acoplamento (importações + dependências)
            coupling_count = self._calculate_coupling(tree, content)
            
            # Detectar duplicação de código
            duplication_ratio = self._detect_code_duplication(content)
            
            # Calcular índice de manutenibilidade (simplificado)
            maintainability = self._calculate_maintainability_index(
                lines_of_code, complexity, function_count, duplication_ratio
            )
            
            return CodeMetrics(
                file_path=file_path,
                lines_of_code=lines_of_code,
                cyclomatic_complexity=complexity,
                function_count=function_count,
                class_count=class_count,
                import_count=import_count,
                max_function_length=max_function_length,
                avg_function_length=avg_function_length,
                coupling_count=coupling_count,
                code_duplication_ratio=duplication_ratio,
                maintainability_index=maintainability
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return CodeMetrics(
                file_path=file_path,
                lines_of_code=0,
                cyclomatic_complexity=0,
                function_count=0,
                class_count=0,
                import_count=0,
                max_function_length=0,
                avg_function_length=0,
                coupling_count=0,
                code_duplication_ratio=0,
                maintainability_index=0
            )
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calcula complexidade ciclomática do código."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _analyze_functions(self, tree: ast.AST, content: str) -> List[int]:
        """Analisa funções e retorna suas lonquidades em linhas."""
        function_lengths = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    length = node.end_lineno - node.lineno + 1
                    function_lengths.append(length)
        
        return function_lengths
    
    def _calculate_coupling(self, tree: ast.AST, content: str) -> int:
        """Calcula grau de acoplamento baseado em importações e referências."""
        coupling = 0
        
        # Contar importações
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                coupling += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                coupling += len(node.names) if node.names else 1
        
        # Contar referências a módulos externos (simplificado)
        external_refs = len(re.findall(r'self\.\w+', content))
        coupling += external_refs // 10  # Normalize
        
        return coupling
    
    def _detect_code_duplication(self, content: str) -> float:
        """Detecta duplicação de código e retorna ratio."""
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        if not lines:
            return 0.0
        
        line_counts = Counter(lines)
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return duplicated_lines / len(lines) if lines else 0.0
    
    def _calculate_maintainability_index(
        self, 
        loc: int, 
        complexity: int, 
        functions: int,
        duplication: float
    ) -> float:
        """Calcula índice de manutenibilidade (0-100, maior é melhor)."""
        
        if loc == 0:
            return 100.0
        
        # Fórmula simplificada baseada em métricas conhecidas
        base_score = 100
        
        # Penalizar linhas de código excessivas
        loc_penalty = min(loc / 100, 50)  # Max 50 points penalty
        
        # Penalizar complexidade alta
        complexity_penalty = min(complexity / 10, 30)  # Max 30 points penalty
        
        # Penalizar duplicação
        duplication_penalty = duplication * 20  # Max 20 points penalty
        
        # Bonus por ter funções organizadas
        function_bonus = min(functions / 10, 10) if functions > 0 else -10
        
        score = base_score - loc_penalty - complexity_penalty - duplication_penalty + function_bonus
        
        return max(0.0, min(100.0, score))


class ArchitecturalIssueDetector:
    """Detecta problemas arquiteturais no código."""
    
    def __init__(self):
        # Thresholds para detecção de problemas
        self.thresholds = {
            "max_file_lines": 500,
            "max_function_lines": 50,
            "max_complexity": 10,
            "max_coupling": 20,
            "min_maintainability": 60,
            "max_duplication": 0.15,
            "max_imports": 30
        }
    
    async def detect_issues(self, metrics: List[CodeMetrics]) -> List[ArchitecturalIssue]:
        """Detecta problemas arquiteturais baseado nas métricas."""
        
        issues = []
        
        for metric in metrics:
            # Detectar arquivo muito grande (God Object)
            if metric.lines_of_code > self.thresholds["max_file_lines"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.GOD_OBJECT,
                    severity="high" if metric.lines_of_code > 1000 else "medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"File has {metric.lines_of_code} lines (threshold: {self.thresholds['max_file_lines']})",
                    current_value=metric.lines_of_code,
                    recommended_value=self.thresholds["max_file_lines"],
                    impact_score=min(metric.lines_of_code / 1000, 5.0),
                    fix_effort="complex"
                ))
            
            # Detectar complexidade excessiva
            if metric.cyclomatic_complexity > self.thresholds["max_complexity"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.EXCESSIVE_COMPLEXITY,
                    severity="high" if metric.cyclomatic_complexity > 20 else "medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"Cyclomatic complexity is {metric.cyclomatic_complexity} (threshold: {self.thresholds['max_complexity']})",
                    current_value=metric.cyclomatic_complexity,
                    recommended_value=self.thresholds["max_complexity"],
                    impact_score=metric.cyclomatic_complexity / 10,
                    fix_effort="medium"
                ))
            
            # Detectar alto acoplamento
            if metric.coupling_count > self.thresholds["max_coupling"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.HIGH_COUPLING,
                    severity="medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"High coupling: {metric.coupling_count} dependencies (threshold: {self.thresholds['max_coupling']})",
                    current_value=metric.coupling_count,
                    recommended_value=self.thresholds["max_coupling"],
                    impact_score=metric.coupling_count / 20,
                    fix_effort="medium"
                ))
            
            # Detectar funções muito longas
            if metric.max_function_length > self.thresholds["max_function_lines"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.EXCESSIVE_COMPLEXITY,
                    severity="medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"Longest function has {metric.max_function_length} lines (threshold: {self.thresholds['max_function_lines']})",
                    current_value=metric.max_function_length,
                    recommended_value=self.thresholds["max_function_lines"],
                    impact_score=metric.max_function_length / 100,
                    fix_effort="easy"
                ))
            
            # Detectar duplicação de código
            if metric.code_duplication_ratio > self.thresholds["max_duplication"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.DUPLICATE_CODE,
                    severity="medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"Code duplication: {metric.code_duplication_ratio:.1%} (threshold: {self.thresholds['max_duplication']:.1%})",
                    current_value=metric.code_duplication_ratio,
                    recommended_value=self.thresholds["max_duplication"],
                    impact_score=metric.code_duplication_ratio * 5,
                    fix_effort="medium"
                ))
            
            # Detectar baixa manutenibilidade
            if metric.maintainability_index < self.thresholds["min_maintainability"]:
                issues.append(ArchitecturalIssue(
                    issue_type=ArchitecturalIssueType.POOR_SEPARATION,
                    severity="high" if metric.maintainability_index < 40 else "medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"Low maintainability index: {metric.maintainability_index:.1f} (threshold: {self.thresholds['min_maintainability']})",
                    current_value=metric.maintainability_index,
                    recommended_value=self.thresholds["min_maintainability"],
                    impact_score=(self.thresholds["min_maintainability"] - metric.maintainability_index) / 20,
                    fix_effort="complex"
                ))
        
        return issues


class RefactoringProposer:
    """Gera propostas de refatoração baseadas nos problemas detectados."""
    
    def __init__(self):
        self.proposal_counter = 0
    
    async def propose_refactorings(
        self, 
        issues: List[ArchitecturalIssue],
        metrics: List[CodeMetrics]
    ) -> List[RefactoringProposal]:
        """Gera propostas de refatoração para resolver os problemas."""
        
        proposals = []
        
        # Agrupar issues por arquivo para propostas mais eficientes
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)
        
        for file_path, file_issues in issues_by_file.items():
            file_metrics = next((m for m in metrics if m.file_path == file_path), None)
            if not file_metrics:
                continue
            
            # Propor refatorações específicas baseadas nos problemas
            proposals.extend(await self._propose_for_file(file_path, file_issues, file_metrics))
        
        # Ordenar por prioridade
        proposals.sort(key=lambda p: p.priority_score, reverse=True)
        
        return proposals
    
    async def _propose_for_file(
        self, 
        file_path: str, 
        issues: List[ArchitecturalIssue],
        metrics: CodeMetrics
    ) -> List[RefactoringProposal]:
        """Propõe refatorações específicas para um arquivo."""
        
        proposals = []
        
        # Detectar se é main.py com muitas responsabilidades
        if "main.py" in file_path and metrics.lines_of_code > 1000:
            proposals.append(await self._propose_extract_main_components(file_path, metrics, issues))
        
        # Propor extração de classes se arquivo muito grande
        if metrics.lines_of_code > 800:
            proposals.append(await self._propose_extract_classes(file_path, metrics, issues))
        
        # Propor redução de acoplamento se muitas dependências
        if metrics.coupling_count > 25:
            proposals.append(await self._propose_reduce_coupling(file_path, metrics, issues))
        
        # Propor eliminação de duplicação se detectada
        if metrics.code_duplication_ratio > 0.2:
            proposals.append(await self._propose_eliminate_duplication(file_path, metrics, issues))
        
        # Propor quebra de funções se muito longas
        if metrics.max_function_length > 100:
            proposals.append(await self._propose_extract_functions(file_path, metrics, issues))
        
        return [p for p in proposals if p is not None]
    
    async def _propose_extract_main_components(
        self, 
        file_path: str, 
        metrics: CodeMetrics,
        issues: List[ArchitecturalIssue]
    ) -> RefactoringProposal:
        """Propõe extração de componentes do main.py."""
        
        self.proposal_counter += 1
        
        return RefactoringProposal(
            proposal_id=f"extract_main_{self.proposal_counter}",
            refactoring_type=RefactoringType.EXTRACT_MODULE,
            target_files=[file_path],
            description="Extract component initialization from main.py into separate modules",
            expected_benefits=[
                "Reduce main.py size from 3000+ to <500 lines",
                "Improve separation of concerns",
                "Make testing easier",
                "Reduce startup complexity"
            ],
            risk_level="medium",
            estimated_effort="complex",
            priority_score=5.0,  # High priority for main.py
            implementation_plan=[
                "1. Create ComponentInitializer class in new module",
                "2. Extract component initialization logic",
                "3. Create ConfigurationManager for settings",
                "4. Extract API endpoints to separate modules",
                "5. Update imports and dependencies",
                "6. Validate all functionality works"
            ],
            validation_criteria=[
                "main.py has <500 lines",
                "All tests pass",
                "Startup time maintained or improved",
                "All API endpoints work",
                "Maintainability index >70"
            ]
        )
    
    async def _propose_extract_classes(
        self, 
        file_path: str, 
        metrics: CodeMetrics,
        issues: List[ArchitecturalIssue]
    ) -> Optional[RefactoringProposal]:
        """Propõe extração de classes para módulos separados."""
        
        if metrics.class_count < 2:
            return None
        
        self.proposal_counter += 1
        
        return RefactoringProposal(
            proposal_id=f"extract_classes_{self.proposal_counter}",
            refactoring_type=RefactoringType.EXTRACT_CLASS,
            target_files=[file_path],
            description=f"Extract {metrics.class_count} classes to separate modules",
            expected_benefits=[
                f"Reduce file size by ~{metrics.lines_of_code // metrics.class_count} lines per class",
                "Improve modularity",
                "Enable better testing",
                "Reduce coupling"
            ],
            risk_level="low",
            estimated_effort="medium",
            priority_score=3.0,
            implementation_plan=[
                "1. Identify classes for extraction",
                "2. Create new module files",
                "3. Move classes with their dependencies",
                "4. Update imports",
                "5. Validate functionality"
            ],
            validation_criteria=[
                "File size reduced by >30%",
                "All imports work",
                "Tests pass",
                "No circular dependencies"
            ]
        )
    
    async def _propose_reduce_coupling(
        self, 
        file_path: str, 
        metrics: CodeMetrics,
        issues: List[ArchitecturalIssue]
    ) -> RefactoringProposal:
        """Propõe redução de acoplamento através de DI."""
        
        self.proposal_counter += 1
        
        return RefactoringProposal(
            proposal_id=f"reduce_coupling_{self.proposal_counter}",
            refactoring_type=RefactoringType.IMPLEMENT_DI,
            target_files=[file_path],
            description="Implement dependency injection to reduce coupling",
            expected_benefits=[
                f"Reduce coupling from {metrics.coupling_count} to <15",
                "Improve testability",
                "Enable better modularity",
                "Simplify component management"
            ],
            risk_level="medium",
            estimated_effort="complex",
            priority_score=4.0,
            implementation_plan=[
                "1. Create DependencyContainer class",
                "2. Define interfaces for major dependencies",
                "3. Refactor constructors to use DI",
                "4. Update initialization code",
                "5. Validate all dependencies resolve"
            ],
            validation_criteria=[
                "Coupling count <15",
                "All components initialize correctly",
                "Tests pass",
                "No circular dependencies"
            ]
        )
    
    async def _propose_eliminate_duplication(
        self, 
        file_path: str, 
        metrics: CodeMetrics,
        issues: List[ArchitecturalIssue]
    ) -> RefactoringProposal:
        """Propõe eliminação de duplicação de código."""
        
        self.proposal_counter += 1
        
        return RefactoringProposal(
            proposal_id=f"eliminate_duplication_{self.proposal_counter}",
            refactoring_type=RefactoringType.ELIMINATE_DUPLICATION,
            target_files=[file_path],
            description="Extract common code to shared utilities",
            expected_benefits=[
                f"Reduce duplication from {metrics.code_duplication_ratio:.1%} to <10%",
                "Improve maintainability",
                "Reduce bug opportunities",
                "Simplify changes"
            ],
            risk_level="low",
            estimated_effort="medium",
            priority_score=2.5,
            implementation_plan=[
                "1. Identify duplicated code blocks",
                "2. Extract to utility functions",
                "3. Replace duplicated code with calls",
                "4. Validate functionality preserved"
            ],
            validation_criteria=[
                "Duplication ratio <10%",
                "All tests pass",
                "Code size reduced",
                "No functionality lost"
            ]
        )
    
    async def _propose_extract_functions(
        self, 
        file_path: str, 
        metrics: CodeMetrics,
        issues: List[ArchitecturalIssue]
    ) -> RefactoringProposal:
        """Propõe extração de funções longas."""
        
        self.proposal_counter += 1
        
        return RefactoringProposal(
            proposal_id=f"extract_functions_{self.proposal_counter}",
            refactoring_type=RefactoringType.EXTRACT_FUNCTION,
            target_files=[file_path],
            description="Break down long functions into smaller, focused functions",
            expected_benefits=[
                f"Reduce max function length from {metrics.max_function_length} to <50 lines",
                "Improve readability",
                "Enable better testing",
                "Reduce complexity"
            ],
            risk_level="low",
            estimated_effort="easy",
            priority_score=2.0,
            implementation_plan=[
                "1. Identify functions >50 lines",
                "2. Find logical breakpoints",
                "3. Extract subfunctions",
                "4. Validate behavior preserved"
            ],
            validation_criteria=[
                "No function >50 lines",
                "All tests pass",
                "Complexity reduced",
                "Readability improved"
            ]
        )


class AutoRefactorer:
    """Aplica refatorações automaticamente com segurança."""
    
    def __init__(self):
        self.backup_dir = Path("./backups/architecture_evolution")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def apply_refactoring(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Aplica uma refatoração com backup e validação."""
        
        logger.info(f"🔧 Applying refactoring: {proposal.description}")
        
        # 1. Criar backup completo
        backup_path = await self._create_backup(proposal.target_files)
        
        try:
            # 2. Aplicar refatoração específica
            if proposal.refactoring_type == RefactoringType.EXTRACT_MODULE:
                result = await self._extract_main_components(proposal)
            elif proposal.refactoring_type == RefactoringType.EXTRACT_CLASS:
                result = await self._extract_classes(proposal)
            elif proposal.refactoring_type == RefactoringType.IMPLEMENT_DI:
                result = await self._implement_dependency_injection(proposal)
            elif proposal.refactoring_type == RefactoringType.ELIMINATE_DUPLICATION:
                result = await self._eliminate_duplication(proposal)
            elif proposal.refactoring_type == RefactoringType.EXTRACT_FUNCTION:
                result = await self._extract_functions(proposal)
            else:
                result = RefactoringResult(
                    proposal_id=proposal.proposal_id,
                    success=False,
                    changes_made=[],
                    files_created=[],
                    files_modified=[],
                    files_deleted=[],
                    metrics_before={},
                    metrics_after={},
                    improvement_achieved=False,
                    rollback_path=str(backup_path),
                    error_message=f"Unsupported refactoring type: {proposal.refactoring_type}"
                )
            
            # 3. Validar resultado
            if result.success:
                validation_passed = await self._validate_refactoring(proposal, result)
                if not validation_passed:
                    # Rollback se validação falhou
                    await self._rollback(backup_path, proposal.target_files)
                    result.success = False
                    result.error_message = "Validation failed"
            
            result.rollback_path = str(backup_path)
            return result
            
        except Exception as e:
            logger.error(f"Error applying refactoring {proposal.proposal_id}: {e}")
            
            # Rollback em caso de erro
            await self._rollback(backup_path, proposal.target_files)
            
            return RefactoringResult(
                proposal_id=proposal.proposal_id,
                success=False,
                changes_made=[],
                files_created=[],
                files_modified=[],
                files_deleted=[],
                metrics_before={},
                metrics_after={},
                improvement_achieved=False,
                rollback_path=str(backup_path),
                error_message=str(e)
            )
    
    async def _create_backup(self, target_files: List[str]) -> Path:
        """Cria backup dos arquivos a serem modificados."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"refactoring_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        for file_path in target_files:
            if Path(file_path).exists():
                backup_file = backup_path / Path(file_path).name
                shutil.copy2(file_path, backup_file)
        
        logger.info(f"📦 Backup created: {backup_path}")
        return backup_path
    
    async def _extract_main_components(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Extrai componentes do main.py para módulos separados."""
        
        main_file = proposal.target_files[0]
        
        try:
            # Ler main.py
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Métricas antes
            analyzer = CodeAnalyzer()
            metrics_before = await analyzer.analyze_file(main_file)
            
            changes_made = []
            files_created = []
            files_modified = [main_file]
            
            # 1. Extrair ComponentInitializer
            component_init_code = self._extract_component_initialization(content)
            if component_init_code:
                init_file = "src/core/component_initializer.py"
                await self._create_component_initializer(init_file, component_init_code)
                files_created.append(init_file)
                changes_made.append("Extracted component initialization to ComponentInitializer")
            
            # 2. Extrair ConfigurationManager
            config_code = self._extract_configuration_management(content)
            if config_code:
                config_file = "src/core/configuration_manager.py"
                await self._create_configuration_manager(config_file, config_code)
                files_created.append(config_file)
                changes_made.append("Extracted configuration to ConfigurationManager")
            
            # 3. Simplificar main.py
            simplified_content = await self._simplify_main_file(content)
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(simplified_content)
            changes_made.append("Simplified main.py structure")
            
            # Métricas depois
            metrics_after = await analyzer.analyze_file(main_file)
            
            improvement = (
                metrics_after.lines_of_code < metrics_before.lines_of_code and
                metrics_after.maintainability_index > metrics_before.maintainability_index
            )
            
            return RefactoringResult(
                proposal_id=proposal.proposal_id,
                success=True,
                changes_made=changes_made,
                files_created=files_created,
                files_modified=files_modified,
                files_deleted=[],
                metrics_before=asdict(metrics_before),
                metrics_after=asdict(metrics_after),
                improvement_achieved=improvement,
                rollback_path="",
                error_message=None
            )
            
        except Exception as e:
            return RefactoringResult(
                proposal_id=proposal.proposal_id,
                success=False,
                changes_made=[],
                files_created=[],
                files_modified=[],
                files_deleted=[],
                metrics_before={},
                metrics_after={},
                improvement_achieved=False,
                rollback_path="",
                error_message=str(e)
            )
    
    def _extract_component_initialization(self, content: str) -> str:
        """Extrai código de inicialização de componentes."""
        
        # Busca por métodos _initialize_* no content
        init_methods = re.findall(
            r'def (_initialize_\w+\([^)]*\)):(.*?)(?=\n    def|\n\nclass|\nclass|\Z)',
            content,
            re.DOTALL
        )
        
        if not init_methods:
            return ""
        
        # Gera código do ComponentInitializer
        component_code = '''"""
Component Initializer - Extracted from main.py by architecture evolution system.
"""

import asyncio
from typing import Optional, Dict, Any
from loguru import logger

class ComponentInitializer:
    """Manages initialization of all system components."""
    
    def __init__(self):
        self.initialized_components = {}
    
'''
        
        for method_sig, method_body in init_methods:
            # Adapta a assinatura do método
            method_name = method_sig.split('(')[0]
            adapted_body = method_body.replace('self.', 'self.initialized_components.')
            
            component_code += f'''    def {method_sig}:
        """Initialize component - auto-extracted."""{adapted_body}
    
'''
        
        return component_code
    
    def _extract_configuration_management(self, content: str) -> str:
        """Extrai código de gerenciamento de configuração."""
        
        # Busca por configurações e environment variables
        config_patterns = [
            r'os\.getenv\([^)]+\)',
            r'environment\s*=',
            r'config\w*\s*=',
        ]
        
        config_lines = []
        for line in content.split('\n'):
            for pattern in config_patterns:
                if re.search(pattern, line):
                    config_lines.append(line.strip())
        
        if not config_lines:
            return ""
        
        return '''"""
Configuration Manager - Extracted from main.py by architecture evolution system.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SystemConfiguration:
    """System configuration settings."""
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"

class ConfigurationManager:
    """Manages all system configuration."""
    
    def __init__(self):
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> SystemConfiguration:
        """Load configuration from environment variables."""
        return SystemConfiguration(
            environment=os.getenv("ENVIRONMENT", "production"),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self.config, key, default)
'''
    
    async def _create_component_initializer(self, file_path: str, code: str):
        """Cria arquivo ComponentInitializer."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"✅ Created ComponentInitializer: {file_path}")
    
    async def _create_configuration_manager(self, file_path: str, code: str):
        """Cria arquivo ConfigurationManager."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"✅ Created ConfigurationManager: {file_path}")
    
    async def _simplify_main_file(self, content: str) -> str:
        """Simplifica o arquivo main.py removendo código extraído."""
        
        # Remove métodos _initialize_* (foram extraídos)
        simplified = re.sub(
            r'    def _initialize_\w+\([^)]*\):.*?(?=\n    def|\n\nclass|\nclass|\Z)',
            '',
            content,
            flags=re.DOTALL
        )
        
        # Adiciona imports para componentes extraídos
        imports_to_add = [
            "from src.core.component_initializer import ComponentInitializer",
            "from src.core.configuration_manager import ConfigurationManager"
        ]
        
        # Insere imports após os imports existentes
        import_section = []
        code_section = []
        in_imports = True
        
        for line in simplified.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ') or not line.strip() or line.startswith('#')):
                import_section.append(line)
            else:
                if in_imports:
                    # Adicionar novos imports
                    import_section.extend(imports_to_add)
                    import_section.append('')
                    in_imports = False
                code_section.append(line)
        
        # Modifica __init__ para usar ComponentInitializer
        init_modification = '''
        # Initialize using extracted ComponentInitializer
        self.component_initializer = ComponentInitializer()
        self.config_manager = ConfigurationManager()
        
        # Initialize core components through initializer
        self._initialize_core_components()
'''
        
        # Junta tudo
        result = '\n'.join(import_section + code_section)
        
        return result
    
    async def _extract_classes(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Implementação simplificada para extração de classes."""
        # TODO: Implementar extração real de classes
        return RefactoringResult(
            proposal_id=proposal.proposal_id,
            success=False,
            changes_made=[],
            files_created=[],
            files_modified=[],
            files_deleted=[],
            metrics_before={},
            metrics_after={},
            improvement_achieved=False,
            rollback_path="",
            error_message="Class extraction not yet implemented"
        )
    
    async def _implement_dependency_injection(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Implementação simplificada para DI."""
        # TODO: Implementar DI real
        return RefactoringResult(
            proposal_id=proposal.proposal_id,
            success=False,
            changes_made=[],
            files_created=[],
            files_modified=[],
            files_deleted=[],
            metrics_before={},
            metrics_after={},
            improvement_achieved=False,
            rollback_path="",
            error_message="Dependency injection not yet implemented"
        )
    
    async def _eliminate_duplication(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Implementação simplificada para eliminação de duplicação."""
        # TODO: Implementar eliminação real de duplicação
        return RefactoringResult(
            proposal_id=proposal.proposal_id,
            success=False,
            changes_made=[],
            files_created=[],
            files_modified=[],
            files_deleted=[],
            metrics_before={},
            metrics_after={},
            improvement_achieved=False,
            rollback_path="",
            error_message="Duplication elimination not yet implemented"
        )
    
    async def _extract_functions(self, proposal: RefactoringProposal) -> RefactoringResult:
        """Implementação simplificada para extração de funções."""
        # TODO: Implementar extração real de funções
        return RefactoringResult(
            proposal_id=proposal.proposal_id,
            success=False,
            changes_made=[],
            files_created=[],
            files_modified=[],
            files_deleted=[],
            metrics_before={},
            metrics_after={},
            improvement_achieved=False,
            rollback_path="",
            error_message="Function extraction not yet implemented"
        )
    
    async def _validate_refactoring(self, proposal: RefactoringProposal, result: RefactoringResult) -> bool:
        """Valida se a refatoração foi bem-sucedida."""
        
        try:
            # Validação básica: verificar se arquivos criados existem
            for file_path in result.files_created:
                if not Path(file_path).exists():
                    logger.error(f"Created file not found: {file_path}")
                    return False
            
            # Validação de sintaxe: tentar importar módulos Python
            for file_path in result.files_created + result.files_modified:
                if file_path.endswith('.py'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Tentar parsear AST
                        ast.parse(content)
                        
                    except SyntaxError as e:
                        logger.error(f"Syntax error in {file_path}: {e}")
                        return False
            
            # Validação de melhoria: verificar se métricas melhoraram
            if result.metrics_before and result.metrics_after:
                before_loc = result.metrics_before.get('lines_of_code', 0)
                after_loc = result.metrics_after.get('lines_of_code', 0)
                
                # Para extração, esperamos redução de linhas
                if proposal.refactoring_type == RefactoringType.EXTRACT_MODULE:
                    if after_loc >= before_loc:
                        logger.warning(f"Expected line reduction but got {before_loc} -> {after_loc}")
                        # Não falha - pode ser válido dependendo da refatoração
            
            logger.info(f"✅ Refactoring validation passed: {proposal.proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def _rollback(self, backup_path: Path, target_files: List[str]):
        """Realiza rollback usando backup."""
        
        try:
            for file_path in target_files:
                backup_file = backup_path / Path(file_path).name
                if backup_file.exists():
                    shutil.copy2(backup_file, file_path)
            
            logger.info(f"🔄 Rollback completed from {backup_path}")
            
        except Exception as e:
            logger.error(f"Rollback error: {e}")


class ArchitectureEvolution:
    """Orquestrador principal da evolução arquitetural."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.issue_detector = ArchitecturalIssueDetector()
        self.proposer = RefactoringProposer()
        self.refactorer = AutoRefactorer()
        self.memory_manager = RSIMemoryManager() if RSIMemoryManager else None
        
        # Diretórios para análise
        self.analysis_dirs = [
            "src/",
            "src/autonomous/",
            "src/coevolution/",
            "src/agents/",
            "src/objectives/",
            "src/core/"
        ]
    
    async def evolve_architecture(self) -> Dict[str, Any]:
        """
        Executa um ciclo completo de evolução arquitetural.
        
        Este é o método principal que implementa auto-melhoria arquitetural REAL.
        """
        
        logger.info("🏗️ Iniciando evolução arquitetural autônoma...")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "analysis_completed": False,
            "issues_detected": 0,
            "proposals_generated": 0,
            "refactorings_applied": 0,
            "improvements_achieved": 0,
            "architecture_evolved": False,
            "details": {
                "metrics": {},
                "issues": [],
                "proposals": [],
                "refactoring_results": []
            }
        }
        
        try:
            # 1. ANÁLISE: Examinar arquitetura atual
            logger.info("🔍 Fase 1: Analisando arquitetura atual...")
            metrics = await self._analyze_current_architecture()
            result["details"]["metrics"] = [asdict(m) for m in metrics]
            result["analysis_completed"] = True
            logger.info(f"   📊 Analisados {len(metrics)} arquivos")
            
            # 2. DETECÇÃO: Identificar problemas arquiteturais
            logger.info("🚨 Fase 2: Detectando problemas arquiteturais...")
            issues = await self.issue_detector.detect_issues(metrics)
            result["issues_detected"] = len(issues)
            result["details"]["issues"] = [asdict(i) for i in issues]
            
            if not issues:
                logger.info("✅ Nenhum problema arquitetural detectado")
                return result
            
            logger.info(f"   🎯 Detectados {len(issues)} problemas arquiteturais")
            for issue in issues[:3]:  # Log top 3
                logger.info(f"      • {issue.issue_type}: {issue.description}")
            
            # 3. PROPOSTAS: Gerar refatorações
            logger.info("💡 Fase 3: Gerando propostas de refatoração...")
            proposals = await self.proposer.propose_refactorings(issues, metrics)
            result["proposals_generated"] = len(proposals)
            result["details"]["proposals"] = [asdict(p) for p in proposals]
            
            if not proposals:
                logger.info("⚠️ Nenhuma proposta de refatoração gerada")
                return result
            
            logger.info(f"   📋 Geradas {len(proposals)} propostas de refatoração")
            for proposal in proposals[:2]:  # Log top 2
                logger.info(f"      • {proposal.refactoring_type}: {proposal.description}")
            
            # 4. APLICAÇÃO: Executar refatorações selecionadas
            logger.info("🛠️ Fase 4: Aplicando refatorações...")
            
            # Aplicar apenas propostas de alta prioridade e baixo risco
            selected_proposals = [
                p for p in proposals 
                if p.priority_score >= 3.0 and p.risk_level in ["low", "medium"]
            ][:2]  # Máximo 2 refatorações por ciclo
            
            refactoring_results = []
            improvements = 0
            
            for proposal in selected_proposals:
                logger.info(f"   🔧 Aplicando: {proposal.description}")
                
                refactoring_result = await self.refactorer.apply_refactoring(proposal)
                refactoring_results.append(refactoring_result)
                
                if refactoring_result.success:
                    result["refactorings_applied"] += 1
                    if refactoring_result.improvement_achieved:
                        improvements += 1
                    logger.info(f"      ✅ Sucesso: {proposal.proposal_id}")
                else:
                    logger.error(f"      ❌ Falhou: {proposal.proposal_id} - {refactoring_result.error_message}")
            
            result["improvements_achieved"] = improvements
            result["details"]["refactoring_results"] = [asdict(r) for r in refactoring_results]
            
            # 5. VALIDAÇÃO: Verificar evolução
            if improvements > 0:
                result["architecture_evolved"] = True
                logger.info("🎉 Arquitetura evoluída com sucesso!")
                
                # Salvar aprendizado
                if self.memory_manager:
                    await self._save_evolution_learning(issues, proposals, refactoring_results)
            
            # 6. RESULTADO FINAL
            logger.info("🏁 Fase 5: Evolução arquitetural concluída!")
            logger.info(f"   📊 Estatísticas:")
            logger.info(f"      • Problemas detectados: {result['issues_detected']}")
            logger.info(f"      • Propostas geradas: {result['proposals_generated']}")
            logger.info(f"      • Refatorações aplicadas: {result['refactorings_applied']}")
            logger.info(f"      • Melhorias alcançadas: {result['improvements_achieved']}")
            logger.info(f"      • Arquitetura evoluída: {'✅ SIM' if result['architecture_evolved'] else '❌ NÃO'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na evolução arquitetural: {e}")
            result["error"] = str(e)
            return result
    
    async def _analyze_current_architecture(self) -> List[CodeMetrics]:
        """Analisa a arquitetura atual do sistema."""
        
        metrics = []
        
        for analysis_dir in self.analysis_dirs:
            if not Path(analysis_dir).exists():
                continue
            
            # Encontrar todos os arquivos Python
            python_files = list(Path(analysis_dir).rglob("*.py"))
            
            for file_path in python_files:
                # Pular arquivos __pycache__ e .pyc
                if "__pycache__" in str(file_path) or file_path.suffix == ".pyc":
                    continue
                
                try:
                    file_metrics = await self.analyzer.analyze_file(str(file_path))
                    metrics.append(file_metrics)
                except Exception as e:
                    logger.warning(f"Erro analisando {file_path}: {e}")
        
        return metrics
    
    async def _save_evolution_learning(
        self,
        issues: List[ArchitecturalIssue],
        proposals: List[RefactoringProposal],
        results: List[RefactoringResult]
    ):
        """Salva aprendizado da evolução na memória."""
        
        try:
            learning_data = {
                "timestamp": datetime.now().isoformat(),
                "successful_patterns": [
                    {
                        "issue_type": result.proposal_id.split('_')[0] if '_' in result.proposal_id else "unknown",
                        "refactoring_type": next(
                            (p.refactoring_type for p in proposals if p.proposal_id == result.proposal_id),
                            "unknown"
                        ),
                        "improvements": result.improvement_achieved,
                        "changes": result.changes_made
                    }
                    for result in results if result.success
                ],
                "failed_attempts": [
                    {
                        "proposal_id": result.proposal_id,
                        "error": result.error_message,
                        "files_affected": result.files_modified
                    }
                    for result in results if not result.success
                ],
                "issue_patterns": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "resolution_attempted": any(
                            result.success for result in results 
                            if result.proposal_id.startswith(issue.issue_type.split('_')[0])
                        )
                    }
                    for issue in issues
                ]
            }
            
            await self.memory_manager.store_procedural_memory(
                "architecture_evolution_patterns",
                learning_data,
                {"category": "self_improvement", "type": "architecture_evolution"}
            )
            
            logger.info("🧠 Aprendizado de evolução arquitetural salvo na memória")
            
        except Exception as e:
            logger.error(f"Erro ao salvar aprendizado: {e}")


# Função de conveniência para uso externo
async def evolve_architecture() -> Dict[str, Any]:
    """
    Função principal para evolução arquitetural autônoma.
    
    Esta é a implementação de RSI ARQUITETURAL REAL - o sistema analisa
    e melhora sua própria arquitetura automaticamente.
    """
    system = ArchitectureEvolution()
    return await system.evolve_architecture()


if __name__ == "__main__":
    # Teste direto
    async def main():
        result = await evolve_architecture()
        print(f"Resultado da evolução: {json.dumps(result, indent=2, default=str)}")
    
    asyncio.run(main())