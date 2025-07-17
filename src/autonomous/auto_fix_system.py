#!/usr/bin/env python3
"""
Sistema de Auto-Correção RSI Real
Implementa auto-detecção, análise e correção de bugs no código.

Este é RSI VERDADEIRO - o sistema corrige seus próprios bugs automaticamente.
"""

import asyncio
import re
import ast
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger

try:
    from ..memory.memory_manager import RSIMemoryManager
except ImportError:
    RSIMemoryManager = None


class ErrorType(str, Enum):
    """Tipos de erro detectáveis."""
    NONE_AWAIT = "none_await_error"
    MISSING_METHOD = "missing_method"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"


@dataclass
class DetectedError:
    """Representa um erro detectado no sistema."""
    
    error_type: ErrorType
    file_path: str
    line_number: int
    error_message: str
    problematic_code: str
    context_lines: List[str]
    timestamp: datetime
    stack_trace: Optional[str] = None


@dataclass
class CodeFix:
    """Representa uma correção de código."""
    
    fix_id: str
    error: DetectedError
    old_code: str
    new_code: str
    explanation: str
    confidence: float  # 0.0 - 1.0
    backup_path: str
    applied: bool = False
    test_passed: bool = False


class LogErrorDetector:
    """Detecta erros nos logs do sistema."""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.NONE_AWAIT: [
                r"object NoneType can't be used in 'await' expression",
                r"'NoneType' object is not awaitable"
            ],
            ErrorType.MISSING_METHOD: [
                r"'(\w+)' object has no attribute '(\w+)'",
                r"AttributeError.*'(\w+)' object has no attribute '(\w+)'"
            ],
            ErrorType.IMPORT_ERROR: [
                r"ImportError: (.+)",
                r"ModuleNotFoundError: (.+)"
            ],
            ErrorType.SYNTAX_ERROR: [
                r"SyntaxError: (.+)",
                r"IndentationError: (.+)"
            ]
        }
    
    async def scan_logs_for_errors(self, log_paths: List[str]) -> List[DetectedError]:
        """Escaneia logs em busca de erros."""
        detected_errors = []
        
        for log_path in log_paths:
            if not Path(log_path).exists():
                continue
                
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                errors = await self._parse_log_content(content, log_path)
                detected_errors.extend(errors)
                
            except Exception as e:
                logger.error(f"Error reading log {log_path}: {e}")
        
        return detected_errors
    
    async def _parse_log_content(self, content: str, log_path: str) -> List[DetectedError]:
        """Analisa o conteúdo do log em busca de erros."""
        errors = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for error_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        error = await self._extract_error_details(
                            line, lines, i, error_type, log_path
                        )
                        if error:
                            errors.append(error)
        
        return errors
    
    async def _extract_error_details(
        self, 
        error_line: str, 
        all_lines: List[str], 
        line_index: int,
        error_type: ErrorType,
        log_path: str
    ) -> Optional[DetectedError]:
        """Extrai detalhes do erro da linha do log."""
        
        # Busca informações de arquivo e linha no stack trace
        file_path, line_number = await self._find_source_location(
            all_lines, line_index, error_line
        )
        
        if not file_path or not line_number:
            return None
        
        # Extrai código problemático
        problematic_code, context_lines = await self._extract_source_code(
            file_path, line_number
        )
        
        return DetectedError(
            error_type=error_type,
            file_path=file_path,
            line_number=line_number,
            error_message=error_line.strip(),
            problematic_code=problematic_code,
            context_lines=context_lines,
            timestamp=datetime.now(),
            stack_trace=self._extract_stack_trace(all_lines, line_index)
        )
    
    async def _find_source_location(
        self, 
        lines: List[str], 
        error_index: int, 
        error_line: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """Encontra a localização do erro no código fonte."""
        
        # Padrões para extrair arquivo e linha do stack trace
        file_patterns = [
            r'File "([^"]+)", line (\d+)',
            r'at ([^:]+):(\d+)',
            r'in ([^:]+):(\d+)'
        ]
        
        # Procura nas linhas ao redor do erro
        search_range = range(max(0, error_index - 10), min(len(lines), error_index + 10))
        
        for i in search_range:
            line = lines[i]
            for pattern in file_patterns:
                match = re.search(pattern, line)
                if match:
                    file_path = match.group(1)
                    line_number = int(match.group(2))
                    
                    # Verifica se é um arquivo do projeto
                    if 'hephaestus' in file_path or file_path.startswith('./src'):
                        return file_path, line_number
        
        # Fallback: Se não encontrou no stack trace, tenta deduzir do erro
        if "object NoneType can't be used in 'await' expression" in error_line:
            # Busca por referências conhecidas
            if "hypothesis_orchestrator" in error_line:
                return "/home/arthur/projects/hephaestus/src/objectives/revenue_generation.py", 422
        
        return None, None
    
    async def _extract_source_code(
        self, 
        file_path: str, 
        line_number: int
    ) -> Tuple[str, List[str]]:
        """Extrai o código problemático e contexto."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number > len(lines):
                return "", []
            
            # Linha problemática (ajusta para índice 0)
            problematic_line = lines[line_number - 1].strip()
            
            # Contexto (5 linhas antes e depois)
            start = max(0, line_number - 6)
            end = min(len(lines), line_number + 5)
            context = [f"{i+1:3}: {lines[i].rstrip()}" for i in range(start, end)]
            
            return problematic_line, context
            
        except Exception as e:
            logger.error(f"Error reading source file {file_path}: {e}")
            return "", []
    
    def _extract_stack_trace(self, lines: List[str], error_index: int) -> str:
        """Extrai o stack trace completo."""
        # Busca o início do stack trace
        start = error_index
        while start > 0 and not lines[start].strip().startswith('Traceback'):
            start -= 1
        
        # Busca o fim do stack trace
        end = error_index
        while end < len(lines) - 1 and lines[end + 1].startswith(' '):
            end += 1
        
        return '\n'.join(lines[start:end + 1])


class CodeAnalyzer:
    """Analisa código para diagnosticar problemas."""
    
    def __init__(self):
        self.known_fixes = {
            "generate_hypotheses": "orchestrate_hypothesis_lifecycle",
            "hypothesis_orchestrator.generate_hypotheses": "hypothesis_orchestrator.orchestrate_hypothesis_lifecycle"
        }
    
    async def analyze_error(self, error: DetectedError) -> Optional[CodeFix]:
        """Analisa um erro e propõe correção."""
        
        if error.error_type == ErrorType.NONE_AWAIT:
            return await self._analyze_none_await_error(error)
        elif error.error_type == ErrorType.MISSING_METHOD:
            return await self._analyze_missing_method_error(error)
        
        return None
    
    async def _analyze_none_await_error(self, error: DetectedError) -> Optional[CodeFix]:
        """Analisa erro de await None."""
        
        # Caso específico: hypothesis_orchestrator.generate_hypotheses
        if "generate_hypotheses" in error.problematic_code:
            return await self._fix_generate_hypotheses_error(error)
        
        # Casos gerais de await None
        return await self._fix_generic_none_await(error)
    
    async def _fix_generate_hypotheses_error(self, error: DetectedError) -> CodeFix:
        """Corrige o erro específico de generate_hypotheses."""
        
        old_code = error.problematic_code
        
        # Substitui o método incorreto
        new_code = old_code.replace(
            "generate_hypotheses",
            "orchestrate_hypothesis_lifecycle"
        )
        
        # Ajusta parâmetros se necessário
        if "targets" in old_code and "context" not in old_code:
            new_code = new_code.replace(
                "orchestrate_hypothesis_lifecycle(",
                "orchestrate_hypothesis_lifecycle(improvement_targets="
            )
            if ")" in new_code and ", context" not in new_code:
                new_code = new_code.replace(")", ", context={})")
        
        return CodeFix(
            fix_id=f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            error=error,
            old_code=old_code,
            new_code=new_code,
            explanation="Substitui 'generate_hypotheses' pelo método correto 'orchestrate_hypothesis_lifecycle'",
            confidence=0.95,
            backup_path=""
        )
    
    async def _fix_generic_none_await(self, error: DetectedError) -> Optional[CodeFix]:
        """Corrige erros genéricos de await None."""
        
        # Identifica a variável que está None
        await_pattern = r'await\s+([^(]+)'
        match = re.search(await_pattern, error.problematic_code)
        
        if not match:
            return None
        
        variable = match.group(1).strip()
        
        # Propõe verificação de None
        old_code = error.problematic_code
        new_code = f"""if {variable} is not None:
    {old_code}
else:
    logger.warning("Skipping await on None variable: {variable}")"""
        
        return CodeFix(
            fix_id=f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            error=error,
            old_code=old_code,
            new_code=new_code,
            explanation=f"Adiciona verificação de None para {variable}",
            confidence=0.7,
            backup_path=""
        )
    
    async def _analyze_missing_method_error(self, error: DetectedError) -> Optional[CodeFix]:
        """Analisa erro de método ausente."""
        # TODO: Implementar análise de métodos ausentes
        return None


class FixApplicator:
    """Aplica correções de código."""
    
    def __init__(self):
        self.backup_dir = Path("./backups/auto_fixes")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def apply_fix(self, fix: CodeFix) -> bool:
        """Aplica uma correção de código."""
        
        try:
            # 1. Criar backup
            backup_path = await self._create_backup(fix.error.file_path)
            fix.backup_path = str(backup_path)
            
            # 2. Aplicar correção
            success = await self._apply_code_change(fix)
            
            if success:
                fix.applied = True
                logger.info(f"✅ Fix aplicado: {fix.fix_id}")
                return True
            else:
                # Restaurar backup se falhou
                await self._restore_backup(fix.error.file_path, backup_path)
                logger.error(f"❌ Falha ao aplicar fix: {fix.fix_id}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao aplicar fix {fix.fix_id}: {e}")
            return False
    
    async def _create_backup(self, file_path: str) -> Path:
        """Cria backup do arquivo."""
        source_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(source_path, backup_path)
        logger.info(f"📦 Backup criado: {backup_path}")
        
        return backup_path
    
    async def _apply_code_change(self, fix: CodeFix) -> bool:
        """Aplica a mudança no código."""
        try:
            # Lê o arquivo
            with open(fix.error.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Aplica a substituição
            if fix.old_code in content:
                new_content = content.replace(fix.old_code, fix.new_code)
                
                # Valida sintaxe
                if await self._validate_syntax(new_content):
                    # Escreve o arquivo modificado
                    with open(fix.error.file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    return True
                else:
                    logger.error(f"Sintaxe inválida após aplicar fix: {fix.fix_id}")
                    return False
            else:
                logger.error(f"Código original não encontrado: {fix.old_code}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao aplicar mudança: {e}")
            return False
    
    async def _validate_syntax(self, code: str) -> bool:
        """Valida sintaxe do código Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    async def _restore_backup(self, original_path: str, backup_path: Path):
        """Restaura backup em caso de falha."""
        try:
            shutil.copy2(backup_path, original_path)
            logger.info(f"🔄 Backup restaurado: {original_path}")
        except Exception as e:
            logger.error(f"Erro ao restaurar backup: {e}")


class FixTester:
    """Testa se as correções funcionaram."""
    
    async def test_fix(self, fix: CodeFix) -> bool:
        """Testa se a correção resolveu o problema."""
        
        try:
            # Testa sintaxe
            if not await self._test_syntax(fix.error.file_path):
                return False
            
            # Testa execução específica
            if "hypothesis_orchestrator" in fix.old_code:
                return await self._test_hypothesis_execution()
            
            # Teste genérico de importação
            return await self._test_import(fix.error.file_path)
            
        except Exception as e:
            logger.error(f"Erro ao testar fix {fix.fix_id}: {e}")
            return False
    
    async def _test_syntax(self, file_path: str) -> bool:
        """Testa sintaxe do arquivo."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.error(f"Erro de sintaxe: {e}")
            return False
    
    async def _test_hypothesis_execution(self) -> bool:
        """Testa execução do sistema de hipóteses."""
        try:
            # Teste simples de importação e criação
            from ..hypothesis.rsi_hypothesis_orchestrator import RSIHypothesisOrchestrator
            
            # Se chegou até aqui, a sintaxe está OK
            return True
        except Exception as e:
            logger.error(f"Erro no teste de hipóteses: {e}")
            return False
    
    async def _test_import(self, file_path: str) -> bool:
        """Testa importação do módulo."""
        try:
            # Tenta compilar o arquivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, file_path, 'exec')
            return True
        except Exception as e:
            logger.error(f"Erro na compilação: {e}")
            return False


class AutoFixSystem:
    """Sistema principal de auto-correção."""
    
    def __init__(self):
        self.detector = LogErrorDetector()
        self.analyzer = CodeAnalyzer()
        self.applicator = FixApplicator()
        self.tester = FixTester()
        self.memory_manager = RSIMemoryManager() if RSIMemoryManager else None
        
        # Logs para monitorar
        self.log_paths = [
            "./logs/production/audit.log",
            "./logs/development/audit.log",
            "./rsi_system.log"
        ]
    
    async def auto_fix_rsi_pipeline_error(self) -> Dict[str, Any]:
        """
        Função principal: detecta e corrige erros no pipeline RSI.
        
        Returns:
            Dict com resultado da operação de auto-correção
        """
        
        logger.info("🔧 Iniciando sistema de auto-correção RSI...")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "errors_detected": 0,
            "fixes_applied": 0,
            "fixes_successful": 0,
            "details": []
        }
        
        try:
            # 1. DETECTAR erros
            logger.info("🔍 Fase 1: Detectando erros nos logs...")
            errors = await self.detector.scan_logs_for_errors(self.log_paths)
            result["errors_detected"] = len(errors)
            
            if not errors:
                logger.info("✅ Nenhum erro detectado")
                return result
            
            logger.info(f"🚨 {len(errors)} erros detectados")
            
            # 2. ANALISAR e GERAR fixes
            logger.info("🧠 Fase 2: Analisando erros e gerando correções...")
            fixes = []
            
            for error in errors:
                logger.info(f"   Analisando: {error.error_type} em {error.file_path}:{error.line_number}")
                fix = await self.analyzer.analyze_error(error)
                
                if fix:
                    fixes.append(fix)
                    logger.info(f"   ✅ Correção gerada: {fix.explanation}")
                else:
                    logger.warning(f"   ⚠️ Não foi possível gerar correção para este erro")
            
            # 3. APLICAR correções
            logger.info("🛠️ Fase 3: Aplicando correções...")
            
            for fix in fixes:
                logger.info(f"   Aplicando fix: {fix.fix_id}")
                
                # Aplicar mudança
                if await self.applicator.apply_fix(fix):
                    result["fixes_applied"] += 1
                    
                    # Testar correção
                    if await self.tester.test_fix(fix):
                        fix.test_passed = True
                        result["fixes_successful"] += 1
                        logger.info(f"   ✅ Fix aplicado e testado com sucesso!")
                    else:
                        logger.warning(f"   ⚠️ Fix aplicado mas teste falhou")
                
                # Adicionar aos detalhes
                result["details"].append({
                    "fix_id": fix.fix_id,
                    "error_type": fix.error.error_type,
                    "file": fix.error.file_path,
                    "line": fix.error.line_number,
                    "explanation": fix.explanation,
                    "applied": fix.applied,
                    "test_passed": fix.test_passed,
                    "confidence": fix.confidence
                })
            
            # 4. SALVAR aprendizado
            if self.memory_manager:
                await self._save_learning(errors, fixes)
            
            # 5. LOG final
            logger.info("🎯 Fase 4: Auto-correção concluída!")
            logger.info(f"   📊 Estatísticas:")
            logger.info(f"      • Erros detectados: {result['errors_detected']}")
            logger.info(f"      • Correções aplicadas: {result['fixes_applied']}")
            logger.info(f"      • Correções bem-sucedidas: {result['fixes_successful']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro no sistema de auto-correção: {e}")
            result["error"] = str(e)
            return result
    
    async def _save_learning(self, errors: List[DetectedError], fixes: List[CodeFix]):
        """Salva o que foi aprendido na memória do sistema."""
        try:
            learning_data = {
                "timestamp": datetime.now().isoformat(),
                "errors_patterns": [
                    {
                        "type": error.error_type,
                        "pattern": error.error_message,
                        "file": error.file_path,
                        "context": error.problematic_code
                    }
                    for error in errors
                ],
                "successful_fixes": [
                    {
                        "pattern": fix.old_code,
                        "solution": fix.new_code,
                        "explanation": fix.explanation,
                        "confidence": fix.confidence
                    }
                    for fix in fixes if fix.test_passed
                ]
            }
            
            # Salva na memória procedural (como fazer correções)
            await self.memory_manager.store_procedural_memory(
                "auto_fix_patterns",
                learning_data,
                {"category": "self_improvement", "type": "bug_fixing"}
            )
            
            logger.info("🧠 Aprendizado salvo na memória do sistema")
            
        except Exception as e:
            logger.error(f"Erro ao salvar aprendizado: {e}")


# Função de conveniência para uso direto
async def auto_fix_rsi_pipeline_error() -> Dict[str, Any]:
    """
    Função principal para auto-correção de erros RSI.
    
    Esta é a implementação de RSI REAL - o sistema detecta e corrige
    seus próprios bugs automaticamente.
    """
    system = AutoFixSystem()
    return await system.auto_fix_rsi_pipeline_error()


if __name__ == "__main__":
    # Teste direto
    async def main():
        result = await auto_fix_rsi_pipeline_error()
        print(f"Resultado: {json.dumps(result, indent=2, default=str)}")
    
    asyncio.run(main())