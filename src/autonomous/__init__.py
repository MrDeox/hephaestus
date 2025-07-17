"""
Autonomous Systems - Sistemas Autônomos de Auto-Melhoria

Este módulo contém sistemas que permitem ao RSI detectar, analisar e corrigir
seus próprios problemas automaticamente, incluindo evolução arquitetural.
"""

from .auto_fix_system import auto_fix_rsi_pipeline_error, AutoFixSystem
from .architecture_evolution import evolve_architecture, ArchitectureEvolution

__all__ = [
    "auto_fix_rsi_pipeline_error",
    "AutoFixSystem",
    "evolve_architecture", 
    "ArchitectureEvolution"
]