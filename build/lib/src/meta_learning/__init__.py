"""
Meta-Learning Module - Sistema de Aprendizado de Segunda Ordem
Implementa conceitos avançados de RSI com feedback loops recursivos.
"""

from .gap_scanner import (
    GapScanner,
    Gap,
    GapType,
    GapSeverity,
    create_gap_scanner
)

from .mml_controller import (
    MMLController,
    LearningLevel,
    FeedbackType,
    LearningPattern,
    FeedbackLoop,
    create_mml_controller
)

__all__ = [
    # Gap Scanner
    'GapScanner',
    'Gap',
    'GapType', 
    'GapSeverity',
    'create_gap_scanner',
    
    # MML Controller
    'MMLController',
    'LearningLevel',
    'FeedbackType',
    'LearningPattern',
    'FeedbackLoop',
    'create_mml_controller'
]