"""
Objectives module for Hephaestus RSI.

This module contains autonomous objective-driven systems that use
RSI capabilities to achieve specific goals.
"""

from .revenue_generation import (
    AutonomousRevenueGenerator,
    RevenueStrategy,
    RevenueOpportunity,
    RevenueProject,
    get_revenue_generator
)

__all__ = [
    'AutonomousRevenueGenerator',
    'RevenueStrategy',
    'RevenueOpportunity', 
    'RevenueProject',
    'get_revenue_generator'
]