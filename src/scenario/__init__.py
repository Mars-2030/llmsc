# src/scenario/__init__.py

"""
Scenario generation module for the pandemic supply chain simulation.
Includes the SIR model based epidemic curve generator and disruption generator.
"""

from .generator import PandemicScenarioGenerator

__all__ = [
    "PandemicScenarioGenerator",
]