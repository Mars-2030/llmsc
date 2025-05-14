# src/agents/__init__.py

"""
Initializes the agents module and exports agent creation factory functions.
These factories are designed to be compatible with the SimPy-based simulation environment.
"""

from .manufacturer import create_openai_manufacturer_agent, ManufacturerAgentLG
from .distributor import create_openai_distributor_agent, DistributorAgentLG
from .hospital import create_openai_hospital_agent, HospitalAgentLG

# Other potential imports from this package if you add more agent types or base classes later
# from .base import BaseAgent # Example if you had a common base class for agents

__all__ = [
    "create_openai_manufacturer_agent",
    "create_openai_distributor_agent",
    "create_openai_hospital_agent",
    "ManufacturerAgentLG",    # Exporting the class itself can be useful for type hinting or direct instantiation
    "DistributorAgentLG",
    "HospitalAgentLG",
    # Add other agent creation functions or agent classes here if you have more
]

# You can also add a log message here if you want to confirm the module is loaded,
# though it's generally not common for __init__.py files unless for specific debugging.
# print("Agents module initialized.")