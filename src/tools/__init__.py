# --- START OF COMPLETE src/tools/__init__.py ---
import json
from typing import Dict, List, Optional

# Import core tool functions from other modules in the package
from .forecasting import epidemic_forecast_tool, disruption_prediction_tool
from .allocation import allocation_priority_tool, optimal_order_quantity_tool
from .assessment import criticality_assessment_tool
from .production import calculate_target_production_quantities_tool # Import the production tool

# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError: # pragma: no cover
    BlockchainInterface = None # Define as None if import fails


class PandemicSupplyChainTools:
    """
    Collection of decision-making tools for supply chain agents.
    Provides static methods for tool execution and OpenAI schema definitions.
    """

    # --- Tool Execution Methods (Static - ensuring all tools are represented) ---
    @staticmethod
    def epidemic_forecast_tool(*args, **kwargs):
        return epidemic_forecast_tool(*args, **kwargs) # pragma: no cover
    @staticmethod
    def disruption_prediction_tool(*args, **kwargs):
        return disruption_prediction_tool(*args, **kwargs) # pragma: no cover
    @staticmethod
    def allocation_priority_tool(*args, **kwargs):
        return allocation_priority_tool(*args, **kwargs) # pragma: no cover
    @staticmethod
    def optimal_order_quantity_tool(*args, **kwargs):
        return optimal_order_quantity_tool(*args, **kwargs) # pragma: no cover
    @staticmethod
    def criticality_assessment_tool(*args, **kwargs):
        return criticality_assessment_tool(*args, **kwargs) # pragma: no cover
    @staticmethod
    def get_blockchain_regional_cases_tool(blockchain_interface: Optional[BlockchainInterface], num_regions: int) -> Optional[Dict[int, int]]: # pragma: no cover
        if blockchain_interface is None: return None
        if not isinstance(num_regions, int) or num_regions <= 0: return {} # Basic validation
        regional_cases = {}
        for region_id in range(num_regions):
            case_count = blockchain_interface.get_regional_case_count(region_id)
            regional_cases[region_id] = case_count if case_count is not None else 0
        return regional_cases
    @staticmethod
    def calculate_target_production_quantities_tool(*args, **kwargs): # New static method for production tool
        return calculate_target_production_quantities_tool(*args, **kwargs) # pragma: no cover


    # --- OpenAI Tool Schema Definitions ---
    @staticmethod
    def get_openai_tool_definitions() -> List[Dict]:
        """Returns a list of tool definitions in OpenAI function calling format."""
        tools_def = [
            # --- Tool 1: calculate_target_production_quantities (ALIGNED SCHEMA) ---
              {
                "type": "function",
                "function": {
                    "name": "calculate_target_production_quantities",
                    "description": "Manufacturer Only: Calculates dynamic production quantities. Provide core inventory/forecast dicts, a subset of other observations, and optional tuning parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "current_inventories": {
                                "type": "object", "additionalProperties": {"type": "number"},
                                "description": "Dict {drug_id_str: qty} of current usable inventories."
                            },
                            "pending_releases_soon": {
                                "type": "object", "additionalProperties": {"type": "number"},
                                "description": "Dict {drug_id_str: qty} of sum of pending warehouse releases expected soon."
                            },
                            "warehouse_inventories": {
                                "type": "object", "additionalProperties": {"type": "number"},
                                "description": "Dict {drug_id_str: qty} of current warehouse inventories."
                            },
                            "sum_recent_orders": {
                                "type": "object", "additionalProperties": {"type": "number"},
                                "description": "Dict {drug_id_str: total_qty} of sum of recent orders from distributors."
                            },
                            "total_downstream_demand_forecast_list": {
                                "type": "object",
                                "additionalProperties": {"type": "array", "items": {"type": "number"}},
                                "description": "Dict {drug_id_str: [forecast_list]} of total downstream demand forecasts."
                            },
                            "any_downstream_stockouts": {
                                "type": "object", "additionalProperties": {"type": "boolean"},
                                "description": "Dict {drug_id_str: True/False} indicating recent downstream stockouts."
                            },
                            "observation_subset": {
                                "type": "object",
                                "properties": {
                                    "production_capacity": {
                                        "type": "object", "additionalProperties": {"type": "number"},
                                        "description": "Dict {drug_id_str: capacity_float} of daily production capacities. Extract from obs_clean.production_capacity."
                                    },
                                    "drug_info": { 
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "object",
                                            "properties": {
                                                "demand_factor": {"type": "number"},
                                                "criticality_value": {"type": "integer"}
                                            },
                                        },
                                        "description": "Full drug_info dictionary from obs_clean.drug_info: {drug_id_str: {name: ..., demand_factor: ..., criticality_value: ...}}."
                                    }
                                },
                                "required": ["production_capacity", "drug_info"], 
                                "description": "A dictionary containing specific data points extracted from the main observation, like production_capacity and full drug_info."
                            },
                            "target_usable_inv_cover_days": {"type": "integer", "description": "Optional. Target days of supply for usable inventory (default: 7)."},
                            "target_warehouse_inv_cover_days": {"type": "integer", "description": "Optional. Target days for warehouse inventory (default: 5)."},
                            "forecast_horizon_for_signal": {"type": "integer", "description": "Optional. Horizon for forecast signal (default: 5)."},
                            "signal_percentile": {"type": "number", "description": "Optional. Percentile for forecast signal (default: 90.0)."},
                            "stockout_boost_factor": {"type": "number", "description": "Optional. Multiplier during stockouts (default: 1.7)."},
                            # Other optional args from production tool schema
                            "startup_factor_cap": {"type": "number", "description": "Optional. Max production as % of capacity on day 0 (default: 0.3)."},
                            "startup_demand_cover_days": {"type": "integer", "description": "Optional. Days of demand to cover with startup production (default: 3)."},
                            "min_heartbeat_production_factor_df": {"type": "number", "description": "Optional. Min production as multiple of drug's base demand factor (default: 10.0)."},
                            "min_heartbeat_production_factor_cap": {"type": "number", "description": "Optional. Min production as % of capacity (default: 0.05)."},
                            "warehouse_shortfall_fulfillment_factor": {"type": "number", "description": "Optional. How much of warehouse target shortfall to produce (default: 0.5)."}
                        },
                        "required": [ 
                            "current_inventories",
                            "pending_releases_soon",
                            "warehouse_inventories",
                            "sum_recent_orders",
                            "total_downstream_demand_forecast_list",
                            "any_downstream_stockouts",
                            "observation_subset" 
                        ]
                    }
                }
            },
            # --- Tool 2: Blockchain Query Tool ---
            {
                "type": "function",
                "function": {
                    "name": "get_blockchain_regional_cases",
                    "description": "Manufacturer Only: Fetches latest trusted regional case counts from the blockchain for prioritizing allocation.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            # --- Tool 3: Forecasting Tool ---
            {
                "type": "function",
                "function": {
                    "name": "epidemic_forecast",
                    "description": "Forecasts epidemic progression based on case history. LLM must provide forecast_horizon. Current cases and history are derived from observation by the tool executor.",
                     "parameters": {
                        "type": "object",
                        "properties": {
                            "forecast_horizon": {
                                "type": "integer", 
                                "description": "Number of days to forecast ahead (e.g., 7)."
                            }
                            # No need for LLM to pass current_cases or case_history; execute_tool will fetch these.
                        },
                        "required": ["forecast_horizon"]
                     }
                }
            },
            # --- Tool 4: Disruption Prediction Tool ---
             {
                "type": "function",
                "function": {
                    "name": "predict_disruptions",
                    "description": "Predicts probability of manufacturing or transportation disruptions. LLM provides look_ahead_days. Historical disruptions and current day are derived from observation by the tool executor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "look_ahead_days": {
                                "type": "integer", 
                                "description": "Number of days ahead to predict risk (e.g., 7)."
                            }
                            # No need for LLM to pass historical_disruptions or current_day.
                        },
                        "required": ["look_ahead_days"]
                    }
                }
            },

            # --- Tool 5: Criticality Assessment Tool (SCHEMA MODIFIED) ---
            {
                "type": "function",
                "function": {
                    "name": "get_criticality_assessment",
                    "description": "Assesses supply criticality for a specific drug ID. LLM provides drug_id. Other data (drug_info, histories, patient_impact) are derived from observation by the tool executor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                           "drug_id": {
                               "type": "integer", 
                               "description": "The ID of the drug to assess."
                           },
                           # Make regional_patient_impact_score OPTIONAL for the LLM to provide.
                           # execute_tool will attempt to fetch it from observation regardless.
                           # If LLM provides it, that value could potentially override or supplement.
                           # For simplicity, it's often better if execute_tool is the sole source from observation.
                           # However, adding it here makes the LLM aware of its existence.
                           "regional_patient_impact_score": {
                               "type": "number",
                               "description": "Optional. If known, the specific regional patient impact score to consider. If not provided, the tool executor will use the relevant score from the observation if available."
                           }
                        },
                        "required": ["drug_id"] 
                    }
                }
            },

            # --- Tool 6: Optimal Order Tool ---
             {
                "type": "function",
                "function": {
                    "name": "calculate_optimal_order",
                    "description": "Calculates a recommended order quantity for a drug. LLM provides drug_id. Inventory, pipeline, and forecast data are derived from observation by the tool executor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                           "drug_id": {
                               "type": "integer", 
                               "description": "The ID of the drug to calculate order for."
                           }
                           # No need for LLM to pass inventory, pipeline, forecast, lead_time, safety_stock_factor.
                        },
                        "required": ["drug_id"]
                    }
                }
            },
            # --- Tool 7: Allocation Priority Tool ---
             {
                "type": "function",
                "function": {
                    "name": "calculate_allocation_priority",
                    "description": "Manufacturer Only: Calculates fair allocation of limited inventory. LLM provides drug_id, region_requests, available_inventory. Region_cases can be from LLM or blockchain tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                           "drug_id": {"type": "integer", "description": "The ID of the drug being allocated."},
                           "region_requests": {
                               "type": "object",
                               "additionalProperties": {"type": "number"},
                               "description": "Dictionary mapping region_id (string or int) to requested/needed amount."
                           },
                           "available_inventory": {"type": "number", "description": "Amount available from usable manufacturer stock for allocation."},
                           "region_cases": { # LLM can provide this if it has a good source (e.g. after calling blockchain tool)
                               "type": "object",
                               "additionalProperties": {"type": "integer"},
                               "description": "(Optional) Dictionary mapping region_id (string or int) to case counts. If not provided, tool may use fallback or simpler logic.",
                           },
                           "drug_info": { # Added: Tool needs drug_info for criticality
                               "type": "object",
                               "description": "Required. The drug_info object for the specified drug_id, containing at least 'criticality_value'. Extract from obs_clean.drug_info[drug_id_str]."
                           }
                        },
                        "required": ["drug_id", "region_requests", "available_inventory", "drug_info"]
                    }
                }
            }
        ]
        return tools_def

# --- Export __all__ list ---
__all__ = [
    'PandemicSupplyChainTools',
    'epidemic_forecast_tool',
    'disruption_prediction_tool',
    'allocation_priority_tool',
    'optimal_order_quantity_tool',
    'criticality_assessment_tool',
    'calculate_target_production_quantities_tool', 
]
# --- END OF COMPLETE src/tools/__init__.py ---