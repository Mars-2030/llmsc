# src/simulation/helpers.py

"""
Helper functions for the SimPy-based pandemic supply chain simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
import simpy
from collections import defaultdict

from config import console as global_console, Colors # Use global console if specific one not passed
from src.scenario.generator import PandemicScenarioGenerator # For type hint

def create_simpy_stores(
    env: simpy.Environment,
    num_regions: int,
    num_drugs: int,
    initial_levels: Dict[str, Dict[Any, float]] # Expects calculated initial levels
) -> Dict[str, Dict[Any, simpy.Container]]:
    """
    Creates and returns dictionaries of SimPy Containers for inventories,
    initialized with provided levels.
    """
    stores = {
        "manufacturer_usable": {}, # {drug_id: Container}
        "warehouse": {},           # {drug_id: Container}
        "distributor": {},         # {(region_id, drug_id): Container}
        "hospital": {}             # {(region_id, drug_id): Container}
    }
    # Capacity is infinite unless specific limits are needed
    capacity = float('inf')
    console_to_use = global_console # Default to global console for this helper

    for drug_id in range(num_drugs):
        stores["manufacturer_usable"][drug_id] = simpy.Container(
            env, init=initial_levels["manufacturer_usable"].get(drug_id, 0.0), capacity=capacity
        )
        stores["warehouse"][drug_id] = simpy.Container(
            env, init=initial_levels["warehouse"].get(drug_id, 0.0), capacity=capacity
        )
        for region_id in range(num_regions):
            stores["distributor"][(region_id, drug_id)] = simpy.Container(
                env, init=initial_levels["distributor"].get((region_id, drug_id), 0.0), capacity=capacity
            )
            stores["hospital"][(region_id, drug_id)] = simpy.Container(
                env, init=initial_levels["hospital"].get((region_id, drug_id), 0.0), capacity=capacity
            )
    if console_to_use: console_to_use.print(f"[{Colors.SIMPY}]SimPy Container stores created and initialized.[/]")
    return stores


def initialize_simpy_stores( # Renamed to reflect it *calculates* levels
    scenario: PandemicScenarioGenerator
) -> Dict[str, Dict[Any, float]]:
    """
    Calculates initial inventory levels based on scenario parameters.
    Returns a dictionary of these levels for use in creating SimPy Containers.
    """
    num_regions = scenario.num_regions
    num_drugs = scenario.num_drugs
    console_to_use = scenario.console if hasattr(scenario, 'console') and scenario.console else global_console


    initial_levels = {
        "manufacturer_usable": defaultdict(float),
        "warehouse": defaultdict(float),
        "distributor": defaultdict(float), # Using (region_id, drug_id) as key
        "hospital": defaultdict(float)    # Using (region_id, drug_id) as key
    }

    # Configuration for initial stocking
    HOSPITAL_COVER_DAYS = 14
    DISTRIBUTOR_COVER_DAYS = 21
    MANUFACTURER_BUFFER_DAYS = 30 # Buffer on top of downstream needs
    WAREHOUSE_BUFFER_DAYS = 25
    MIN_ABS_HOSP_STOCK = 75.0
    MIN_ABS_DIST_STOCK = 150.0
    MIN_ABS_MANU_STOCK = 750.0
    MIN_ABS_WAREHOUSE_STOCK = 750.0
    # This scaling factor MUST match the one used in scenario.get_daily_drug_demand
    # if base_demand is "units per 100 cases".
    DEMAND_SCALING_FACTOR_FOR_INIT = 100.0

    initial_dist_needs_for_manu_calc = defaultdict(float) # {drug_id: total_dist_need}

    for drug_id in range(num_drugs):
        drug_info = scenario.drugs[drug_id]
        drug_base_demand_cfg = drug_info.get("base_demand", 1.0)
        # Calculate a minimum daily signal for stocking purposes
        min_daily_stock_signal = max(
            (drug_base_demand_cfg / DEMAND_SCALING_FACTOR_FOR_INIT) * (0.05 * DEMAND_SCALING_FACTOR_FOR_INIT), # 5% of its potential from 100 cases
            0.1 # Absolute minimum signal
        )

        total_stable_day0_downstream_demand_for_drug = 0.0

        for region_id in range(num_regions):
            day0_active_cases = 0
            if region_id in scenario.epidemic_curves and \
               scenario.epidemic_curves[region_id] is not None and \
               len(scenario.epidemic_curves[region_id]) > 0:
                day0_active_cases = max(0, scenario.epidemic_curves[region_id][0])

            expected_demand_day0 = (day0_active_cases * drug_base_demand_cfg / DEMAND_SCALING_FACTOR_FOR_INIT)
            stable_day0_hosp_proj_demand = max(expected_demand_day0, min_daily_stock_signal)
            if stable_day0_hosp_proj_demand < 0.01 and drug_base_demand_cfg > 0.01: # Ensure some stock if drug has usage
                stable_day0_hosp_proj_demand = max(stable_day0_hosp_proj_demand, drug_base_demand_cfg * 0.01) # e.g. 1% of base demand if cases are zero

            # Hospital Initial Inventory
            hosp_initial_inv = max(MIN_ABS_HOSP_STOCK, math.ceil(stable_day0_hosp_proj_demand * HOSPITAL_COVER_DAYS))
            initial_levels["hospital"][(region_id, drug_id)] = hosp_initial_inv

            # Distributor Initial Inventory
            dist_initial_inv = max(MIN_ABS_DIST_STOCK, math.ceil(stable_day0_hosp_proj_demand * DISTRIBUTOR_COVER_DAYS))
            initial_levels["distributor"][(region_id, drug_id)] = dist_initial_inv
            initial_dist_needs_for_manu_calc[drug_id] += dist_initial_inv

            total_stable_day0_downstream_demand_for_drug += stable_day0_hosp_proj_demand

        # Manufacturer Usable Stock
        manu_buffer_stock = math.ceil(total_stable_day0_downstream_demand_for_drug * MANUFACTURER_BUFFER_DAYS)
        manu_initial_inv = max(MIN_ABS_MANU_STOCK, initial_dist_needs_for_manu_calc[drug_id] + manu_buffer_stock)
        initial_levels["manufacturer_usable"][drug_id] = manu_initial_inv

        # Warehouse Stock
        warehouse_initial_inv = max(MIN_ABS_WAREHOUSE_STOCK, math.ceil(total_stable_day0_downstream_demand_for_drug * WAREHOUSE_BUFFER_DAYS))
        initial_levels["warehouse"][drug_id] = warehouse_initial_inv

    if console_to_use: console_to_use.print(f"[{Colors.SIMPY}]Calculated initial inventory levels for SimPy store creation.[/]")
    return initial_levels


def format_observation_for_agent(
    env: simpy.Environment,
    sim: 'PandemicSupplyChainSimulation', # Forward reference with quotes
    agent_type: str,
    agent_id_param: Any # region_id for dist/hosp, 0 for manu (or specific manufacturer ID if multiple)
) -> Dict:
    """
    Formats the observation for a given agent by querying SimPy stores and other sim states.
    This needs to replicate the structure expected by the agent's LLM prompt.
    """
    current_day = int(env.now)
    console_to_use = sim.console # Use the console from the simulation instance

    obs = {
        "day": current_day,
        "num_regions": sim.num_regions,
        "drug_info": {str(i): d for i, d in enumerate(sim.scenario.drugs)},
        "all_scenario_disruptions": sim.scenario.disruptions, # For predict_disruptions tool
         # Add patient impact scores
        "system_wide_patient_impact_summary": {
            str(r_id): sim.patient_impact_score.get(str(r_id), 0.0)
            for r_id in range(sim.num_regions)
        }
    }

    def get_in_transit_to_destination(
        destination_type_str_target: str,
        destination_entity_id_target: Optional[int]
    ) -> Dict[str, float]:
        in_transit = defaultdict(float)
        for transport_entry in sim.active_transports:
            if transport_entry["status"] == "in_transit" and \
               transport_entry["destination_type"] == destination_type_str_target and \
               (destination_entity_id_target is None or transport_entry["destination_entity_id"] == destination_entity_id_target):
                in_transit[str(transport_entry["drug_id"])] += transport_entry["quantity"]
        return dict(in_transit)

    # --- Epidemiological Data Helper ---
    def get_epi_data_for_region(r_id: int, current_day_obs: int) -> Dict:
        current_cases_region = sim.current_regional_cases.get(r_id,0)
        
        hist_cases_list = []
        if r_id in sim.scenario.epidemic_curves:
            curve = sim.scenario.epidemic_curves[r_id]
            # Ensure current_day_obs is within bounds for curve access
            # History up to and including current_day_obs
            start_idx = max(0, current_day_obs - 13) # e.g. last 14 days of data points if available
            end_idx = current_day_obs + 1            # Include current day's data point
            
            for day_idx in range(start_idx, end_idx):
                if day_idx < len(curve):
                    hist_cases_list.append(float(curve[day_idx]))
                # else: # If requesting future data beyond curve length, might append 0 or placeholder
                #    hist_cases_list.append(0.0) # Or handle as needed for trend calc
        
        # current_active_cases should be from sim.current_regional_cases for consistency with other sim logic
        # but hist_cases_list is derived directly from the scenario curve for forecasting tools.
        # For trend calculation, use the history list.
        case_change_1d = 0.0
        if len(hist_cases_list) >= 2:
            case_change_1d = hist_cases_list[-1] - hist_cases_list[-2]
        elif len(hist_cases_list) == 1: # Only current day's data available in hist_cases_list
            case_change_1d = 0.0 # No trend calculable

        trend_cat = "stable"
        # Use the current_cases_region from sim state for trend category, as it's what agents act on for demand
        current_cases_for_trend_check = float(sim.current_regional_cases.get(r_id, 0.0))
        if current_cases_for_trend_check > 10 and case_change_1d > current_cases_for_trend_check * 0.15: trend_cat = "increasing_strongly"
        elif case_change_1d > 0: trend_cat = "increasing"
        elif case_change_1d < 0: trend_cat = "decreasing"
        
        # Ensure current_active_cases in epi_data is the one from sim.current_regional_cases
        # as this is used for projected_demand.
        return {
            "projected_demand": {str(d): sim.current_regional_projected_demand[r_id][d] for d in range(sim.num_drugs)},
            "current_active_cases": int(sim.current_regional_cases.get(r_id, 0)), # From sim state
            "case_trend_category": trend_cat,
            "case_change_last_3d": 0, # Placeholder, more complex trend can be added
            "historical_active_cases_list": hist_cases_list # For epidemic_forecast tool
        }
    # --- End Epidemiological Data Helper ---

    if agent_type == "manufacturer":
        obs["agent_id_for_logging"] = agent_id_param # Should be 0
        obs["inventories"] = {str(d): sim.manu_usable_inv_stores[d].level for d in range(sim.num_drugs)}
        obs["warehouse_inventories"] = {str(d): sim.warehouse_inv_stores[d].level for d in range(sim.num_drugs)}
        obs["production_capacity"] = {str(d): sim.scenario.get_manufacturing_capacity(current_day, d) for d in range(sim.num_drugs)}

        obs["downstream_inventory_summary"] = defaultdict(lambda: defaultdict(float))
        obs["distributor_inventory_summary"] = defaultdict(lambda: defaultdict(float))
        obs["downstream_pipeline_summary"] = defaultdict(lambda: defaultdict(float)) # TODO: Populate this if needed by prompt
        obs["downstream_projected_demand_summary"] = defaultdict(float)
        obs["downstream_stockout_summary"] = {str(d): {str(r): sim.stockouts.get(str(d),{}).get(str(r),0) for r in range(sim.num_regions)} for d in range(sim.num_drugs)}

        obs["total_downstream_demand_forecast"] = defaultdict(list)
        obs["regional_hospital_demand_forecast"] = defaultdict(lambda: defaultdict(list))
        obs["epidemiological_data"] = {}

        for r_id in range(sim.num_regions):
            obs["epidemiological_data"][str(r_id)] = get_epi_data_for_region(r_id, current_day)
            for d_id in range(sim.num_drugs):
                obs["downstream_projected_demand_summary"][str(d_id)] += obs["epidemiological_data"][str(r_id)]["projected_demand"].get(str(d_id), 0.0)
                obs["distributor_inventory_summary"][str(d_id)][str(r_id)] = sim.dist_inv_stores[(r_id, d_id)].level
                obs["downstream_inventory_summary"][str(d_id)]["distributor"] += sim.dist_inv_stores[(r_id, d_id)].level
                obs["downstream_inventory_summary"][str(d_id)]["hospital"] += sim.hosp_inv_stores[(r_id, d_id)].level

                for day_offset_prod in range(1, 8): # 7-day forecast starting from next day
                    fcst_day_prod = current_day + day_offset_prod
                    demand_val = sim.scenario.get_daily_drug_demand(fcst_day_prod, r_id, d_id) if fcst_day_prod < sim.duration_days else 0.0
                    if r_id == 0: 
                         obs["total_downstream_demand_forecast"][str(d_id)].append(demand_val)
                    else: 
                         if len(obs["total_downstream_demand_forecast"][str(d_id)]) > day_offset_prod -1:
                            obs["total_downstream_demand_forecast"][str(d_id)][day_offset_prod-1] += demand_val
                         else: 
                            obs["total_downstream_demand_forecast"][str(d_id)].append(demand_val)


                for day_offset_alloc in range(0, 3): # 3-day forecast starting from current day
                    fcst_day_alloc = current_day + day_offset_alloc
                    demand_val_alloc = sim.scenario.get_daily_drug_demand(fcst_day_alloc, r_id, d_id) if fcst_day_alloc < sim.duration_days else 0.0
                    obs["regional_hospital_demand_forecast"][str(d_id)][str(r_id)].append(demand_val_alloc)

        obs["downstream_inventory_summary"] = {k: dict(v) for k,v in obs["downstream_inventory_summary"].items()}
        obs["distributor_inventory_summary"] = {k: dict(v) for k,v in obs["distributor_inventory_summary"].items()}
        obs["total_downstream_demand_forecast"] = dict(obs["total_downstream_demand_forecast"])
        obs["regional_hospital_demand_forecast"] = {k: dict(v) for k,v in obs["regional_hospital_demand_forecast"].items()}

        obs["warehouse_release_delay"] = sim.warehouse_release_delay
        obs["pending_releases"] = [
            {"drug_id": str(entry["drug_id"]), "amount": entry["amount_produced"],
             "production_day": entry["day"], "days_in_warehouse": current_day - entry["day"],
             "expected_release_day": entry["day"] + sim.warehouse_release_delay} 
            for entry in sim.production_history
            if not entry.get("released", False)
        ]

        # --- Populate `recent_distributor_orders_by_drug` (for Manufacturer's allocation scratchpad) ---
        obs["recent_distributor_orders_by_drug"] = defaultdict(lambda: defaultdict(list))
        current_day_int_for_filter = int(current_day) 
        for order in sim.order_history:
            # Filter for orders TO manufacturer (to_id == 0) and within the last 3 days (for allocation needs context)
            if order.get("to_id") == 0 and order.get("day", -1) >= current_day_int_for_filter - 3:
                try:
                    dist_region_id = order["from_id"] - 1 # from_id for distributor is region_id + 1
                    drug_id_str = str(order["drug_id"])
                    amount = float(order["amount"])
                    # Ensure dist_region_id is valid before using as key
                    if 0 <= dist_region_id < sim.num_regions:
                         obs["recent_distributor_orders_by_drug"][drug_id_str][str(dist_region_id)].append(amount)
                    elif console_to_use:
                         console_to_use.print(f"[{Colors.WARNING}] Invalid dist_region_id {dist_region_id} derived from order in recent_distributor_orders_by_drug (Manu Obs): {order}")
                except (ValueError, TypeError, KeyError) as e:
                    if console_to_use:
                        console_to_use.print(f"[{Colors.WARNING}] Skipping invalid order in recent_distributor_orders_by_drug calculation (Manu Obs): {order} - Error: {e}[/{Colors.WARNING}]")
        obs["recent_distributor_orders_by_drug"] = {k: dict(v) for k,v in obs["recent_distributor_orders_by_drug"].items()}
        # Ensure all drugs/regions are present in recent_distributor_orders_by_drug with empty lists if no orders
        for d_id_str_key in obs.get("drug_info", {}).keys():
            if d_id_str_key not in obs["recent_distributor_orders_by_drug"]:
                obs["recent_distributor_orders_by_drug"][d_id_str_key] = {}
            for r_id_key in range(sim.num_regions):
                r_id_str_key = str(r_id_key)
                if r_id_str_key not in obs["recent_distributor_orders_by_drug"].get(d_id_str_key, {}):
                    # Ensure the drug key exists before trying to add a region key to its dictionary value
                    if d_id_str_key not in obs["recent_distributor_orders_by_drug"]:
                        obs["recent_distributor_orders_by_drug"][d_id_str_key] = {} # Initialize drug key if totally missing
                    obs["recent_distributor_orders_by_drug"][d_id_str_key][r_id_str_key] = []


        # --- Pre-calculate `sum_recent_distributor_orders` (for Manufacturer's production tool) ---
        obs["sum_recent_distributor_orders"] = defaultdict(float)
        # current_day_int_for_filter is already defined above
        # *** MODIFICATION HERE: Only include orders from *before* the current_day ***
        # For a 3-day lookback EXCLUDING the current day:
        # e.g., if current_day_int_for_filter is 1 (Day 2), we look at Day 0, -1, -2.
        # Max lookback_day is current_day_int_for_filter - 1
        # Min lookback_day is current_day_int_for_filter - 3
        max_lookback_day = current_day_int_for_filter - 1
        min_lookback_day = current_day_int_for_filter - 3

        for order in sim.order_history:
            order_day = order.get("day", -99) # Use a default far in the past if "day" is missing
            if order.get("to_id") == 0 and \
               min_lookback_day <= order_day <= max_lookback_day: # Check if order_day is in the lookback window
                try:
                    drug_id_str = str(order["drug_id"])
                    amount = float(order["amount"])
                    obs["sum_recent_distributor_orders"][drug_id_str] += amount
                except (ValueError, TypeError, KeyError) as e: 
                    if console_to_use: 
                        console_to_use.print(f"[{Colors.WARNING}] Skipping invalid order in sum_recent_distributor_orders calculation (Manu Obs): {order} - Error: {e}[/{Colors.WARNING}]")
        obs["sum_recent_distributor_orders"] = dict(obs["sum_recent_distributor_orders"]) 
        for d_id_str_key in obs.get("drug_info", {}).keys():
            if d_id_str_key not in obs["sum_recent_distributor_orders"]:
                obs["sum_recent_distributor_orders"][d_id_str_key] = 0.0
        


    elif agent_type == "distributor":
        region_id = agent_id_param
        obs["agent_id_for_logging"] = region_id + 1
        obs["region_id"] = region_id
        obs["my_region_patient_impact_score"] = sim.patient_impact_score.get(str(region_id), 0.0) # For patient impact feedback
        obs["inventories"] = {str(d): sim.dist_inv_stores[(region_id, d)].level for d in range(sim.num_drugs)}
        obs["inbound_pipeline"] = get_in_transit_to_destination("distributor", region_id)
        obs["recent_orders"] = [o for o in sim.order_history if o.get("to_id") == (region_id + 1) and o.get("day", -1) >= current_day - 3]
        obs["recent_allocations"] = [a for a in sim.allocation_history if a.get("to_id") == (region_id + 1) and a.get("day", -1) >= current_day - 7]
        
        obs["epidemiological_data"] = get_epi_data_for_region(region_id, current_day) # For its own region (hospital context)
        
        obs["hospital_stockout_summary"] = {str(d): sim.stockouts.get(str(d),{}).get(str(region_id),0) for d in range(sim.num_drugs)}
        obs["downstream_hospital_stockout_history"] = [ # For criticality assessment tool
            s_hist for s_hist in sim.stockout_history
            if s_hist.get("region_id") == region_id and s_hist.get("day",-1) >= current_day - 10 # e.g. last 10 days
        ]
        obs["downstream_hospital_demand_history"] = [ # For criticality assessment tool
            d_hist for d_hist in sim.demand_history
            if d_hist.get("region_id") == region_id and d_hist.get("day",-1) >= current_day - 10 # e.g. last 10 days
        ]
        obs["downstream_hospital_demand_forecast"] = {
            str(d): [sim.scenario.get_daily_drug_demand(current_day + day_offset, region_id, d) if (current_day + day_offset) < sim.duration_days else 0.0
                     for day_offset in range(0, 7)] # Forecast for its hospital, 7 days starting today
            for d in range(sim.num_drugs)
        }
        obs["dist_obs_demand_over_planning_horizon"] = defaultdict(float)
        obs["dist_obs_avg_daily_demand_in_horizon"] = defaultdict(float)
        
        ORDER_LEAD_TIME_DAYS_DIST = 5 # Must match LLM prompt if hardcoded there
        ORDER_REVIEW_PERIOD_DAYS_DIST = 1
        PLANNING_HORIZON_ORDER_DAYS_DIST = ORDER_LEAD_TIME_DAYS_DIST + ORDER_REVIEW_PERIOD_DAYS_DIST

        current_day_int_for_filter = int(current_day) # Ensure current_day is int

        for d_id_str in obs.get("drug_info", {}).keys():
            drug_id = int(d_id_str)
            hosp_fcst_list_for_drug = obs.get("downstream_hospital_demand_forecast", {}).get(d_id_str, [])
            
            # Fallback if forecast is empty or too short
            if not hosp_fcst_list_for_drug or len(hosp_fcst_list_for_drug) < PLANNING_HORIZON_ORDER_DAYS_DIST:
                # Use projected demand today for this hospital as fallback
                proj_demand_today_val = obs.get("epidemiological_data", {}).get("projected_demand", {}).get(d_id_str, 0.0)
                hosp_fcst_relevant_period = [proj_demand_today_val] * PLANNING_HORIZON_ORDER_DAYS_DIST
            else:
                hosp_fcst_relevant_period = [float(val) for val in hosp_fcst_list_for_drug[:PLANNING_HORIZON_ORDER_DAYS_DIST]]

            obs["dist_obs_demand_over_planning_horizon"][d_id_str] = sum(hosp_fcst_relevant_period)
            obs["dist_obs_avg_daily_demand_in_horizon"][d_id_str] = \
                sum(hosp_fcst_relevant_period) / max(1, len(hosp_fcst_relevant_period))
        
        obs["dist_obs_demand_over_planning_horizon"] = dict(obs["dist_obs_demand_over_planning_horizon"])
        obs["dist_obs_avg_daily_demand_in_horizon"] = dict(obs["dist_obs_avg_daily_demand_in_horizon"])

    elif agent_type == "hospital":
        region_id = agent_id_param
        obs["agent_id_for_logging"] = sim.num_regions + 1 + region_id
        obs["region_id"] = region_id
        obs["my_region_patient_impact_score"] = sim.patient_impact_score.get(str(region_id), 0.0) # For patient impact feedback
        obs["inventories"] = {str(d): sim.hosp_inv_stores[(region_id, d)].level for d in range(sim.num_drugs)}
        obs["inbound_pipeline"] = get_in_transit_to_destination("hospital", region_id)
        obs["recent_allocations"] = [a for a in sim.allocation_history if a.get("to_id") == (sim.num_regions + 1 + region_id) and a.get("day",-1) >= current_day - 7]
        
        # Filtered history for THIS hospital and recency (e.g., last 10 days)
        obs["demand_history"] = [
            d_hist for d_hist in sim.demand_history 
            if d_hist.get("region_id") == region_id and d_hist.get("day",-1) >= current_day - 10
        ]
        obs["stockout_history"] = [
            s_hist for s_hist in sim.stockout_history 
            if s_hist.get("region_id") == region_id and s_hist.get("day",-1) >= current_day - 10
        ]
        
        obs["epidemiological_data"] = get_epi_data_for_region(region_id, current_day) # For its own region
        
        obs["daily_demand_forecast_list_for_my_needs"] = { # For calculate_optimal_order tool
             str(d): [sim.scenario.get_daily_drug_demand(current_day + day_offset, region_id, d) if (current_day + day_offset) < sim.duration_days else 0.0
                      for day_offset in range(0, 7)] # Next 7 days starting today
             for d in range(sim.num_drugs)
         }

        obs["recent_actual_demand"] = {}
        for d_id_str in obs["drug_info"].keys():
            d_id = int(d_id_str)
            # Use the already filtered obs["demand_history"] specific to this hospital and recent days
            demands_for_drug = [h["demand"] for h in obs["demand_history"] if h["drug_id"] == d_id]
            current_cases_val = obs["epidemiological_data"].get("current_active_cases", 0.0)
            base_demand_val = float(obs["drug_info"][d_id_str].get("base_demand",0.0))
            
            obs["recent_actual_demand"][d_id_str] = round(np.mean(demands_for_drug),1) if demands_for_drug else \
                                                   round(base_demand_val * (current_cases_val / 100.0), 1)

        obs["hosp_obs_demand_sum_planning_horizon"] = defaultdict(float)
        obs["hosp_obs_avg_daily_demand_in_horizon"] = defaultdict(float)
        obs["hosp_obs_smoothed_daily_demand_signal"] = defaultdict(float)

        ORDER_LEAD_TIME_DAYS_HOSP = 3 # Must match LLM prompt
        ORDER_REVIEW_PERIOD_DAYS_HOSP = 1
        PLANNING_HORIZON_DAYS_HOSP = ORDER_LEAD_TIME_DAYS_HOSP + ORDER_REVIEW_PERIOD_DAYS_HOSP
        SMOOTHING_FACTOR_HOSP = 0.4 # Must match LLM prompt

        current_day_int_for_filter = int(current_day) # Ensure current_day is int

        for d_id_str in obs.get("drug_info", {}).keys():
            drug_id = int(d_id_str)
            # Use the existing daily_demand_forecast_list_for_my_needs from obs
            my_demand_fcst_list = obs.get("daily_demand_forecast_list_for_my_needs", {}).get(d_id_str, [])
            
            if not my_demand_fcst_list or len(my_demand_fcst_list) < PLANNING_HORIZON_DAYS_HOSP:
                proj_demand_today_val = obs.get("epidemiological_data", {}).get("projected_demand", {}).get(d_id_str, 0.0)
                demand_forecast_planning_horizon_hosp = [proj_demand_today_val] * PLANNING_HORIZON_DAYS_HOSP
            else:
                demand_forecast_planning_horizon_hosp = [float(val) for val in my_demand_fcst_list[:PLANNING_HORIZON_DAYS_HOSP]]

            obs["hosp_obs_demand_sum_planning_horizon"][d_id_str] = sum(demand_forecast_planning_horizon_hosp)
            avg_daily_demand_horizon_hosp = sum(demand_forecast_planning_horizon_hosp) / max(1, len(demand_forecast_planning_horizon_hosp))
            obs["hosp_obs_avg_daily_demand_in_horizon"][d_id_str] = avg_daily_demand_horizon_hosp
            
            # Calculate smoothed demand signal
            recent_actual_demand_val_hosp = float(obs.get("recent_actual_demand", {}).get(d_id_str, 0.0))
            smoothed_signal = (recent_actual_demand_val_hosp * SMOOTHING_FACTOR_HOSP) + \
                              (avg_daily_demand_horizon_hosp * (1 - SMOOTHING_FACTOR_HOSP))
            smoothed_signal = max(smoothed_signal, avg_daily_demand_horizon_hosp * 0.5) # Floor
            obs["hosp_obs_smoothed_daily_demand_signal"][d_id_str] = smoothed_signal

        obs["hosp_obs_demand_sum_planning_horizon"] = dict(obs["hosp_obs_demand_sum_planning_horizon"])
        obs["hosp_obs_avg_daily_demand_in_horizon"] = dict(obs["hosp_obs_avg_daily_demand_in_horizon"])
        obs["hosp_obs_smoothed_daily_demand_signal"] = dict(obs["hosp_obs_smoothed_daily_demand_signal"])



    # Add active disruptions relevant to this agent/location
    # This is fine as is, 'disruptions' key contains currently active AND relevant ones.
    obs["disruptions"] = []
    for disruption_event in sim.scenario.disruptions: # Iterate all scenario disruptions
        is_active = disruption_event["start_day"] <= current_day <= disruption_event["end_day"]
        is_relevant = False
        if agent_type == "manufacturer" and disruption_event["type"] == "manufacturing":
            is_relevant = True
        elif agent_type == "distributor" and disruption_event["type"] == "transportation" and disruption_event["region_id"] == agent_id_param:
            is_relevant = True
        elif agent_type == "hospital" and disruption_event["type"] == "transportation" and disruption_event["region_id"] == agent_id_param:
            is_relevant = True
        
        if is_active and is_relevant:
            obs["disruptions"].append(disruption_event)

    return obs