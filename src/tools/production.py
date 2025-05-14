# src/tools/production.py
"""
Production calculation tools for the Manufacturer agent.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import math

try:
    from config import console, Colors
except ImportError:
    import sys
    class MockConsole:
        def print(self, *args, **kwargs): print(*args, file=sys.stderr)
    console = MockConsole()
    class Colors: # Ensure all colors used in this file are defined in fallback
        YELLOW = "yellow"; DIM = "dim"; CYAN = "cyan"; BOLD = "bold"; GREEN = "green"; RED = "red"

def calculate_target_production_quantities_tool(
    # === Arguments LLM *must* construct as per schema's 'required' list ===
    current_inventories: Dict[str, float],
    pending_releases_soon: Dict[str, float],
    warehouse_inventories: Dict[str, float],
    sum_recent_orders: Dict[str, float],
    total_downstream_demand_forecast_list: Dict[str, List[float]],
    any_downstream_stockouts: Dict[str, bool],

    # === NEW: Data LLM extracts from its main observation ===
    observation_subset: Dict[str, Any],

    # === NEW: Contextual data that execute_tool will add ===
    simulation_context: Dict[str, Any],

    # === Optional Config Parameters (LLM can override these if schema allows) ===
    target_usable_inv_cover_days: int = 7,
    target_warehouse_inv_cover_days: int = 5,
    forecast_horizon_for_signal: int = 5, # How many days of forecast to use for demand signal
    signal_percentile: float = 90.0,     # Percentile of forecast (or >=100 for max)
    stockout_boost_factor: float = 1.7,  # Multiplier for targets during stockouts
    startup_factor_cap: float = 0.3,     # Max production as % of capacity on day 0
    startup_demand_cover_days: int = 3,  # Days of demand to cover with startup production
    min_heartbeat_production_factor_df: float = 10.0, # Min production as multiple of drug's base demand factor
    min_heartbeat_production_factor_cap: float = 0.05,# Min production as % of capacity
    warehouse_shortfall_fulfillment_factor: float = 0.5 # How much of warehouse target shortfall to produce
) -> Dict[str, float]:
    """
    Version 1.9 (Simpler Args for LLM): Calculates dynamic target production quantities.
    - LLM provides core calculated dicts and a subset of other observations.
    - Other context like num_regions, current_sim_day is passed via simulation_context.
    """
    production_quantities: Dict[str, float] = {}
    DEBUG_TOOL = True # Set to True for verbose logging from this tool
    console_to_use = console # Use the imported console (either from config or mock)

    # --- Extract necessary data from the new arguments ---
    try:
        production_capacities = observation_subset.get("production_capacity", {})
        drug_info_full = observation_subset.get("drug_info", {}) # Expects {drug_id_str: {details...}}
        
        if not isinstance(production_capacities, dict):
            raise ValueError(f"production_capacity in observation_subset must be a dict, got {type(production_capacities)}")
        if not isinstance(drug_info_full, dict):
             raise ValueError(f"drug_info in observation_subset must be a dict, got {type(drug_info_full)}")

        drug_demand_factors = {
            k: v.get("demand_factor", 1.0) # Default to 1.0 if 'demand_factor' is missing
            for k, v in drug_info_full.items() if isinstance(v, dict)
        }

        num_regions = simulation_context.get("num_regions", 1)
        current_sim_day = simulation_context.get("current_sim_day", 0)
    except Exception as e:
        if DEBUG_TOOL and console_to_use:
            console_to_use.print(f"[{Colors.RED}][TOOL_PROD_ERROR] Error extracting initial params: {e}. Args received: observation_subset={str(observation_subset)[:200]}, simulation_context={simulation_context}[/{Colors.RED}]")
        return {"error": f"Tool setup error: {e}"}


    if not production_capacities:
        if DEBUG_TOOL and console_to_use:
            console_to_use.print(f"[{Colors.RED}][TOOL_PROD_WARNING] production_capacities is missing or empty in observation_subset. Returning no production.[/]", style=Colors.RED)
        return {} # No capacities, no production
    if not drug_info_full: # Check if drug_info led to empty demand_factors
        if DEBUG_TOOL and console_to_use:
            console_to_use.print(f"[{Colors.RED}][TOOL_PROD_WARNING] drug_info in observation_subset is missing or empty. Cannot determine demand_factors. Returning no production.[/]", style=Colors.RED)
        return {}


    for drug_id_str in production_capacities.keys():
        try:
            # --- Get inputs for this drug (from direct arguments or extracted) ---
            inv_usable = float(current_inventories.get(drug_id_str, 0.0))
            pending_r = float(pending_releases_soon.get(drug_id_str, 0.0))
            inv_wh = float(warehouse_inventories.get(drug_id_str, 0.0))
            raw_forecast_list_for_drug = total_downstream_demand_forecast_list.get(drug_id_str, [])
            cap = float(production_capacities.get(drug_id_str, 0.0))
            df = float(drug_demand_factors.get(drug_id_str, 1.0)) # Default to 1.0 if missing for a specific drug
            has_stockouts_flag = any_downstream_stockouts.get(drug_id_str, False)

            # --- Calculate Demand Signal (Percentile/Max from Forecast List) ---
            demand_signal = 0.0
            if isinstance(raw_forecast_list_for_drug, list) and raw_forecast_list_for_drug:
                # Use forecast_horizon_for_signal to determine how many days of forecast to consider
                horizon = min(int(forecast_horizon_for_signal), len(raw_forecast_list_for_drug))
                if horizon > 0:
                    relevant_forecast = [max(0.0, float(f_val)) for f_val in raw_forecast_list_for_drug[:horizon] if isinstance(f_val, (int, float))]
                    if relevant_forecast:
                        if float(signal_percentile) >= 100.0: # Use max if percentile is 100 or more
                            demand_signal = np.max(relevant_forecast)
                        else:
                            demand_signal = np.percentile(relevant_forecast, float(signal_percentile))
            demand_signal = max(0.0, demand_signal) # Ensure non-negative

            # --- Determine Effective Stockout Boost Factor ---
            current_stockout_boost_factor = float(stockout_boost_factor)
            drug_info_this_drug = drug_info_full.get(drug_id_str, {})
            if isinstance(drug_info_this_drug, dict): # Ensure it's a dict before .get
                drug_criticality_value = int(drug_info_this_drug.get("criticality_value", 1))
                if drug_criticality_value >= 3 and current_stockout_boost_factor < 2.0: # If critical/high-crit and boost is not already high
                    # Increase boost for more critical drugs
                    current_stockout_boost_factor = max(current_stockout_boost_factor, 2.0 + (drug_criticality_value - 3) * 0.25) # e.g. 2.0 for crit 3, 2.25 for crit 4
                    if DEBUG_TOOL and console_to_use:
                        console_to_use.print(f"  [ToolCritBoost] D{current_sim_day} Drug {drug_id_str}: CritVal={drug_criticality_value}, Boost factor for stockouts increased to {current_stockout_boost_factor:.2f}", style=Colors.CYAN)

            # --- Adjust targets based on stockouts ---
            effective_target_usable_cover_days_val = float(target_usable_inv_cover_days)
            effective_wh_shortfall_factor_val = float(warehouse_shortfall_fulfillment_factor)

            if has_stockouts_flag:
                effective_target_usable_cover_days_val *= current_stockout_boost_factor
                effective_wh_shortfall_factor_val *= current_stockout_boost_factor

            # --- Calculations ---
            min_demand_signal_for_targets = max(df * num_regions * 0.05, cap * 0.01, 0.1) # Small positive floor
            demand_signal_for_target_calc = max(demand_signal, min_demand_signal_for_targets)

            target_usable_inv = demand_signal_for_target_calc * effective_target_usable_cover_days_val
            effective_inv_pos_usable = inv_usable + pending_r
            needed_for_usable = target_usable_inv - effective_inv_pos_usable

            target_wh_inv = demand_signal_for_target_calc * float(target_warehouse_inv_cover_days)
            wh_shortfall = max(0.0, target_wh_inv - inv_wh)
            needed_for_warehouse = wh_shortfall * effective_wh_shortfall_factor_val # Use adjusted factor

            needed_production_raw = needed_for_usable + needed_for_warehouse
            needed_production_floored = max(0.0, needed_production_raw)

            min_heartbeat = math.ceil(max(df * float(min_heartbeat_production_factor_df), cap * float(min_heartbeat_production_factor_cap)))
            if cap > 0: min_heartbeat = max(min_heartbeat, 1.0) # Ensure at least 1 if capacity exists

            prod_qty = 0.0
            logic_branch_msg = "Initial"

            if DEBUG_TOOL and console_to_use:
                console_to_use.print(f"--- Tool Debug D{current_sim_day} Drug {drug_id_str} (v1.9 Refined) ---", style=Colors.YELLOW)
                console_to_use.print(f"  [ArgsDirect] UsableInv: {inv_usable:.1f}, PendingRelease: {pending_r:.1f}, WH_Inv: {inv_wh:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [ArgsDirect] Forecasts(D{drug_id_str}): {str(raw_forecast_list_for_drug)[:60]}...", style=Colors.DIM)
                console_to_use.print(f"  [ArgsDirect] Stockouts(D{drug_id_str}): {has_stockouts_flag}", style=Colors.DIM)
                console_to_use.print(f"  [ObsSubset] Capacity(D{drug_id_str}): {cap:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [ObsSubset] DemandFactor(D{drug_id_str}): {df:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Context] NumRegions: {num_regions}, SimDay: {current_sim_day}", style=Colors.DIM)
                console_to_use.print(f"  [Params] UsableCoverDaysArg: {target_usable_inv_cover_days}, WHCoverDaysArg: {target_warehouse_inv_cover_days}", style=Colors.DIM)
                console_to_use.print(f"  [Params] FcstHorizonSignal: {forecast_horizon_for_signal}, SignalPercentile: {signal_percentile}", style=Colors.DIM)
                console_to_use.print(f"  [Params] StockoutBoostArg: {stockout_boost_factor}, ActualBoostUsed: {current_stockout_boost_factor:.2f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] DemandSignal (P{float(signal_percentile):.0f} of Fcst[:{forecast_horizon_for_signal}]): {demand_signal:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] EffectiveUsableCoverDays (after boost): {effective_target_usable_cover_days_val:.1f}d", style=Colors.DIM)
                console_to_use.print(f"  [Calc] MinDemandForTargets: {min_demand_signal_for_targets:.1f}, DemandForTargetCalc: {demand_signal_for_target_calc:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] TargetUsableInv: {target_usable_inv:.1f}, EffectiveUsablePos: {effective_inv_pos_usable:.1f}, NeededForUsable: {needed_for_usable:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] TargetWH_Inv: {target_wh_inv:.1f}, WH_Shortfall: {wh_shortfall:.1f}, NeededForWH: {needed_for_warehouse:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] NeededProdRaw: {needed_production_raw:.1f} -> Floored: {needed_production_floored:.1f}", style=Colors.DIM)
                console_to_use.print(f"  [Calc] MinHeartbeat: {min_heartbeat:.1f}", style=Colors.DIM)

            # --- Production Logic (same as v1.8) ---
            if cap <= 0:
                prod_qty = 0.0
                logic_branch_msg = "No Capacity"
            elif current_sim_day == 0: # Startup logic
                startup_demand_est = max(demand_signal, df * num_regions * 0.1) * float(startup_demand_cover_days)
                prod_qty = min(max(startup_demand_est, min_heartbeat * 1.5), cap * float(startup_factor_cap))
                logic_branch_msg = f"Startup (EstNeed={startup_demand_est:.1f}, MinHB*1.5={min_heartbeat * 1.5:.1f}, CapFactorProd={cap * float(startup_factor_cap):.1f})"
            elif has_stockouts_flag:
                prod_qty = max(needed_production_floored, min_heartbeat)
                logic_branch_msg = f"Downstream Stockout (BoostedNeeded: {needed_production_floored:.1f}, MinHB: {min_heartbeat:.1f})"
            else: # No downstream stockouts
                prod_qty = max(needed_production_floored, min_heartbeat if cap > 0 and needed_production_floored < min_heartbeat else needed_production_floored)
                logic_branch_msg = f"No Stockout (Needed: {needed_production_floored:.1f}, MinHB: {min_heartbeat:.1f})"

            final_prod_qty = round(min(max(0.0, prod_qty), cap)) # Final cap and non-negative
            production_quantities[drug_id_str] = final_prod_qty

            if DEBUG_TOOL and console_to_use:
                console_to_use.print(f"  [Logic] Branch: {logic_branch_msg}", style=Colors.CYAN)
                console_to_use.print(f"  [Logic] QtyPreCap: {prod_qty:.1f}", style=Colors.CYAN)
                console_to_use.print(f"  [Output] FinalProd(D{drug_id_str}): {final_prod_qty}", style=f"{Colors.BOLD} {Colors.GREEN if final_prod_qty > 0 else Colors.DIM}")
                console_to_use.print("-" * 30, style=Colors.YELLOW)

        except ValueError as ve:
            if DEBUG_TOOL and console_to_use:
                console_to_use.print(f"[{Colors.RED}][TOOL_PROD_ERROR] ValueError for Drug {drug_id_str}: {ve}. Setting production to 0.[/]", style=Colors.RED)
            production_quantities[drug_id_str] = 0.0
        except Exception as e_drug: # Catch any other unexpected error for this specific drug
            if DEBUG_TOOL and console_to_use:
                console_to_use.print(f"[{Colors.RED}][TOOL_PROD_ERROR] Unexpected error processing Drug {drug_id_str}: {e_drug}. Setting production to 0.[/]", style=Colors.RED)
                # console_to_use.print_exception(max_frames=3) # For more detail
            production_quantities[drug_id_str] = 0.0


    if DEBUG_TOOL and console_to_use:
        console_to_use.print(f"[{Colors.GREEN}]Production Tool Final Output: {production_quantities}[/{Colors.GREEN}]")
    return production_quantities