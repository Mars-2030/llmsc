# --- START OF src/tools/allocation.py ---
"""
Allocation and order quantity optimization tools for pandemic supply chain simulation.
"""

import numpy as np
import math
from typing import Dict, List

# --- allocation_priority_tool (v1.1 - Increased Case Weight) ---
def allocation_priority_tool(
    drug_info: Dict, # Information about the specific drug being allocated
    region_requests: Dict[int, float], # {region_id: requested_amount}
    region_cases: Dict[int, float], # {region_id: current_cases} - Used for prioritization
    available_inventory: float
) -> Dict[int, float]:
    """
    Calculates fair allocation of limited inventory across regions based on
    requests, drug criticality, and regional case loads. Gives higher priority
    to higher requests, higher raw case counts, and higher drug criticality.
    Version 1.1: Increased weight given to raw case counts in priority calculation.

    Args:
        drug_info: Dict containing info like 'criticality_value'.
        region_requests: Requested amounts by region ID.
        region_cases: Current case loads by region ID.
        available_inventory: Total inventory available for allocation.

    Returns:
        Dict[int, float]: Calculated allocation amounts by region ID. Keys match input region_requests.
                          Ensures all original requesting regions are in the output, even if allocation is 0.
    """
    # --- Input Validation ---
    try:
        available_inventory = max(0.0, float(available_inventory))
    except (ValueError, TypeError):
        available_inventory = 0.0

    valid_requests = {}
    clean_region_cases = {}
    # Keep track of all originally requesting regions to ensure they are in the output
    original_region_ids = set()
    if isinstance(region_requests, dict):
        original_region_ids = set(region_requests.keys())
        for r_id, req in region_requests.items():
            try:
                r_id_int = int(r_id)
                valid_requests[r_id_int] = max(0.0, float(req)) if req is not None else 0.0
            except (ValueError, TypeError):
                 # If key is already int, handle potential value error
                 if isinstance(r_id, int): valid_requests[r_id] = 0.0

    if isinstance(region_cases, dict):
        for r_id, cases in region_cases.items():
            try:
                r_id_int = int(r_id)
                clean_region_cases[r_id_int] = max(0.0, float(cases)) if cases is not None else 0.0
            except (ValueError, TypeError):
                 if isinstance(r_id, int): clean_region_cases[r_id] = 0.0

    total_requested = sum(valid_requests.values())

    # Initialize result dict with all original region IDs mapped to 0.0
    # Convert potential string keys from original_region_ids to int if necessary
    int_original_region_ids = set()
    for r_id in original_region_ids:
        try:
            int_original_region_ids.add(int(r_id))
        except (ValueError, TypeError):
            pass # Ignore keys that cannot be converted to int
    final_output = {r_id_int: 0.0 for r_id_int in int_original_region_ids}


    # If nothing valid requested or no inventory, return zero allocations
    if total_requested <= 1e-6 or available_inventory <= 1e-6:
        return final_output # Already initialized with zeros

    # If enough inventory, fulfill all valid (non-negative) requests
    if total_requested <= available_inventory:
        for r_id, req in valid_requests.items():
            if r_id in final_output: # Ensure the key exists in the initialized output
                final_output[r_id] = req
        return final_output

    # --- Not enough inventory - Calculate Priorities (REVISED WEIGHTING) ---
    allocations = {} # Temporary dict for calculated allocations
    priorities = {}
    total_priority = 0.0
    # Default criticality = 1 (Lowest) if not found or invalid
    drug_criticality = drug_info.get("criticality_value", 1)
    if not isinstance(drug_criticality, (int, float)) or drug_criticality <= 0:
        drug_criticality = 1

    for region_id, request in valid_requests.items():
        if request <= 1e-6: continue # Skip zero requests in priority calculation

        regional_cases = clean_region_cases.get(region_id, 0.0)

        # --- MODIFIED CASE WEIGHT (Using Raw Cases + 1) ---
        # Gives strong weight to regions with higher absolute case numbers.
        # Adding 1 avoids zero weight for zero cases and provides a base weight.
        case_weight = 1.0 + regional_cases
        # --- END MODIFICATION ---

        # Square criticality to give more weight to higher values
        criticality_weight = drug_criticality**2

        # Priority = Request size * Case Load Weight * Criticality Weight
        priority = request * case_weight * criticality_weight
        priorities[region_id] = max(0.0, priority) # Ensure non-negative
        total_priority += priorities[region_id]

    # --- Allocate proportionally based on priority ---
    if total_priority > 1e-6:
        for region_id, priority in priorities.items():
            proportion = priority / total_priority
            # Allocate proportionally, but cap at the original request
            allocated_amount = min(valid_requests[region_id], proportion * available_inventory)
            allocations[region_id] = allocated_amount
    else:
        # Fallback: If total priority is zero (e.g., zero cases, zero requests),
        # distribute available inventory equally among positive requesters, capped by their request.
        positive_requesters = {r_id: req for r_id, req in valid_requests.items() if req > 1e-6}
        num_positive_requesters = len(positive_requesters)
        if num_positive_requesters > 0:
            equal_share = available_inventory / num_positive_requesters
            for region_id, request_amount in positive_requesters.items():
                # Cap equal share by the original request amount
                allocations[region_id] = min(request_amount, equal_share)
        # If no positive requesters, allocations remains empty

    # --- Final check & update ---
    # Ensure total allocated doesn't exceed available due to potential float issues
    total_allocated = sum(allocations.values())
    if total_allocated > available_inventory * 1.0001: # Allow tiny tolerance
        if total_allocated > 1e-6: # Avoid division by zero
            scale_down = available_inventory / total_allocated
            allocations = {r: a * scale_down for r, a in allocations.items()}
        else:
            allocations = {r: 0.0 for r in allocations} # Should not happen

    # Update the final_output dict with calculated allocations, ensuring non-negative
    for r_id, alloc in allocations.items():
        if r_id in final_output: # Ensure the region was part of the original request
            final_output[r_id] = max(0.0, alloc)

    return final_output


# --- optimal_order_quantity_tool (v1.0 - Trend Aware) ---
def optimal_order_quantity_tool(
    inventory_level: float,
    pipeline_quantity: float,
    daily_demand_forecast: List[float], # Expects a list of forecasted demands
    lead_time: int = 3,
    safety_stock_factor: float = 1.5 # Base factor (can be influenced by criticality upstream)
) -> float:
    """
    Calculates optimal order quantity using an order-up-to inventory policy,
    incorporating safety stock adjusted for DEMAND FORECAST TREND.

    Args:
        inventory_level: Current on-hand inventory.
        pipeline_quantity: Inventory already ordered but not yet received.
        daily_demand_forecast: Forecasted daily demand for future periods (list of floats).
        lead_time: Expected lead time for orders (days).
        safety_stock_factor: Base factor to apply to demand variability for safety stock.

    Returns:
        float: Optimal order quantity.
    """
    # --- Input Validation and Cleaning ---
    try: inventory_level = max(0.0, float(inventory_level))
    except (ValueError, TypeError): inventory_level = 0.0
    try: pipeline_quantity = max(0.0, float(pipeline_quantity))
    except (ValueError, TypeError): pipeline_quantity = 0.0
    try: lead_time = max(1, int(lead_time)) # Lead time must be at least 1
    except (ValueError, TypeError): lead_time = 1
    try: safety_stock_factor = max(1.0, float(safety_stock_factor)) # Ensure factor is at least 1
    except (ValueError, TypeError): safety_stock_factor = 1.0

    valid_forecast = []
    if isinstance(daily_demand_forecast, list): # Check if it's a list
        for d in daily_demand_forecast:
             try: valid_forecast.append(max(0.0, float(d))) # Ensure non-negative floats
             except (ValueError, TypeError): continue # Skip non-numeric elements

    # Define planning horizon: Lead Time + Review Period (assuming daily review = 1 day)
    review_period = 1
    planning_horizon = lead_time + review_period

    # --- Calculate Trend Multiplier based on Forecast ---
    trend_multiplier = 1.0 # Default: No trend adjustment
    sensitivity_factor = 0.75 # TUNABLE: How strongly trend impacts safety stock
    max_trend_multiplier = 3.5 # TUNABLE: Maximum boost from trend

    if len(valid_forecast) >= planning_horizon and planning_horizon >= 2:
        # More robust trend calculation: slope of linear regression over the planning horizon
        try:
            X = np.arange(planning_horizon).reshape(-1, 1)
            y = np.array(valid_forecast[:planning_horizon])
            # Handle cases with zero variance (e.g., constant forecast)
            if np.std(y) > 1e-9:
                model = np.polyfit(X.flatten(), y, 1) # Linear fit (slope is model[0])
                slope = model[0]
                avg_demand_in_period = np.mean(y) if len(y)>0 else 0.0
                if avg_demand_in_period > 1e-6: # Avoid division by zero
                    relative_slope = slope / avg_demand_in_period
                    # Scale multiplier by magnitude of relative slope and period length
                    # Only apply positive trend boost for safety stock calculation
                    trend_multiplier = 1.0 + max(0, relative_slope) * sensitivity_factor * planning_horizon
                    trend_multiplier = min(trend_multiplier, max_trend_multiplier)
            # else: keep trend_multiplier = 1.0 if demand is constant/zero
        except Exception: # Fallback if regression fails
            trend_multiplier = 1.0 # Default to no trend adjustment on error
    elif len(valid_forecast) >= 2: # Simpler trend if forecast too short
         avg_change = (valid_forecast[-1] - valid_forecast[0]) / max(1, len(valid_forecast) - 1)
         avg_demand = np.mean(valid_forecast) if len(valid_forecast)>0 else 0.0
         if avg_demand > 1e-6:
              relative_change = avg_change / avg_demand
              trend_multiplier = 1.0 + max(0, relative_change) * sensitivity_factor * planning_horizon
              trend_multiplier = min(trend_multiplier, max_trend_multiplier)

    # --- Calculate Demand Metrics over Planning Horizon ---
    demand_during_horizon = 0.0
    std_dev_demand_horizon = 0.0

    if not valid_forecast:
        # Handle case with no valid forecast data
        demand_during_horizon = 0.0
        std_dev_demand_horizon = 0.0 # No variability known
    elif len(valid_forecast) < planning_horizon:
        # Extrapolate if forecast is too short using the average of available forecast
        avg_daily_demand = np.mean(valid_forecast) if valid_forecast else 0.0
        known_demand = sum(valid_forecast)
        extrapolated_demand = avg_daily_demand * max(0, planning_horizon - len(valid_forecast))
        demand_during_horizon = known_demand + extrapolated_demand

        # Estimate standard deviation based on available history
        std_dev_daily = np.std(valid_forecast) if len(valid_forecast) > 1 else (valid_forecast[0] * 0.3 if valid_forecast else 0.0)
        std_dev_demand_horizon = std_dev_daily * math.sqrt(planning_horizon)
    else:
        # Use the forecast directly for the planning horizon
        horizon_forecast = valid_forecast[:planning_horizon]
        demand_during_horizon = sum(horizon_forecast)
        std_dev_daily = np.std(horizon_forecast) if len(horizon_forecast) > 0 else 0.0
        std_dev_demand_horizon = std_dev_daily * math.sqrt(planning_horizon)

    # Ensure demand and std dev are non-negative
    demand_during_horizon = max(0.0, demand_during_horizon)
    std_dev_demand_horizon = max(0.0, std_dev_demand_horizon)

    # --- Calculate Safety Stock ---
    # Base safety stock accounts for variability (standard deviation)
    base_safety_stock = safety_stock_factor * std_dev_demand_horizon
    # Adjusted safety stock accounts for positive trend (increases buffer if demand is rising)
    adjusted_safety_stock = base_safety_stock * trend_multiplier

    # --- Calculate Order-Up-To Level ---
    # Target level covers demand during lead time + review period, plus safety stock
    target_level = demand_during_horizon + adjusted_safety_stock

    # --- Calculate Inventory Position ---
    # Position = On-hand inventory + On-order inventory
    inventory_position = inventory_level + pipeline_quantity

    # --- Calculate Order Quantity ---
    # Order = Target Level - Current Position
    order_quantity = target_level - inventory_position

    # Ensure non-negative order quantity
    return max(0.0, order_quantity)

# --- END OF src/tools/allocation.py ---