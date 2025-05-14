# src/simulation/processes.py

"""
SimPy process definitions for the pandemic supply chain simulation.
All processes here should be standard SimPy generator functions.
"""

import simpy
import json
from typing import Dict, List, Optional, Any, Callable # Ensure Callable is imported for type hint

from config import console, Colors 
from .helpers import format_observation_for_agent 

from langchain_core.messages import HumanMessage

# Note: allocation_priority_tool is now passed as an argument to manufacturer_process,
# so it does not need to be (and generally shouldn't be) imported directly here
# unless used by other processes in this file (which is not currently the case).

# --- Manufacturer Process ---
def manufacturer_process(
    env: simpy.Environment, 
    sim: 'PandemicSupplyChainSimulation', 
    agent_instance, 
    allocation_priority_tool_func: Optional[Callable] = None # Tool function passed as argument
):
    """SimPy process for the manufacturer agent's daily decision cycle."""
    while True:
        current_sim_time = env.now # Capture current time at the start of the loop
        day_str = f"Day {int(current_sim_time) + 1}"
        if sim.verbose: sim.console.print(f"[{Colors.MANUFACTURER}] Manufacturer deciding... ({day_str})[/]")

        try:
            # 1. Get Observation
            observation = format_observation_for_agent(env, sim, "manufacturer", 0) 

            # 2. LLM Decision
            raw_decision_json = agent_instance.decide(observation)

            # 3. Process Production
            production_orders = raw_decision_json.get("manufacturer_production", {})
            if production_orders:
                for drug_id_str, quantity in production_orders.items():
                    try:
                        drug_id = int(drug_id_str)
                        amount_to_produce = float(quantity)
                        if amount_to_produce <= 0.01: continue # Use epsilon

                        capacity = sim.scenario.get_manufacturing_capacity(int(current_sim_time), drug_id)
                        actual_production = min(amount_to_produce, capacity)

                        if actual_production > 0.01: # Use epsilon
                            yield sim.warehouse_inv_stores[drug_id].put(actual_production)
                            sim.production_history.append({
                                "day": int(current_sim_time), "drug_id": drug_id, 
                                "amount_ordered": amount_to_produce,
                                "amount_produced": actual_production, 
                                "capacity_at_time": capacity,
                                "released": False, # Initialize as not released
                                "release_day": None # Initialize release day
                            })
                            if sim.verbose:
                                sim.console.print(f"  [{Colors.MANUFACTURER}] Produced {actual_production:.1f} of Drug {drug_id} (Capacity: {capacity:.1f}). ({day_str})[/]")
                    except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                        sim.console.print(f"[{Colors.ERROR}] Manufacturer production error for D{drug_id_str}: {e} ({day_str})[/]")

            # 4. Handle Allocations (Batching & Blockchain Trigger)
            allocation_requests_from_llm = raw_decision_json.get("manufacturer_allocation", {})
            if allocation_requests_from_llm:
                for drug_id_str, regional_allocs in allocation_requests_from_llm.items():
                    for region_id_str, quantity in regional_allocs.items():
                        try:
                            sim.manufacturer_pending_batch_allocations[int(drug_id_str)][int(region_id_str)] += float(quantity)
                        except (ValueError, TypeError): # pragma: no cover
                            sim.console.print(f"[{Colors.ERROR}] Invalid allocation format from LLM: D{drug_id_str}, R{region_id_str} ({day_str})[/]")

            if (int(current_sim_time) + 1) % sim.allocation_batch_frequency == 0 and sim.manufacturer_pending_batch_allocations:
                if sim.verbose: sim.console.print(f"  [{Colors.MANUFACTURER}] Processing batched allocations... ({day_str})[/]")
                allocations_to_process_this_batch = sim.manufacturer_pending_batch_allocations.copy()
                sim.manufacturer_pending_batch_allocations.clear() 

                for drug_id, regional_requests_for_drug in allocations_to_process_this_batch.items():
                    # Ensure keys are integers if they might be strings from LLM/JSON
                    current_regional_requests = {int(k): float(v) for k,v in regional_requests_for_drug.items()}

                    # available_for_alloc = sim.manu_usable_inv_stores[drug_id].level
                    available_for_alloc = sim.manu_usable_inv_stores[drug_id].level if hasattr(sim.manu_usable_inv_stores[drug_id], 'level') else len(sim.manu_usable_inv_stores[drug_id].items)
                    
                    if available_for_alloc <= 0.01: # Use epsilon
                        if sim.verbose: sim.console.print(f"  [{Colors.MANUFACTURER}] No inventory of Drug {drug_id} to allocate. ({day_str})[/]")
                        continue

                    final_allocations_for_drug: Dict[int, float] = {} 
                    if sim.use_blockchain and sim.blockchain: # pragma: no cover
                        try:
                            region_ids_list = list(current_regional_requests.keys()) # Already int keys
                            requested_amounts_list = list(current_regional_requests.values())

                            bc_alloc_result_raw = sim.blockchain.execute_fair_allocation(
                                drug_id, region_ids_list, requested_amounts_list, available_for_alloc
                            )
                            # bc_alloc_result_raw is Dict[int, float] or None
                            sim.blockchain_tx_log.append({
                                "day": int(current_sim_time), "type": "execute_fair_allocation", "drug_id": drug_id,
                                "request": current_regional_requests, "available": available_for_alloc,
                                "result": bc_alloc_result_raw, "status": "success" if bc_alloc_result_raw is not None else "call_failed_or_returned_none"
                            })
                            if bc_alloc_result_raw is not None:
                                final_allocations_for_drug = bc_alloc_result_raw
                            elif allocation_priority_tool_func: 
                                if sim.verbose: sim.console.print(f"  [{Colors.WARNING}] BC allocation failed/None for D{drug_id}, using local fallback. ({day_str})[/]")
                                final_allocations_for_drug = allocation_priority_tool_func(
                                    sim.scenario.drugs[drug_id], current_regional_requests,
                                    {r_id: sim.current_regional_cases.get(r_id, 0) for r_id in current_regional_requests.keys()},
                                    available_for_alloc
                                )
                            # else: blockchain failed AND no local tool, final_allocations_for_drug remains empty
                        except Exception as e_bc:
                            sim.console.print(f"[{Colors.ERROR}] Error during BC allocation for D{drug_id}: {e_bc} ({day_str})[/]")
                            sim.blockchain_tx_log.append({
                                "day": int(current_sim_time), "type": "execute_fair_allocation", "drug_id": drug_id,
                                "request": current_regional_requests, "status": "exception", "error": str(e_bc)
                            })
                            if allocation_priority_tool_func:
                                final_allocations_for_drug = allocation_priority_tool_func(
                                    sim.scenario.drugs[drug_id], current_regional_requests,
                                    {r_id: sim.current_regional_cases.get(r_id, 0) for r_id in current_regional_requests.keys()},
                                    available_for_alloc
                                )
                    elif allocation_priority_tool_func: 
                        final_allocations_for_drug = allocation_priority_tool_func(
                            sim.scenario.drugs[drug_id], 
                            current_regional_requests,
                            {r_id: sim.current_regional_cases.get(r_id, 0) for r_id in current_regional_requests.keys()},
                            available_for_alloc
                        )
                    else: # Fallback if no blockchain and no local tool function passed
                        if sim.verbose: sim.console.print(f"[{Colors.WARNING}] Manufacturer: allocation_priority_tool_func not available and blockchain disabled, using basic proportional fallback for D{drug_id}. ({day_str})[/]") # pragma: no cover
                        total_req = sum(current_regional_requests.values()) # pragma: no cover
                        if total_req > 0: # pragma: no cover
                            for r_id_alloc, req_alloc in current_regional_requests.items():
                                final_allocations_for_drug[r_id_alloc] = min(req_alloc, (req_alloc / total_req) * available_for_alloc)
                    
                    # Initiate transport for final allocations
                    for region_id, allocated_amount_float in final_allocations_for_drug.items():
                        # region_id should be int from tool/BC, allocated_amount_float should be float
                        if allocated_amount_float > 0.01: # Use epsilon
                            # Check current inventory before attempting to get
                            current_manu_usable_inv = sim.manu_usable_inv_stores[drug_id].level
                            actual_to_get = min(allocated_amount_float, current_manu_usable_inv)

                            if actual_to_get > 0.01: # Use epsilon
                                yield sim.manu_usable_inv_stores[drug_id].get(actual_to_get)
                                dest_dist_store_key = (region_id, drug_id)
                                env.process(transport_process(env, sim, "manufacturer_to_distributor",
                                                              sim.dist_inv_stores[dest_dist_store_key],
                                                              drug_id, actual_to_get,
                                                              delay_factor_type="manu_to_dist",
                                                              origin_id=0, destination_id=region_id+1, 
                                                              destination_region_id_for_disruption=region_id))
                                sim.allocation_history.append({
                                    "day": int(current_sim_time), "drug_id": drug_id, "from_id": 0, "to_id": region_id + 1,
                                    "amount_allocated_by_logic": allocated_amount_float, 
                                    "amount_shipped": actual_to_get,
                                    "source": "blockchain" if sim.use_blockchain and sim.blockchain else "local_tool_or_fallback"
                                })
                                if sim.verbose:
                                    sim.console.print(f"  [{Colors.MANUFACTURER}] Shipped {actual_to_get:.1f} of D{drug_id} to Dist R{region_id}. (Allocated: {allocated_amount_float:.1f}) ({day_str})[/]")
                            elif sim.verbose: # pragma: no cover (hard to hit this if available_for_alloc was accurate)
                                sim.console.print(f"  [{Colors.MANUFACTURER}] Insufficient stock for D{drug_id} to Dist R{region_id} after checks. Needed {allocated_amount_float:.1f}, have {current_manu_usable_inv:.1f}. ({day_str})[/]")
                        
            if sim.verbose: sim.console.print(f"[{Colors.MANUFACTURER}] Manufacturer finished. ({day_str})[/]")
        except simpy.Interrupt: # pragma: no cover
            sim.console.print(f"[{Colors.WARNING}] Manufacturer process interrupted. ({day_str})[/]")
            break 
        except Exception as e: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Unhandled error in Manufacturer process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)

        yield env.timeout(1) 

# --- Distributor Process ---
def distributor_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation', agent_instance, region_id: int):
    """SimPy process for a distributor agent's daily decision cycle."""
    dist_node_id_for_log = region_id + 1 

    while True:
        current_sim_time = env.now
        day_str = f"Day {int(current_sim_time) + 1}"
        if sim.verbose: sim.console.print(f"[{Colors.DISTRIBUTOR}] Distributor R{region_id} deciding... ({day_str})[/]")

        try:
            observation = format_observation_for_agent(env, sim, "distributor", region_id)
            raw_decision_json = agent_instance.decide(observation) 

            # Process Orders to Manufacturer
            dist_orders_to_manu = raw_decision_json.get("distributor_orders", {})
            if dist_orders_to_manu: 
                # LLM decision for distributor_orders is {region_id: {drug_id: quantity}}
                orders_for_my_region = dist_orders_to_manu.get(str(region_id), dist_orders_to_manu.get(region_id, {}))

                for drug_id_str, quantity in orders_for_my_region.items():
                    try:
                        drug_id = int(drug_id_str)
                        amount_ordered = float(quantity)
                        if amount_ordered > 0.01: # Use epsilon
                            sim.order_history.append({
                                "day": int(current_sim_time), "drug_id": drug_id, "from_id": dist_node_id_for_log,
                                "to_id": 0, "amount": amount_ordered, "type": "dist_to_manu"
                            })
                            if sim.verbose:
                                sim.console.print(f"  [{Colors.DISTRIBUTOR}] Dist R{region_id} ordered {amount_ordered:.1f} of D{drug_id} from Manu. ({day_str})[/]")
                    except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                        sim.console.print(f"[{Colors.ERROR}] Dist R{region_id} order error for D{drug_id_str}: {e} ({day_str})[/]")

            # Process Allocations to Hospital
            dist_allocs_to_hosp = raw_decision_json.get("distributor_allocation", {})
            if dist_allocs_to_hosp:
                # LLM decision for distributor_allocation is {region_id: {drug_id: quantity}}
                allocs_for_my_region_hosp = dist_allocs_to_hosp.get(str(region_id), dist_allocs_to_hosp.get(region_id, {}))

                for drug_id_str, quantity in allocs_for_my_region_hosp.items():
                    try:
                        drug_id = int(drug_id_str)
                        amount_to_allocate = float(quantity)
                        if amount_to_allocate <= 0.01: continue # Use epsilon

                        my_inv_store = sim.dist_inv_stores[(region_id, drug_id)]
                        available_to_ship = my_inv_store.level
                        actual_shipped = min(amount_to_allocate, available_to_ship)

                        if actual_shipped > 0.01: # Use epsilon
                            yield my_inv_store.get(actual_shipped)
                            dest_hosp_store_key = (region_id, drug_id) 
                            env.process(transport_process(env, sim, "distributor_to_hospital",
                                                          sim.hosp_inv_stores[dest_hosp_store_key],
                                                          drug_id, actual_shipped,
                                                          delay_factor_type="dist_to_hosp",
                                                          origin_id=dist_node_id_for_log, 
                                                          destination_id=sim.num_regions + 1 + region_id, # Hospital's SimPy node ID
                                                          destination_region_id_for_disruption=region_id))
                            sim.allocation_history.append({
                                "day": int(current_sim_time), "drug_id": drug_id, "from_id": dist_node_id_for_log,
                                "to_id": sim.num_regions + 1 + region_id, 
                                "amount_allocated_by_logic": amount_to_allocate, # LLM decision
                                "amount_shipped": actual_shipped, # What was actually shipped
                                "source": "distributor_llm"
                            })
                            if sim.verbose:
                                sim.console.print(f"  [{Colors.DISTRIBUTOR}] Dist R{region_id} shipped {actual_shipped:.1f} of D{drug_id} to its Hosp. ({day_str})[/]")
                    except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                        sim.console.print(f"[{Colors.ERROR}] Dist R{region_id} allocation error for D{drug_id_str}: {e} ({day_str})[/]")

            if sim.verbose: sim.console.print(f"[{Colors.DISTRIBUTOR}] Distributor R{region_id} finished. ({day_str})[/]")
        except simpy.Interrupt: # pragma: no cover
            sim.console.print(f"[{Colors.WARNING}] Distributor R{region_id} process interrupted. ({day_str})[/]")
            break
        except Exception as e: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Unhandled error in Distributor R{region_id} process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)

        yield env.timeout(1)

# --- Hospital Process ---
def hospital_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation', agent_instance, region_id: int):
    """SimPy process for a hospital agent's daily decision cycle."""
    hosp_node_id_for_log = sim.num_regions + 1 + region_id 
    dist_node_id_for_log = region_id + 1 # Its corresponding distributor

    while True:
        current_sim_time = env.now
        day_str = f"Day {int(current_sim_time) + 1}"
        if sim.verbose: sim.console.print(f"[{Colors.HOSPITAL}] Hospital R{region_id} deciding... ({day_str})[/]")

        try:
            observation = format_observation_for_agent(env, sim, "hospital", region_id)
            raw_decision_json = agent_instance.decide(observation) 

            # Process Orders to Distributor
            hosp_orders_to_dist = raw_decision_json.get("hospital_orders", {})
            if hosp_orders_to_dist:
                # LLM decision for hospital_orders is {region_id: {drug_id: quantity}}
                orders_for_my_hosp = hosp_orders_to_dist.get(str(region_id), hosp_orders_to_dist.get(region_id, {}))

                for drug_id_str, quantity in orders_for_my_hosp.items():
                    try:
                        drug_id = int(drug_id_str)
                        amount_ordered = float(quantity)
                        if amount_ordered > 0.01: # Use epsilon
                            sim.order_history.append({
                                "day": int(current_sim_time), "drug_id": drug_id, "from_id": hosp_node_id_for_log,
                                "to_id": dist_node_id_for_log, "amount": amount_ordered, "type": "hosp_to_dist"
                            })
                            if sim.verbose:
                                sim.console.print(f"  [{Colors.HOSPITAL}] Hosp R{region_id} ordered {amount_ordered:.1f} of D{drug_id} from its Dist. ({day_str})[/]")
                    except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                        sim.console.print(f"[{Colors.ERROR}] Hosp R{region_id} order error for D{drug_id_str}: {e} ({day_str})[/]")

            if sim.verbose: sim.console.print(f"[{Colors.HOSPITAL}] Hospital R{region_id} finished. ({day_str})[/]")
        except simpy.Interrupt: # pragma: no cover
            sim.console.print(f"[{Colors.WARNING}] Hospital R{region_id} process interrupted. ({day_str})[/]")
            break
        except Exception as e: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Unhandled error in Hospital R{region_id} process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)

        yield env.timeout(1)

# --- Warehouse Release Process ---
def warehouse_release_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation'):
    """SimPy process for releasing inventory from warehouse to manufacturer's usable stock."""
    while True:
        current_sim_time = env.now
        day_str = f"Day {int(current_sim_time) + 1}"
        # This process is internal, maybe less verbose unless issues.
        # if sim.verbose: sim.console.print(f"[{Colors.SIMPY}] Warehouse releasing... ({day_str})[/]")
        
        try:
            processed_today = False
            # Iterate through production_history to find items due for release
            # Make a copy if modifying (though here we only modify entry's 'released' flag)
            for entry in sim.production_history: 
                if not entry.get("released", False) and \
                   entry["day"] <= int(current_sim_time) - sim.warehouse_release_delay: # Delay has passed
                    
                    drug_id = entry["drug_id"]
                    amount_produced_in_entry = entry["amount_produced"]
                    wh_store = sim.warehouse_inv_stores[drug_id]
                    manu_usable_store = sim.manu_usable_inv_stores[drug_id]

                    # Check if the amount is still in the warehouse (it should be if logic is correct)
                    # This is a safety check; ideally, amount_produced_in_entry should be takeable.
                    available_in_wh = wh_store.level
                    amount_to_release = min(amount_produced_in_entry, available_in_wh)

                    if amount_to_release > 0.01: # Use epsilon
                        yield wh_store.get(amount_to_release)
                        yield manu_usable_store.put(amount_to_release)
                        entry["released"] = True
                        entry["release_day"] = int(current_sim_time)
                        
                        if sim.verbose:
                            sim.console.print(f"  [{Colors.SIMPY}] Released {amount_to_release:.1f} of D{drug_id} from WH to Manu usable. (Produced on Day {entry['day']+1}) ({day_str})[/]")
                        processed_today = True
            
            # if not processed_today and sim.verbose and int(current_sim_time) > 0 : # Don't log for day 0 if nothing to release
            #      sim.console.print(f"  [{Colors.DIM}] No items eligible for warehouse release today. ({day_str})[/]")

        except simpy.Interrupt: # pragma: no cover
            sim.console.print(f"[{Colors.WARNING}] Warehouse release process interrupted. ({day_str})[/]")
            break
        except Exception as e: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Error in warehouse_release_process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)

        yield env.timeout(1)

# --- Epidemic and Demand Process ---
def epidemic_and_demand_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation'):
    """
    Daily SimPy process to:
    1. Update shared regional case counts and projected demands based on the scenario.
    2. Process patient demand at hospitals, updating inventories and metrics.
    3. Calculate and record daily backlog costs.
    """
    while True:
        current_day_int = int(env.now)
        day_str = f"Day {current_day_int + 1}"
        if sim.verbose: sim.console.print(f"[{Colors.SIMPY}] Updating epidemic state & processing patient demand... ({day_str})[/]")
        try:
            # 1. Update shared epidemic data from scenario curves
            for r_id in range(sim.num_regions):
                # Ensure current_day_int is within bounds of the pre-generated curve
                if r_id in sim.scenario.epidemic_curves and \
                   sim.scenario.epidemic_curves[r_id] is not None and \
                   current_day_int < len(sim.scenario.epidemic_curves[r_id]):
                    sim.current_regional_cases[r_id] = int(round(max(0, sim.scenario.epidemic_curves[r_id][current_day_int])))
                else: # pragma: no cover (should not happen if scenario_length matches sim duration)
                    sim.current_regional_cases[r_id] = 0 
                
                for d_id in range(sim.num_drugs):
                    sim.current_regional_projected_demand[r_id][d_id] = \
                        sim.scenario.get_daily_drug_demand(current_day_int, r_id, d_id)

            # 2. Process patient demand & calculate daily backlog cost
            daily_backlog_cost_for_today = 0.0
            for region_id in range(sim.num_regions):
                for drug_id in range(sim.num_drugs):
                    demand = sim.current_regional_projected_demand[region_id][drug_id]
                    if demand <= 1e-6: continue # Use epsilon

                    hosp_store = sim.hosp_inv_stores[(region_id, drug_id)]
                    available = hosp_store.level
                    
                    sim.total_demand_units[str(drug_id)][str(region_id)] += demand
                    sim.demand_history.append({
                        "day": current_day_int, "drug_id": drug_id, "region_id": region_id,
                        "demand": demand, "available_at_hospital": available
                    })

                    fulfilled = 0.0
                    if available >= demand:
                        yield hosp_store.get(demand)
                        fulfilled = demand
                    elif available > 1e-6: # Use epsilon
                        yield hosp_store.get(available)
                        fulfilled = available
                    
                    unfulfilled = demand - fulfilled

                    if unfulfilled > 1e-6: # Use epsilon
                        sim.stockouts[str(drug_id)][str(region_id)] += 1
                        sim.unfulfilled_demand_units[str(drug_id)][str(region_id)] += unfulfilled
                        sim.stockout_history.append({
                            "day": current_day_int, "drug_id": drug_id, "region_id": region_id,
                            "demand": demand, "unfulfilled": unfulfilled
                        })
                        drug_crit_val = sim.scenario.drugs[drug_id].get("criticality_value", 1)
                        sim.patient_impact_score[str(region_id)] += unfulfilled * drug_crit_val
                        
                        crit_scale_factor = 1.0 + (drug_crit_val - 1) * (sim.backlog_crit_multiplier - 1.0) / 3.0 
                        daily_backlog_cost_for_today += unfulfilled * sim.backlog_cost_per_unit * crit_scale_factor

            sim.total_backlog_cost += daily_backlog_cost_for_today
            
            # Find or create cost_history entry for today
            cost_entry_for_today = next((item for item in sim.cost_history if item["day"] == current_day_int), None)
            if cost_entry_for_today:
                cost_entry_for_today["backlog_cost"] = daily_backlog_cost_for_today
            else:
                sim.cost_history.append({
                    "day": current_day_int,
                    "holding_cost": 0, # Will be filled by _calculate_and_record_daily_holding_cost
                    "backlog_cost": daily_backlog_cost_for_today
                })

            if sim.verbose: sim.console.print(f"  [{Colors.SIMPY}] Epidemic & demand processing done. Daily Backlog Cost: ${daily_backlog_cost_for_today:.2f} ({day_str})[/]")
        except simpy.Interrupt: # pragma: no cover
            sim.console.print(f"[{Colors.WARNING}] Epidemic/Demand process interrupted. ({day_str})[/]")
            break
        except Exception as e: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Error in epidemic_and_demand_process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)
        yield env.timeout(1)

# --- Blockchain Daily Update Process ---
def blockchain_daily_update_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation'): # pragma: no cover
    """Daily SimPy process to update regional case counts on the blockchain."""
    if not sim.use_blockchain or not sim.blockchain:
        if sim.verbose: sim.console.print(f"[{Colors.DIM}] Blockchain daily update process skipped (not enabled).[/]")
        return 

    while True:
        current_sim_time = env.now
        day_str = f"Day {int(current_sim_time) + 1}"
        if sim.verbose: sim.console.print(f"[{Colors.BLOCKCHAIN}] Updating regional cases on blockchain... ({day_str})[/]")
        try:
            for r_id in range(sim.num_regions):
                cases = sim.current_regional_cases.get(r_id, 0) 
                tx_receipt_info = sim.blockchain.update_regional_case_count(r_id, cases)
                sim.blockchain_tx_log.append({
                    "day": int(current_sim_time), "type": "update_regional_cases", "region_id": r_id, "cases": cases,
                    "status": tx_receipt_info.get('status') if tx_receipt_info else 'error_before_send',
                    "receipt": tx_receipt_info.get('receipt') if tx_receipt_info else None,
                    "error_message": tx_receipt_info.get('error') if tx_receipt_info and tx_receipt_info.get('status') != 'success' else None
                })
                if tx_receipt_info and tx_receipt_info.get('status') == 'success':
                    if sim.verbose: sim.console.print(f"  [{Colors.BLOCKCHAIN}] Cases for R{r_id} ({cases}) updated on BC. ({day_str})[/]")
                else:
                    sim.console.print(f"  [{Colors.ERROR}] Failed to update R{r_id} cases on BC. Status: {tx_receipt_info.get('status') if tx_receipt_info else 'Unknown/NoReceipt'} ({day_str})[/]")
                yield env.timeout(0.01) 

            if sim.verbose: sim.console.print(f"[{Colors.BLOCKCHAIN}] Blockchain daily case updates finished. ({day_str})[/]")
        except simpy.Interrupt:
            sim.console.print(f"[{Colors.WARNING}] Blockchain daily update process interrupted. ({day_str})[/]")
            break
        except Exception as e:
            sim.console.print(f"[{Colors.ERROR}] Error in blockchain_daily_update_process: {e} ({day_str})[/]")
            sim.console.print_exception(max_frames=5)
        yield env.timeout(1)

# --- Transport Process ---
def transport_process(env: simpy.Environment, sim: 'PandemicSupplyChainSimulation',
                      transport_type: str, destination_store: simpy.Container, # Changed to simpy.Container
                      drug_id: int, quantity: float, delay_factor_type: str,
                      origin_id: Any, destination_id: Any, destination_region_id_for_disruption: int):
    current_sim_time = env.now
    day_str = f"Day {int(current_sim_time) + 1}"
    transport_entry = None 

    try:
        base_delay = 0.0
        dest_type_str = "" # Initialize
        dest_entity_id_for_active_transports = -1 # Initialize

        if delay_factor_type == "manu_to_dist": 
            base_delay = sim.scenario.drugs[drug_id].get("transportation_difficulty", 0.1) * 5.0 + \
                         sim.scenario.regions[destination_region_id_for_disruption].get("transportation_reliability", 1.0) * 2.0
            dest_type_str = "distributor"
            dest_entity_id_for_active_transports = destination_region_id_for_disruption
        elif delay_factor_type == "dist_to_hosp": 
            base_delay = sim.scenario.drugs[drug_id].get("transportation_difficulty", 0.1) * 2.0 + \
                         sim.scenario.regions[destination_region_id_for_disruption].get("transportation_reliability", 1.0) * 1.0
            dest_type_str = "hospital"
            dest_entity_id_for_active_transports = destination_region_id_for_disruption
        else: # pragma: no cover
            sim.console.print(f"[{Colors.ERROR}] Unknown transport delay_factor_type: {delay_factor_type} in transport_process. ({day_str})[/]")
            return 

        base_delay = max(1.0, base_delay)
        transport_capacity_factor = sim.scenario.get_transportation_capacity(
            int(current_sim_time), destination_region_id_for_disruption
        )
        actual_delay = base_delay / max(0.01, transport_capacity_factor) 
        actual_delay = round(max(1.0, actual_delay)) 
        expected_arrival_sim_time = current_sim_time + actual_delay

        transport_id = sim.next_transport_id
        sim.next_transport_id += 1
        transport_entry = {
            "transport_id": transport_id, "drug_id": drug_id, "quantity": quantity,
            "departure_time": current_sim_time, "expected_arrival_time": expected_arrival_sim_time,
            "destination_type": dest_type_str,
            "destination_entity_id": dest_entity_id_for_active_transports,
            "origin_entity_id": origin_id, "status": "in_transit"
        }
        sim.active_transports.append(transport_entry)
        
        if sim.verbose:
            sim.console.print(f"  [{Colors.SIMPY}] Transport DEPARTED (ID:{transport_id}): {quantity:.1f} of D{drug_id} from Node {origin_id} to Node {destination_id} (for Region {destination_region_id_for_disruption}). ETA: Day {int(expected_arrival_sim_time)+1}. ({day_str})[/]")
        
        yield env.timeout(actual_delay)

        yield destination_store.put(quantity)
        arrival_day_str = f"Day {int(env.now) + 1}" # Use env.now for arrival day

        if transport_entry: transport_entry["status"] = "arrived"
        
        if sim.verbose:
            sim.console.print(f"  [{Colors.SUCCESS}] Transport ARRIVED (ID:{transport_id}): {quantity:.1f} of D{drug_id} at Node {destination_id}. ({arrival_day_str})[/]")

    except simpy.Interrupt: # pragma: no cover
        sim.console.print(f"[{Colors.WARNING}] Transport process (ID:{transport_entry['transport_id'] if transport_entry else 'N/A'}) for D{drug_id} to Node {destination_id} interrupted. ({day_str})[/]")
        if transport_entry: transport_entry["status"] = "interrupted"
    except Exception as e: # pragma: no cover
        sim.console.print(f"[{Colors.ERROR}] Error in transport_process (ID:{transport_entry['transport_id'] if transport_entry else 'N/A'}) for D{drug_id} to Node {destination_id}: {e} ({day_str})[/]")
        if transport_entry: transport_entry["status"] = "error"
        sim.console.print_exception(max_frames=5)