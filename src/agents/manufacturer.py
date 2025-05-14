# src/agents/manufacturer.py
import re
from collections import deque
import json
from typing import Dict, List, Optional, Any
import simpy # For type hinting env
import numpy as np # For memory formatting

from langchain_core.messages import HumanMessage, AIMessage
from .base import AgentState, create_agent_graph # LangGraph core
from .utils import clean_observation_for_prompt
from src.tools import PandemicSupplyChainTools
from src.llm.openai_integration import OpenAILLMIntegration
from config import console as global_console_config, Colors
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError: # pragma: no cover
    BlockchainInterface = None

class ManufacturerAgentLG:
    def __init__(
        self,
        env: simpy.Environment,
        agent_id: int, # Typically 0 for manufacturer
        num_regions: int,
        tools_instance: PandemicSupplyChainTools,
        openai_integration: OpenAILLMIntegration,
        usable_inventory_stores: Dict[int, simpy.Container],
        warehouse_inventory_stores: Dict[int, simpy.Container],
        memory_length: int = 3,
        verbose: bool = True,
        console_obj = None,
        blockchain_interface: Optional[BlockchainInterface] = None
    ):
        self.env = env
        self.agent_type = "manufacturer"
        self.agent_id = agent_id
        self.num_regions = num_regions
        self.tools_instance = tools_instance
        self.openai_integration = openai_integration
        self.console = console_obj if console_obj else global_console_config
        self.verbose = verbose
        self.blockchain_interface = blockchain_interface

        self.graph_app = create_agent_graph(
            llm_integration=self.openai_integration,
            tools_instance=self.tools_instance,
            bc_interface=self.blockchain_interface
        )
        self.memory = deque(maxlen=memory_length)
        self.usable_inventory_stores = usable_inventory_stores
        self.warehouse_inventory_stores = warehouse_inventory_stores

    def _print(self, message: str): # pragma: no cover
        if self.verbose and self.console:
            self.console.print(message)

    def _create_decision_prompt(self, observation: Dict) -> str:
        """
        Manufacturer prompt for SimPy.
        Ensures allocation tool call includes drug_info from LLM.
        Includes patient impact consideration.
        """
        day = observation.get("day", "N/A")
        obs_summary_str = clean_observation_for_prompt(observation.copy(), max_len=6000)
        agent = "Manufacturer"

        mem_summary = "My Recent history (oldest first, max 3 shown):\n"
        if not self.memory: # pragma: no cover
            mem_summary += "  (No history yet)"
        else:
            entries_to_show = list(self.memory)
            if len(entries_to_show) > 3 : entries_to_show = entries_to_show[-3:] # pragma: no cover

            for e_idx, e_entry in enumerate(entries_to_show):
                try:
                    prod_decision = e_entry.get("decision", {}).get("manufacturer_production", {})
                    alloc_decision = e_entry.get("decision", {}).get("manufacturer_allocation", {})
                    prod_str = {k_p: f"{float(v_p):.0f}" for k_p, v_p in prod_decision.items() if isinstance(v_p, (int, float, np.number))} if prod_decision else "{}"
                    alloc_sum_str = {}
                    if alloc_decision:
                        for d_id_alloc, r_allocs in alloc_decision.items():
                            if isinstance(r_allocs, dict) and r_allocs:
                                alloc_sum_str[d_id_alloc] = f"{sum(float(q_alloc) for q_alloc in r_allocs.values() if isinstance(q_alloc, (int, float, np.number))):.0f}"
                            else:
                                alloc_sum_str[d_id_alloc] = "0" # pragma: no cover
                    else: # pragma: no cover
                        alloc_sum_str = "{}"
                    mem_lines_detail = f"  - Day {e_entry.get('day', '?')+1}: Prod={prod_str}, AllocSum={alloc_sum_str}"
                    mem_summary += mem_lines_detail + "\n"
                except Exception as mem_e: # pragma: no cover
                    mem_summary += f"  - Day {e_entry.get('day', '?')}: Error formatting memory ({mem_e})\n"

        contract = (
            "Return ONE JSON object:\n"
            "{\n"
            "\"manufacturer_production\": {\"<drug_id_str>\": <quantity_float_or_int>, ...},\n"
            "\"manufacturer_allocation\": {\"<drug_id_str>\": {\"<region_id_str>\": <quantity_float_or_int>, ...}, ...},\n"
            "\"_debug_info\": { \"production_tool_args_used\": <object_json_string_of_args_passed_to_prod_tool>, \"allocation_method_reasoning\": \"<string>\" }\n"
            "}\n\n"
            "• Keys must be strings. Values must be numbers. Omit if quantity is zero.\n"
            "• For allocation, if a drug has no regions to allocate to, its value MUST be an empty dict `{}`.\n"
            "• `_debug_info.production_tool_args_used` should be a JSON string dump of the *actual arguments you passed* to `calculate_target_production_quantities`.\n"
            "• Ensure your entire response is ONLY this valid JSON object."
        )

        tools_for_prompt = (
            "Available Tools:\n\n"
            "1.  **`calculate_target_production_quantities` (Primary for Production):** Call this ONCE. You MUST construct its arguments carefully as per instructions and schema.\n"
            "    Required args include `current_inventories`, `pending_releases_soon`, `warehouse_inventories`, `sum_recent_orders`, `total_downstream_demand_forecast_list`, `any_downstream_stockouts`, and `observation_subset` (which must contain `production_capacity` and full `drug_info`).\n\n"
            "2.  `get_blockchain_regional_cases` (Optional, for Allocation Context if scarce and high criticality regions exist): Fetches trusted regional case counts.\n\n"
            "3.  `calculate_allocation_priority` (Optional, for Complex Allocation under Scarcity): Helps prioritize if simple proportional allocation (Step 3.3 in scratchpad) seems insufficient. Requires `drug_id`, `region_requests`, `available_inventory`, AND THE `drug_info` OBJECT for the specific drug (extracted from `obs_clean.drug_info`).\n\n"
            "*Tool Call Order:* First, `calculate_target_production_quantities` for production. Then, proceed to allocation logic (Step 3 in scratchpad). Only use allocation tools if Step 3.3 logic is clearly inadequate for a highly contested, critical drug.\n"
            "**Max 1 production tool call. Max 1, or rarely 2, allocation-related tool calls if absolutely necessary. Then STOP and output JSON.**"
        )
        
        production_tool_arg_instructions = f"""
**CRITICAL INSTRUCTIONS FOR `calculate_target_production_quantities` TOOL ARGUMENTS:**
You must provide the following arguments to this tool, correctly formatted:
1.  **Core Data Dictionaries** (keys are STRING DRUG IDs, e.g., "0", "1"):
    - `current_inventories`: {{ "drug_id_str": quantity_float, ... }}
    - `pending_releases_soon`: {{ "drug_id_str": qty, ... }}
    - `warehouse_inventories`: {{ "drug_id_str": qty, ... }}
    - `sum_recent_orders`: {{ "drug_id_str": total_quantity_float, ... }}
    - `total_downstream_demand_forecast_list`: {{ "drug_id_str": [fcst_day1_float, ...], ... }}
    - `any_downstream_stockouts`: {{ "drug_id_str": boolean_value, ... }}
2.  **`observation_subset`** (a dictionary containing key data extracted from `obs_clean`):
    -   It MUST include:
        -   `"production_capacity"`: {{ "drug_id_str": capacity_float, ... }}
        -   `"drug_info"`: {{ "drug_id_str": {{ "name": ..., "demand_factor": ..., "criticality_value": ... }}, ... }}
3.  **Optional Tuning Parameters** (e.g., `target_usable_inv_cover_days` as an integer).

Follow the scratchpad carefully to construct these arguments in a variable like `tool_arguments_to_pass`. The tool will fail if required arguments are missing or incorrectly formatted. Log the `tool_arguments_to_pass` in `_debug_info.production_tool_args_used`.
"""

        prompt = f"""
SYSTEM
You are the **{agent}**. Your primary goals are to maintain supply chain resilience through proactive production and to perform fair, daily allocations. Prioritize regions with severe stockouts, high patient impact, or high criticality needs.
Your Decision Process:
1.  **Prepare Production Inputs (Scratchpad Part 1):** Analyze the `Current observation` (referred to as `obs_clean` in scratchpad).
    {production_tool_arg_instructions}
    Calculate dynamic target cover days for production, considering overall trends, downstream stockouts, and system-wide patient impact (`obs_clean.system_wide_patient_impact_summary`).
2.  **Determine Production (Scratchpad Part 2):** Call the `calculate_target_production_quantities` tool ONCE using the fully constructed arguments from Step 1. Use its output for your `manufacturer_production` decision unless the tool call fails (then use a simple fallback).
3.  **Perform DAILY Allocation (Scratchpad Part 3):** Based on your current usable inventory (from `obs_clean.inventories`) PLUS any inventory expected to be released from the warehouse TODAY (calculated from `obs_clean.pending_releases` and `obs_clean.warehouse_release_delay`), allocate to regions. Use the proportional allocation logic detailed in the scratchpad, factoring in regional stockouts and `obs_clean.system_wide_patient_impact_summary` to adjust urgency. Only consider `get_blockchain_regional_cases` or `calculate_allocation_priority` (passing the specific `drug_info` object for the drug, extracted from `obs_clean.drug_info[drug_id_str]`) if there is extreme scarcity for a high-criticality drug and multiple regions have high, competing needs.
4.  **STOP & OUTPUT (Scratchpad Part 4):** Construct the final JSON object including `manufacturer_production`, `manufacturer_allocation`, and `_debug_info`.
5.  **Final Response:** Respond ONLY with the single, valid JSON object.

USER (Day {day+1})
Current observation (JSON string, referred to as 'obs_clean' in scratchpad):
{obs_summary_str}

{mem_summary}

{tools_for_prompt}

────────────────── Scratch-pad (private) ────────────────────────────────
# === Part 1: Prepare Inputs for `calculate_target_production_quantities` tool ===
# This section MUST correctly populate all required arguments for the tool.
`tool_arg_current_inventories` = {{}}; `tool_arg_pending_releases_soon` = {{}};
`tool_arg_warehouse_inventories` = {{}}; `tool_arg_sum_recent_orders` = {{}};
`tool_arg_any_downstream_stockouts` = {{}};
`current_day_for_calc` = int(obs_clean.day); 

`tool_arg_current_inventories` = obs_clean.inventories
`tool_arg_warehouse_inventories` = obs_clean.warehouse_inventories

`release_horizon_for_prod_tool` = 2;
For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `sum_pending_val_prod` = 0.0;
    for r_entry in obs_clean.pending_releases:
        if str(r_entry.get("drug_id")) == d_id_str and int(r_entry.get("expected_release_day", 999)) <= `current_day_for_calc` + `release_horizon_for_prod_tool` -1 :
            `sum_pending_val_prod` += float(r_entry.get("amount", 0.0))
    `tool_arg_pending_releases_soon[d_id_str]` = `sum_pending_val_prod`

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `sum_for_drug_orders` = 0.0
    if d_id_str in obs_clean.recent_distributor_orders_by_drug:
        for r_id_str, order_list_val in obs_clean.recent_distributor_orders_by_drug[d_id_str].items():
            if isinstance(order_list_val, list): sum_for_drug_orders += sum(float(o) for o in order_list_val if isinstance(o, (int,float, np.number)))
    `tool_arg_sum_recent_orders[d_id_str]` = sum_for_drug_orders

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `stockouts_dict_for_drug` = obs_clean.downstream_stockout_summary.get(d_id_str, {{}})
    `tool_arg_any_downstream_stockouts[d_id_str]` = `any(int(v_so) > 0 for v_so in stockouts_dict_for_drug.values())`

`tool_arg_observation_subset` = {{
    "production_capacity": obs_clean.production_capacity,
    "drug_info": obs_clean.drug_info // Pass the full drug_info from observation
}}

`base_target_cover_days` = 7
`dynamic_target_usable_cover_days_val` = `base_target_cover_days`
`max_patient_impact_overall` = max(float(v) for v in obs_clean.system_wide_patient_impact_summary.values()) if obs_clean.system_wide_patient_impact_summary else 0.0
If `max_patient_impact_overall` > 1000: `dynamic_target_usable_cover_days_val` += 3
elif `max_patient_impact_overall` > 200: `dynamic_target_usable_cover_days_val` += 1

`has_any_severe_stockouts` = False
For `drug_stockouts` in `obs_clean.downstream_stockout_summary.values()`:
    if any(int(count) > 3 for count in drug_stockouts.values()): `has_any_severe_stockouts` = True; break
If `has_any_severe_stockouts`: `dynamic_target_usable_cover_days_val` = max(4, `dynamic_target_usable_cover_days_val` - 2)

`tool_stockout_boost_factor_to_pass_val` = 1.7 
If `max_patient_impact_overall` > 1500 or `has_any_severe_stockouts`: `tool_stockout_boost_factor_to_pass_val` = 2.2
elif `max_patient_impact_overall` > 500: `tool_stockout_boost_factor_to_pass_val` = 1.9

`tool_arguments_to_pass` = {{
    "current_inventories": `tool_arg_current_inventories`,
    "pending_releases_soon": `tool_arg_pending_releases_soon`,
    "warehouse_inventories": `tool_arg_warehouse_inventories`,
    "sum_recent_orders": `tool_arg_sum_recent_orders`,
    "total_downstream_demand_forecast_list": obs_clean.total_downstream_demand_forecast,
    "any_downstream_stockouts": `tool_arg_any_downstream_stockouts`,
    "observation_subset": `tool_arg_observation_subset`
}}
If `dynamic_target_usable_cover_days_val` != `base_target_cover_days`:
    `tool_arguments_to_pass["target_usable_inv_cover_days"]` = int(round(`dynamic_target_usable_cover_days_val`))
If `tool_stockout_boost_factor_to_pass_val` != 1.7:
    `tool_arguments_to_pass["stockout_boost_factor"]` = `tool_stockout_boost_factor_to_pass_val`

`debug_production_tool_args_json_string` = json.dumps(`tool_arguments_to_pass`, default=str)

# === Part 2: Call Production Tool & Set Production Decision ===
`production_decision_from_tool` = Call `calculate_target_production_quantities` tool with `tool_arguments_to_pass`.
If isinstance(`production_decision_from_tool`, dict) AND "error" not in `production_decision_from_tool`:
    `manufacturer_production` = `production_decision_from_tool`
Else:
    `manufacturer_production` = {{}} 
    For each `d_id_str_fb` in `obs_clean.production_capacity.keys()`:
         `cap_fb_val` = float(obs_clean.production_capacity.get(d_id_str_fb, 0.0))
         `df_fb_val` = float(obs_clean.drug_info.get(d_id_str_fb, {{}}).get("demand_factor", 1.0))
         `manufacturer_production[d_id_str_fb]` = `min(max(df_fb_val * 1.5, cap_fb_val * 0.01), cap_fb_val * 0.15)`

# === Part 3: Allocation Planning ===
`manufacturer_allocation` = {{}}; `alloc_method_reason` = "Proportional based on weighted need."
For each `d_id_str_alloc_init` in `obs_clean.drug_info.keys()`: `manufacturer_allocation[d_id_str_alloc_init]` = {{}}

`current_day_for_alloc_calc` = int(obs_clean.day)
`release_delay_for_alloc_calc` = int(obs_clean.warehouse_release_delay)

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `inv_at_start_of_day_for_alloc` = float(obs_clean.inventories.get(d_id_str, 0.0))
    `expected_release_today_for_alloc` = 0.0
    for r_entry_alloc in obs_clean.pending_releases:
        if str(r_entry_alloc.get("drug_id")) == d_id_str:
            if int(r_entry_alloc.get("production_day", -999)) + `release_delay_for_alloc_calc` <= `current_day_for_alloc_calc`:
                 `expected_release_today_for_alloc` += float(r_entry_alloc.get("amount", 0.0))
    `current_usable_inv_for_alloc_calc` = `inv_at_start_of_day_for_alloc + expected_release_today_for_alloc`

    if `current_usable_inv_for_alloc_calc` <= 0.01: continue

    `internal_weighted_needs_alloc` = {{}}
    `stockouts_for_this_drug_alloc` = obs_clean.downstream_stockout_summary.get(d_id_str, {{}})
    `all_region_ids_obs_alloc` = list(obs_clean.epidemiological_data.keys()) 

    For each `r_id_str_alloc_loop` in `all_region_ids_obs_alloc`:
        `region_epi_data_alloc` = obs_clean.epidemiological_data.get(r_id_str_alloc_loop, {{}})
        `hosp_proj_demand_today_alloc` = float(region_epi_data_alloc.get("projected_demand", {{}}).get(d_id_str, 0.0))
        `hosp_forecast_for_region_drug_alloc` = obs_clean.regional_hospital_demand_forecast.get(d_id_str, {{}}).get(r_id_str_alloc_loop, [])
        `sum_hosp_forecast_for_alloc_calc` = sum(float(f) for f in `hosp_forecast_for_region_drug_alloc`[:3] if isinstance(f, (int, float, np.number)))

        `recent_orders_from_this_dist_list_alloc` = obs_clean.recent_distributor_orders_by_drug.get(d_id_str, {{}}).get(r_id_str_alloc_loop, [])
        `sum_recent_dist_orders_for_region_alloc` = sum(float(o) for o in `recent_orders_from_this_dist_list_alloc` if isinstance(o, (int, float, np.number)))

        `region_base_need_alloc` = max(`sum_hosp_forecast_for_alloc_calc`, `sum_recent_dist_orders_for_region_alloc`, `hosp_proj_demand_today_alloc`)
        `region_base_need_alloc` = max(0.0, `region_base_need_alloc`)

        `urgency_multiplier_alloc` = 1.0
        if int(stockouts_for_this_drug_alloc.get(r_id_str_alloc_loop, 0)) > 2: `urgency_multiplier_alloc` = 4.0
        elif int(stockouts_for_this_drug_alloc.get(r_id_str_alloc_loop, 0)) > 0: `urgency_multiplier_alloc` = 2.5
        
        `region_trend_alloc` = region_epi_data_alloc.get("case_trend_category", "stable")
        if `region_trend_alloc` == "increasing_strongly": `urgency_multiplier_alloc` += 2.5
        elif `region_trend_alloc` == "increasing": `urgency_multiplier_alloc` += 1.5
        
        `patient_impact_for_region_alloc` = float(obs_clean.system_wide_patient_impact_summary.get(r_id_str_alloc_loop, 0.0))
        if `patient_impact_for_region_alloc` > 500 : `urgency_multiplier_alloc` += 2.0
        elif `patient_impact_for_region_alloc` > 100 : `urgency_multiplier_alloc` += 1.0
        
        `dist_inv_for_drug_region_alloc` = float(obs_clean.distributor_inventory_summary.get(d_id_str, {{}}).get(r_id_str_alloc_loop, 0.0))
        `avg_daily_need_for_dist_supply_calc` = `region_base_need_alloc / max(1, 3)`
        `dist_days_of_supply_alloc` = `dist_inv_for_drug_region_alloc / max(1.0, avg_daily_need_for_dist_supply_calc)` if `avg_daily_need_for_dist_supply_calc` > 0 else 999
    
        if `dist_days_of_supply_alloc` < 1.5: `urgency_multiplier_alloc` += 2.0 
        elif `dist_days_of_supply_alloc` < 3.0: `urgency_multiplier_alloc` += 1.0
        `urgency_multiplier_alloc` = min(max(1.0, `urgency_multiplier_alloc`), 10.0)
        
        `internal_weighted_needs_alloc[r_id_str_alloc_loop]` = `region_base_need_alloc * urgency_multiplier_alloc`

    `total_weighted_need_for_drug_alloc` = sum(internal_weighted_needs_alloc.values())
    
    `drug_crit_val_alloc` = int(obs_clean.drug_info.get(d_id_str, {{}}).get("criticality_value", 1)) # Ensure correct key for criticality value
    `use_advanced_alloc_tool` = False
    if `drug_crit_val_alloc` >= 3 and `total_weighted_need_for_drug_alloc` > `current_usable_inv_for_alloc_calc` * 1.2:
        `use_advanced_alloc_tool` = True
        `alloc_method_reason` = "Advanced allocation tool due to scarcity of critical drug."

    if `use_advanced_alloc_tool`:
        `alloc_tool_drug_info_arg` = obs_clean.drug_info.get(d_id_str) # Get the specific drug_info object
        `alloc_tool_region_requests_arg` = {{r_id_str: float(internal_weighted_needs_alloc.get(r_id_str,0.0)) for r_id_str in all_region_ids_obs_alloc}}
        `alloc_tool_args` = {{
            "drug_id": int(d_id_str),
            "region_requests": `alloc_tool_region_requests_arg`,
            "available_inventory": `current_usable_inv_for_alloc_calc`,
            "drug_info": `alloc_tool_drug_info_arg` // THIS IS THE CRITICAL PART
        }}
        // Optional: if get_blockchain_regional_cases was called, add `region_cases` to `alloc_tool_args`
        // if `blockchain_cases_data` exists: `alloc_tool_args["region_cases"]` = `blockchain_cases_data`
        
        `advanced_alloc_result` = Call `calculate_allocation_priority` with `alloc_tool_args`.
        if isinstance(`advanced_alloc_result`, dict) and "error" not in `advanced_alloc_result`:
            for r_id_adv, alloc_val_adv in `advanced_alloc_result`.items():
                 if float(alloc_val_adv) > 0: `manufacturer_allocation[d_id_str][str(r_id_adv)]` = round(max(0.0, float(alloc_val_adv)))
        else: 
            `alloc_method_reason` += " Tool call failed, using proportional fallback."
            `use_advanced_alloc_tool` = False

    if not `use_advanced_alloc_tool`: 
        if `total_weighted_need_for_drug_alloc` > 0.01:
            `scale_factor_alloc` = min(1.0, `current_usable_inv_for_alloc_calc / total_weighted_need_for_drug_alloc`)
            for r_id_str_final_alloc, weighted_need_alloc in internal_weighted_needs_alloc.items():
                `allocated_amount_final` = round(max(0.0, float(weighted_need_alloc) * `scale_factor_alloc`))
                if `allocated_amount_final > 0`: `manufacturer_allocation[d_id_str][r_id_str_final_alloc]` = `allocated_amount_final`
        elif `current_usable_inv_for_alloc_calc` > 0.01 and len(all_region_ids_obs_alloc) > 0: 
            `equal_share_alloc` = `current_usable_inv_for_alloc_calc / len(all_region_ids_obs_alloc)`
            for r_id_str_eq_alloc in all_region_ids_obs_alloc:
                `df_alloc_eq` = float(obs_clean.drug_info.get(d_id_str, {{}}).get("demand_factor", 1.0))
                `heartbeat_amount_alloc` = round(min(`equal_share_alloc`, df_alloc_eq * 0.5, 5.0)) 
                if `heartbeat_amount_alloc > 0`: `manufacturer_allocation[d_id_str][r_id_str_eq_alloc]` = `heartbeat_amount_alloc`
            `alloc_method_reason` = "Equal heartbeat due to no specific weighted regional need."

# === Part 4: Final Output ===
For each `d_id_str_final_check` in `obs_clean.drug_info.keys()`:
    if `d_id_str_final_check` not in `manufacturer_allocation`: `manufacturer_allocation[d_id_str_final_check]` = {{}}
`final_debug_info` = {{ "production_tool_args_used": `debug_production_tool_args_json_string`, "allocation_method_reasoning": `alloc_method_reason` }}
`final_json_output` = {{"manufacturer_production": `manufacturer_production`, "manufacturer_allocation": `manufacturer_allocation`, "_debug_info": `final_debug_info` }}
────────────────── End Scratch-pad ──────────────────────────────────────────

{contract}
"""
        return prompt.strip()

    def _apply_hard_constraints(self, decision_json: Dict, observation: Dict) -> Dict:
        validated_production = {}
        if isinstance(decision_json.get("manufacturer_production"), dict):
            raw_production = decision_json["manufacturer_production"]
            for drug_id_str, amount_val in raw_production.items():
                try:
                    if drug_id_str not in observation.get("production_capacity", {}): # pragma: no cover
                        self._print(f"[{Colors.WARNING}] Manufacturer: Production decision for unknown drug_id {drug_id_str} ignored.[/]")
                        continue
                    capacity = float(observation.get("production_capacity", {}).get(drug_id_str, 0.0))
                    validated_amount = min(max(0.0, float(amount_val)), capacity)
                    if validated_amount > 0.01: 
                        validated_production[str(drug_id_str)] = validated_amount
                except (ValueError, TypeError): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Manufacturer: Invalid production value for D{drug_id_str}. Value: {amount_val}[/]")
                    pass 

        validated_allocation = {}
        if isinstance(decision_json.get("manufacturer_allocation"), dict):
            raw_allocation = decision_json["manufacturer_allocation"]
            for drug_id_str, region_allocs_val in raw_allocation.items():
                if drug_id_str not in observation.get("drug_info", {}): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Manufacturer: Allocation decision for unknown drug_id {drug_id_str} ignored.[/]")
                    continue
                try:
                    drug_allocs_validated = {}
                    if isinstance(region_allocs_val, dict):
                        for region_id_str, amount_val_alloc in region_allocs_val.items():
                            try:
                                if 0 <= int(region_id_str) < self.num_regions:
                                    alloc_amount = max(0.0, float(amount_val_alloc))
                                    if alloc_amount > 0.01: 
                                        drug_allocs_validated[str(region_id_str)] = alloc_amount
                                else: # pragma: no cover
                                     self._print(f"[{Colors.WARNING}] Manufacturer: Allocation for invalid region_id {region_id_str} (Drug {drug_id_str}) ignored.[/]")
                            except (ValueError, TypeError): # pragma: no cover
                                self._print(f"[{Colors.WARNING}] Manufacturer: Invalid allocation amount for D{drug_id_str}-R{region_id_str}. Value: {amount_val_alloc}[/]")
                                pass 
                    validated_allocation[str(drug_id_str)] = drug_allocs_validated
                except (ValueError, TypeError): # pragma: no cover
                     self._print(f"[{Colors.WARNING}] Manufacturer: Invalid region_allocs structure for D{drug_id_str}. Value: {region_allocs_val}[/]")
                     pass
        
        for d_id_str_key in observation.get("drug_info", {}).keys():
            if str(d_id_str_key) not in validated_allocation: # pragma: no cover
                validated_allocation[str(d_id_str_key)] = {}

        return {
            "manufacturer_production": validated_production,
            "manufacturer_allocation": validated_allocation,
            "_debug_info": decision_json.get("_debug_info", {}) 
        }


    def decide(self, observation: Dict) -> Dict:
        agent_name = "Manufacturer"
        current_day_from_obs = observation.get('day', int(self.env.now))
        self._print(f"\n[{Colors.MANUFACTURER}] {agent_name} LLM deciding (SimPy Day {current_day_from_obs + 1})...[/]")

        initial_prompt = self._create_decision_prompt(observation)
        initial_state: AgentState = {
            "messages": [HumanMessage(content=initial_prompt)],
            "observation": observation, 
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "num_regions": self.num_regions,
            "verbose": self.verbose, 
            "final_decision": None
        }

        final_json_decision = {}
        try:
            config = {"recursion_limit": 15} 
            final_state_from_graph = self.graph_app.invoke(initial_state, config=config)

            content_to_parse = None
            if final_state_from_graph and final_state_from_graph.get('messages'):
                last_message = final_state_from_graph['messages'][-1]
                if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
                    content_to_parse = last_message.content
                elif isinstance(last_message, ToolMessage) and self.verbose: # pragma: no cover
                     self._print(f"[{Colors.WARNING}] {agent_name}: Graph ended with ToolMessage. LLM might need more steps or prompt needs adjustment. Content: {last_message.content[:200]}[/{Colors.WARNING}]")
                     content_to_parse = '{"manufacturer_production": {}, "manufacturer_allocation": {}, "_debug_info": {"error": "Graph ended with ToolMessage, using fallback."}}'

            if content_to_parse:
                if self.verbose: self._print(f"[{Colors.DIM}][LLM Raw Output] {agent_name}:\n{content_to_parse}\n[/]")
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_to_parse, re.DOTALL | re.IGNORECASE)
                json_str = match.group(1) if match else None
                
                if not json_str: 
                    json_start = content_to_parse.find('{')
                    json_end = content_to_parse.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = content_to_parse[json_start : json_end + 1]
                
                if json_str:
                    try:
                        final_json_decision = json.loads(json_str)
                        if self.verbose: self._print(f"[{Colors.LLM}][LG Parsed JSON] {agent_name}: {str(final_json_decision)[:300]}...[/]")
                    except json.JSONDecodeError as e: # pragma: no cover
                        self._print(f"[{Colors.ERROR}] {agent_name}: Failed to parse LLM JSON: {e}. Extracted: '{json_str[:200]}...'[/]")
                elif "error" not in content_to_parse.lower(): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] {agent_name}: No valid JSON found in LLM output and not an error message. Output: {content_to_parse[:200]}[/]")
            elif not final_state_from_graph or not final_state_from_graph.get('messages'): # pragma: no cover
                 self._print(f"[{Colors.ERROR}] {agent_name}: LangGraph did not return any messages in final state.[/]")

        except Exception as e: # pragma: no cover
            self._print(f"[{Colors.ERROR}] {agent_name}: LangGraph invocation error: {type(e).__name__} - {e}[/]")
            if self.console and hasattr(self.console, 'print_exception'): 
                self.console.print_exception(show_locals=False, max_frames=2)

        if not final_json_decision or not isinstance(final_json_decision, dict) or \
           not ("manufacturer_production" in final_json_decision and "manufacturer_allocation" in final_json_decision):
            self._print(f"[{Colors.FALLBACK}] {agent_name}: Using empty/minimal fallback decision due to parsing/invocation issues or incomplete JSON structure.[/]")
            final_json_decision = { 
                "manufacturer_production": final_json_decision.get("manufacturer_production", {}),
                "manufacturer_allocation": final_json_decision.get("manufacturer_allocation", {}),
                "_debug_info": final_json_decision.get("_debug_info", {"error": "Fallback due to parsing, invocation, or incomplete JSON."})
            }

        validated_decision = self._apply_hard_constraints(final_json_decision, observation)
        if self.verbose: self._print(f"[{Colors.MANUFACTURER}][Final Validated Decision]: {str(validated_decision)[:300]}...[/]")

        obs_summary_for_memory = clean_observation_for_prompt(observation.copy(), max_len=500)
        self.memory.append({
            "day": current_day_from_obs,
            "obs_summary": obs_summary_for_memory,
            "decision": validated_decision 
        })
        return validated_decision


# Factory function
def create_openai_manufacturer_agent(
    env: simpy.Environment,
    tools: PandemicSupplyChainTools,
    openai_integration: OpenAILLMIntegration,
    num_regions: int,
    usable_inventory_stores: Dict[int, Any],
    warehouse_inventory_stores: Dict[int, Any],
    memory_length: int = 3,
    verbose: bool = True,
    console_obj = None,
    blockchain_interface: Optional[BlockchainInterface] = None
) -> ManufacturerAgentLG: # pragma: no cover
    return ManufacturerAgentLG(
        env=env,
        agent_id=0, 
        num_regions=num_regions,
        tools_instance=tools,
        openai_integration=openai_integration,
        usable_inventory_stores=usable_inventory_stores,
        warehouse_inventory_stores=warehouse_inventory_stores,
        memory_length=memory_length,
        verbose=verbose,
        console_obj=console_obj,
        blockchain_interface=blockchain_interface
    )