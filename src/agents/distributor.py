# src/agents/distributor.py
import re
from collections import deque
import json
from typing import Dict, List, Optional, Any, Tuple
import simpy # For type hinting env
import numpy as np # For memory formatting

from langchain_core.messages import HumanMessage, AIMessage
from .base import AgentState, create_agent_graph
from .utils import clean_observation_for_prompt
from src.tools import PandemicSupplyChainTools
from src.llm.openai_integration import OpenAILLMIntegration
from config import console as global_console_config, Colors # Use global console

try:
    from src.blockchain.interface import BlockchainInterface
except ImportError: # pragma: no cover
    BlockchainInterface = None

class DistributorAgentLG:
    def __init__(
        self,
        env: simpy.Environment,
        region_id: int,
        num_regions: int,
        tools_instance: PandemicSupplyChainTools,
        openai_integration: OpenAILLMIntegration,
        inventory_stores: Dict[Tuple[int, int], simpy.Container], # Its own stores
        manufacturer_usable_stores: Dict[int, simpy.Container], # For context (e.g. Manu capacity if visible)
        memory_length: int = 3,
        verbose: bool = True,
        console_obj = None,
        blockchain_interface: Optional[BlockchainInterface] = None
    ):
        self.env = env
        self.agent_type = "distributor"
        self.agent_id = region_id # This is the region_id
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

        self.inventory_stores = inventory_stores
        self.manufacturer_usable_stores = manufacturer_usable_stores # Though less directly used by dist prompt

    def _print(self, message: str): # pragma: no cover
        if self.verbose and self.console:
            self.console.print(message)

    def _create_decision_prompt(self, observation: Dict) -> str:
        """
        Distributor prompt (SimPy adapted, considering patient impact and updated observation structure).
        """
        current_day_val = observation.get("day", "N/A")
        agent_name = f"Distributor (Region {self.agent_id})"
        # Observation is the full SimPy-derived dict for this agent
        obs_clean_str = clean_observation_for_prompt(observation.copy(), max_len=4500) # Increased max_len

        memory_summary = "My Recent history (newest first, max 2 shown):\n"
        if not self.memory: # pragma: no cover
            memory_summary += "  (No history yet)"
        else:
            recent = list(self.memory)[-2:]
            lines  = []
            for entry in reversed(recent):
                try:
                    agent_decisions = entry["decision"]
                    orders_dict = agent_decisions.get("distributor_orders", {}).get(self.agent_id, {})
                    allocs_dict = agent_decisions.get("distributor_allocation", {}).get(self.agent_id, {})

                    orders_str = {k: f"{float(v):.0f}" for k, v in orders_dict.items() if isinstance(v, (int,float,np.number))} if isinstance(orders_dict, dict) else "{}"
                    allocs_str = {k: f"{float(v):.0f}" for k, v in allocs_dict.items() if isinstance(v, (int,float,np.number))} if isinstance(allocs_dict, dict) else "{}"
                    lines.append(f"- Day {entry.get('day', '?')+1}: MyOrdersToManu={orders_str}, MyAllocsToHosp={allocs_str}")
                except Exception as e: # pragma: no cover
                    lines.append(f"- Day {entry.get('day', '?')}: Error formatting memory ({e})")
            if lines:
                memory_summary += "\n".join(lines)
            else: # pragma: no cover
                memory_summary += "  (No valid history entries to display)"


        response_contract = """
    Return exactly ONE valid JSON object and nothing else:
    {
    "distributor_orders":     {"<drug_id_str>": <quantity_float_or_int>, ...},
    "distributor_allocation": {"<drug_id_str>": <quantity_float_or_int>, ...}
    }
    • Keys must be strings; values are numbers. Omit if 0.
    • Do NOT wrap in ```json or add commentary.
    """
        tool_policy = """
    Available tools: get_criticality_assessment, predict_disruptions, calculate_optimal_order.
    You can also call `epidemic_forecast` if needed for your hospital's case trend.
    ↳ Call **at most ONE or TWO** tools this turn if essential for decision making.
    ↳ `calculate_optimal_order` for YOUR order to manufacturer (Step 6) ONLY as a *secondary check*. Provide only `drug_id`.
    ↳ `get_criticality_assessment` for a drug supplied to YOUR hospital if its stock is low or patient impact is high. Provide only `drug_id`.
    """

        prompt = f"""
You are **{agent_name}** on **Simulation Day {current_day_val+1}**.
PRIMARY GOAL: Ensure your downstream hospital (Region {self.agent_id}) has drugs. Consider its stockouts and patient impact (`obs_clean.my_region_patient_impact_score`).
SECONDARY GOAL: Supply stability. Use forecasts. Be more aggressive if hospital demand surges, trends are strongly increasing, or patient impact is high.

{memory_summary}

Current observation (JSON string, refers to 'obs_clean' in scratchpad):
{obs_clean_str}

{tool_policy}

Your Process:
1. Review current inventory (`obs_clean.inventories`), inbound pipeline (`obs_clean.inbound_pipeline`), and recent orders from your hospital (`obs_clean.recent_orders`).
2. Analyze your hospital's situation: projected demand, case trends (`obs_clean.epidemiological_data`), stockout days (`obs_clean.hospital_stockout_summary`), and patient impact (`obs_clean.my_region_patient_impact_score`).
3. If a drug is critical for your hospital (low stock, high demand/impact), use `get_criticality_assessment` (pass `drug_id`).
4. Determine order quantities to send to the Manufacturer (`distributor_orders`) using the logic in Step 6 of the scratchpad.
5. Determine allocation quantities to send to your Hospital (`distributor_allocation`) using the logic in Step 7 of the scratchpad.
6. STOP & OUTPUT: Respond ONLY with the final JSON.

──────────────────────── Scratchpad (private) ───────────────────────────
# For each drug_id_str (e.g., "0", "1", "2"):
# --- Lookup helpers (parse from 'Current observation' JSON string above) ---
# `my_inv[drug_id_str]` = float(obs_clean.inventories.get(drug_id_str, 0.0))
# `pipeline_to_me[drug_id_str]` = float(obs_clean.inbound_pipeline.get(drug_id_str, 0.0)) # From Manu to Me
# `hosp_region_epi_data` = obs_clean.epidemiological_data # Specific to my hospital's region
# `hosp_proj_demand_today[drug_id_str]` = float(hosp_region_epi_data.get("projected_demand", {{}}).get(drug_id_str, 0.0))
# `hosp_case_trend` = hosp_region_epi_data.get("case_trend_category", "stable")
# `hosp_current_cases` = int(hosp_region_epi_data.get("current_active_cases",0))
# `hosp_demand_forecast_list[drug_id_str]` = obs_clean.downstream_hospital_demand_forecast.get(drug_id_str, []) # List for my hosp, 7 days starting today
# `hosp_stockout_days[drug_id_str]` = int(obs_clean.hospital_stockout_summary.get(drug_id_str, 0))
# `my_region_impact_score` = float(obs_clean.my_region_patient_impact_score)

# `recent_hosp_orders_for_drug[drug_id_str]` = [] // Extract from obs_clean.recent_orders for this drug_id
For order_event in obs_clean.recent_orders:
    if str(order_event.get("drug_id")) == drug_id_str:
        `recent_hosp_orders_for_drug[drug_id_str]`.append(float(order_event.get("amount", 0.0)))

# --- Step 6: Ordering (from Manufacturer) - FOCUSED ON STABLE SUPPLY & FORECASTED NEEDS ---
`distributor_orders` = {{}}
`ORDER_LEAD_TIME_DAYS` = 5 # Assumed average lead time from manufacturer
`ORDER_REVIEW_PERIOD_DAYS` = 1 # Daily review
`PLANNING_HORIZON_ORDER_DAYS` = `ORDER_LEAD_TIME_DAYS` + `ORDER_REVIEW_PERIOD_DAYS` # = 6
`SAFETY_STOCK_TARGET_DAYS_ORDER_CALC` = 7 # Base safety stock in days of demand
`MAX_ORDER_AS_MULTIPLE_OF_HORIZON_DEMAND` = 2.5
`MIN_ORDER_QTY_IF_NEEDED_BASE_FACTOR` = 0.2 # Min order as factor of avg daily demand in horizon

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `drug_criticality_val` = int(obs_clean.drug_info.get(d_id_str, {{}}).get("criticality_value", 1))
    `current_safety_stock_days` = `SAFETY_STOCK_TARGET_DAYS_ORDER_CALC`
    `urgency_boost_order_calc` = 1.0

    if `hosp_stockout_days[d_id_str]` > 2 : `current_safety_stock_days` += 4; `urgency_boost_order_calc` = 1.5
    elif `hosp_stockout_days[d_id_str]` > 0 : `current_safety_stock_days` += 2; `urgency_boost_order_calc` = 1.2
    if `hosp_case_trend` == "increasing_strongly": `current_safety_stock_days` += 3; `urgency_boost_order_calc` = max(`urgency_boost_order_calc`, 1.4)
    elif `hosp_case_trend` == "increasing": `current_safety_stock_days` += 1.5; `urgency_boost_order_calc` = max(`urgency_boost_order_calc`, 1.1)
    if `my_region_impact_score` > 500 and `drug_criticality_val` >=3 : `current_safety_stock_days` += 3; `urgency_boost_order_calc` = max(`urgency_boost_order_calc`, 1.6)
    elif `my_region_impact_score` > 100 : `current_safety_stock_days` += 1;

    `current_safety_stock_days` = min(max(3, `current_safety_stock_days`), 15) # Bound safety stock days

    `hosp_fcst_relevant_period` = `hosp_demand_forecast_list[d_id_str]`[:`PLANNING_HORIZON_ORDER_DAYS`]
    if not `hosp_fcst_relevant_period`: `hosp_fcst_relevant_period` = [`hosp_proj_demand_today[d_id_str]`] * `PLANNING_HORIZON_ORDER_DAYS` # Fallback
    `demand_over_planning_horizon` = sum(`hosp_fcst_relevant_period`)
    `avg_daily_demand_in_horizon` = `demand_over_planning_horizon` / max(1, len(`hosp_fcst_relevant_period`))
    
    `safety_stock_units` = `avg_daily_demand_in_horizon` * `current_safety_stock_days`
    `order_up_to_level` = `demand_over_planning_horizon` + `safety_stock_units`
    `inventory_position` = `my_inv[d_id_str]` + `pipeline_to_me[d_id_str]`
    `calculated_order_qty` = `order_up_to_level` - `inventory_position`
    `calculated_order_qty` *= `urgency_boost_order_calc` # Apply urgency boost

    `final_order_qty` = 0.0
    if `calculated_order_qty` > 0.01:
        `min_meaningful_order` = max(1.0, `avg_daily_demand_in_horizon` * `MIN_ORDER_QTY_IF_NEEDED_BASE_FACTOR`)
        `final_order_qty` = max(`calculated_order_qty`, `min_meaningful_order`)
        `final_order_qty` = min(`final_order_qty`, `avg_daily_demand_in_horizon` * `PLANNING_HORIZON_ORDER_DAYS` * `MAX_ORDER_AS_MULTIPLE_OF_HORIZON_DEMAND`) # Cap order
        `distributor_orders[d_id_str]` = round(max(0.0, `final_order_qty`))

# --- Step 7: Allocation (to My Hospital) ---
`distributor_allocation` = {{}}
`ALLOC_SAFETY_STOCK_DAYS_HOSP` = 2.0 # How many days of hospital's demand to keep as safety for them at MY level (influences my allocation)
`MAX_ALLOC_MULTIPLE_OF_DEMAND` = 2.5 # Max to allocate as multiple of today's projected demand

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `current_inv_for_alloc` = `my_inv[d_id_str]`
    if `current_inv_for_alloc` <= 0.01: continue

    `hosp_demand_today_alloc` = `hosp_proj_demand_today[d_id_str]`
    `sum_recent_hosp_orders` = sum(`recent_hosp_orders_for_drug[d_id_str]`) if `recent_hosp_orders_for_drug[d_id_str]` else 0.0
    
    // Base allocation on max of today's demand or recent orders
    `base_alloc_need` = max(`hosp_demand_today_alloc`, `sum_recent_hosp_orders` * 0.5) # Factor down sum of recent orders a bit

    `alloc_urgency_multiplier` = 1.0
    if `hosp_stockout_days[d_id_str]` > 1: `alloc_urgency_multiplier` = 2.0
    elif `hosp_stockout_days[d_id_str]` > 0: `alloc_urgency_multiplier` = 1.5
    if `hosp_case_trend` == "increasing_strongly": `alloc_urgency_multiplier` = max(`alloc_urgency_multiplier`, 1.8)
    if `my_region_impact_score` > 300 : `alloc_urgency_multiplier` = max(`alloc_urgency_multiplier`, 2.2)
    
    `target_allocation` = `base_alloc_need` * `alloc_urgency_multiplier`
    
    // Ensure I try to keep some safety stock for the hospital's *next few days* if possible
    `hosp_next_3day_demand_forecast` = sum(`hosp_demand_forecast_list[d_id_str]`[:3]) # Uses the 7-day forecast from obs
    `desired_buffer_for_hosp_at_my_level` = `hosp_next_3day_demand_forecast` * (`ALLOC_SAFETY_STOCK_DAYS_HOSP` / 3.0) # Pro-rata safety based on 3 days
    
    // Amount to allocate = target_allocation, but capped by what I have minus what I want to keep as buffer for hospital
    `max_can_send_while_keeping_buffer` = `current_inv_for_alloc` - `desired_buffer_for_hosp_at_my_level`
    
    `final_allocation_amount` = min(`target_allocation`, max(0.0, `max_can_send_while_keeping_buffer`))
    `final_allocation_amount` = min(`final_allocation_amount`, `hosp_demand_today_alloc` * `MAX_ALLOC_MULTIPLE_OF_DEMAND`) // Cap
    `final_allocation_amount` = min(`final_allocation_amount`, `current_inv_for_alloc`) // Absolute cap by my inventory

    if `final_allocation_amount` > 0.01:
        `distributor_allocation[d_id_str]` = round(max(0.0, `final_allocation_amount`))

# Final JSON construction
`final_json_output` = {{"distributor_orders": `distributor_orders`, "distributor_allocation": `distributor_allocation`}}
──────────────────────── End Scratchpad ────────────────────────────────

{response_contract}
"""
        return prompt.strip()

    def _apply_hard_constraints(self, decision_json: Dict, observation: Dict) -> Dict:
        validated_orders = {} 
        validated_allocation = {} 

        drug_info_obs = observation.get("drug_info", {})
        defined_drug_ids_str = list(drug_info_obs.keys())

        raw_orders = decision_json.get("distributor_orders", {})
        if isinstance(raw_orders, dict):
            for drug_id_str, amount_val in raw_orders.items():
                if drug_id_str not in defined_drug_ids_str: # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Distributor R{self.agent_id}: Order for unknown drug_id {drug_id_str} ignored.[/]")
                    continue
                try:
                    validated_amount = max(0.0, float(amount_val))
                    if validated_amount > 0.01:
                        validated_orders[drug_id_str] = validated_amount
                except (ValueError, TypeError): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Distributor R{self.agent_id}: Invalid order value for D{drug_id_str}. Value: {amount_val}[/]")
                    pass

        raw_allocation = decision_json.get("distributor_allocation", {})
        my_current_inventory_obs = observation.get("inventories", {})
        if isinstance(raw_allocation, dict):
            for drug_id_str, amount_val_alloc in raw_allocation.items():
                if drug_id_str not in defined_drug_ids_str: # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Distributor R{self.agent_id}: Allocation for unknown drug_id {drug_id_str} ignored.[/]")
                    continue
                try:
                    available_inv_for_drug = float(my_current_inventory_obs.get(drug_id_str, 0.0))
                    alloc_amount = min(max(0.0, float(amount_val_alloc)), available_inv_for_drug)
                    if alloc_amount > 0.01:
                        validated_allocation[drug_id_str] = alloc_amount
                except (ValueError, TypeError): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Distributor R{self.agent_id}: Invalid allocation value for D{drug_id_str}. Value: {amount_val_alloc}[/]")
                    pass 
        return {
            "distributor_orders": validated_orders,
            "distributor_allocation": validated_allocation
        }

    def decide(self, observation: Dict) -> Dict:
        agent_name = f"Distributor R{self.agent_id}"
        current_day_from_obs = observation.get('day', int(self.env.now))
        self._print(f"\n[{Colors.DISTRIBUTOR}] {agent_name} LLM deciding (SimPy Day {current_day_from_obs + 1})...[/]")

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
            config = {"recursion_limit": 10} # Distributors typically have simpler decision chains
            final_state_from_graph = self.graph_app.invoke(initial_state, config=config)
            
            content_to_parse = None
            if final_state_from_graph and final_state_from_graph.get('messages'):
                last_message = final_state_from_graph['messages'][-1]
                if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
                    content_to_parse = last_message.content
                elif isinstance(last_message, ToolMessage) and self.verbose: # pragma: no cover
                     self._print(f"[{Colors.WARNING}] {agent_name}: Graph ended with ToolMessage. LLM might need more steps. Content: {last_message.content[:200]}[/{Colors.WARNING}]")
                     content_to_parse = '{"distributor_orders": {}, "distributor_allocation": {}}' # Fallback

            if content_to_parse:
                if self.verbose: self._print(f"[{Colors.DIM}][LLM Raw Output] {agent_name}:\n{content_to_parse}\n[/]")
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_to_parse, re.DOTALL | re.IGNORECASE)
                json_str = match.group(1) if match else None
                if not json_str:
                    json_start = content_to_parse.find('{'); json_end = content_to_parse.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = content_to_parse[json_start : json_end + 1]
                
                if json_str:
                    try:
                        final_json_decision = json.loads(json_str)
                        if self.verbose: self._print(f"[{Colors.LLM}][LG Parsed JSON] {agent_name}: {str(final_json_decision)[:300]}...[/]")
                    except json.JSONDecodeError as e: # pragma: no cover
                        self._print(f"[{Colors.ERROR}] {agent_name} JSON parse error: {e}. Extracted: '{json_str[:200]}...'[/]")
                elif "error" not in content_to_parse.lower(): # pragma: no cover
                     self._print(f"[{Colors.WARNING}] {agent_name}: No valid JSON found in LLM output: {content_to_parse[:200]}[/]")
            elif not final_state_from_graph or not final_state_from_graph.get('messages'): # pragma: no cover
                 self._print(f"[{Colors.ERROR}] {agent_name}: LangGraph did not return any messages in final state.[/]")

        except Exception as e: # pragma: no cover
            self._print(f"[{Colors.ERROR}] {agent_name} LangGraph invocation error: {type(e).__name__} - {e}[/]")
            if self.console and hasattr(self.console, 'print_exception'):
                self.console.print_exception(show_locals=False, max_frames=2)

        if not final_json_decision or not isinstance(final_json_decision, dict) or \
           not ("distributor_orders" in final_json_decision and "distributor_allocation" in final_json_decision) :
            self._print(f"[{Colors.FALLBACK}] {agent_name} using empty/minimal fallback decision.[/]")
            final_json_decision = {
                "distributor_orders": final_json_decision.get("distributor_orders", {}),
                "distributor_allocation": final_json_decision.get("distributor_allocation", {}),
            }

        validated_decision_parts = self._apply_hard_constraints(final_json_decision, observation)

        return_for_process = {
            "distributor_orders": {self.agent_id: validated_decision_parts.get("distributor_orders", {})},
            "distributor_allocation": {self.agent_id: validated_decision_parts.get("distributor_allocation", {})}
        }
        if self.verbose: self._print(f"[{Colors.DISTRIBUTOR}][Final Validated Decision] {agent_name}: {return_for_process}[/]")

        obs_summary_for_memory = clean_observation_for_prompt(observation.copy(), max_len=500)
        self.memory.append({
            "day": current_day_from_obs,
            "obs_summary": obs_summary_for_memory,
            "decision": return_for_process 
        })
        return return_for_process


# Factory function
def create_openai_distributor_agent(
    env: simpy.Environment,
    region_id: int,
    num_regions: int,
    tools: PandemicSupplyChainTools,
    openai_integration: OpenAILLMIntegration,
    inventory_stores: Dict[Tuple[int, int], Any], # Its own stores
    manufacturer_usable_stores: Dict[int, Any], # Context for Manu capacity etc.
    memory_length: int = 3,
    verbose: bool = True,
    console_obj = None,
    blockchain_interface: Optional[BlockchainInterface] = None
) -> DistributorAgentLG: # pragma: no cover
    return DistributorAgentLG(
        env=env,
        region_id=region_id,
        num_regions=num_regions,
        tools_instance=tools,
        openai_integration=openai_integration,
        inventory_stores=inventory_stores,
        manufacturer_usable_stores=manufacturer_usable_stores,
        memory_length=memory_length,
        verbose=verbose,
        console_obj=console_obj,
        blockchain_interface=blockchain_interface
    )