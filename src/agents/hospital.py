# src/agents/hospital.py
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

class HospitalAgentLG:
    def __init__(
        self,
        env: simpy.Environment,
        region_id: int,
        num_regions: int,
        tools_instance: PandemicSupplyChainTools,
        openai_integration: OpenAILLMIntegration,
        inventory_stores: Dict[Tuple[int, int], simpy.Container], # Its own stores
        distributor_inventory_stores: Dict[Tuple[int, int], simpy.Container], # For context (e.g. dist inv)
        memory_length: int = 3,
        verbose: bool = True,
        console_obj = None,
        blockchain_interface: Optional[BlockchainInterface] = None
    ):
        self.env = env
        self.agent_type = "hospital"
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
        self.distributor_inventory_stores = distributor_inventory_stores # For context if needed

    def _print(self, message: str): # pragma: no cover
        if self.verbose and self.console:
            self.console.print(message)

    def _create_decision_prompt(self, observation: Dict) -> str:
        """
        Hospital prompt (SimPy adapted, considering patient impact and updated observation structure).
        """
        current_day = observation.get("day", "N/A")
        agent_name = f"Hospital (Region {self.agent_id})"
        # Observation is the full SimPy-derived dict for this agent
        obs_summary_str = clean_observation_for_prompt(observation.copy(), max_len=4000)

        memory_summary = "My Recent History (newest first, max 2 shown):\n"
        if not self.memory: # pragma: no cover
            memory_summary += "  (No history yet)"
        else:
            recent = list(self.memory)[-2:]
            lines  = []
            for entry in reversed(recent):
                try:
                    agent_decisions = entry["decision"] # Output of decide()
                    # Hospital decide() returns: {"hospital_orders": {self.agent_id: validated_orders}}
                    orders_dict = agent_decisions.get("hospital_orders", {}).get(self.agent_id, {})
                    orders_str = {k: f"{float(v):.0f}" for k, v in orders_dict.items() if isinstance(v,(int,float,np.number))} if isinstance(orders_dict, dict) else "{}"
                    lines.append(f"- Day {entry.get('day', '?')+1}: MyOrdersToDist={orders_str}")
                except Exception as e: # pragma: no cover
                    lines.append(f"- Day {entry.get('day', '?')}: Error formatting memory ({e})")
            if lines:
                memory_summary += "\n".join(lines)
            else: # pragma: no cover
                memory_summary += "  (No valid history entries to display)"


        required_decisions = """
        You MUST provide decisions for 'hospital_orders': a JSON object mapping drug_id (string) to order quantity (number) requested FROM the distributor in your region.
        Your final response MUST be a single JSON object containing only the 'hospital_orders' key. Example:
        {"hospital_orders": {"0": 100.0, "1": 50.0}}
        Ensure keys are strings and values are numbers. Omit if 0. **CRITICAL: Ensure your *entire* final response is ONLY the valid JSON object and nothing else.**
        """
        available_tools = "[get_criticality_assessment, predict_disruptions, calculate_optimal_order, epidemic_forecast]"
        tool_guidance = f"""
        Available Tools: {available_tools}
        - Use `get_criticality_assessment` (pass `drug_id`) IF inventory for a drug is low relative to smoothed demand OR `my_region_patient_impact_score` is high.
        - Use `calculate_optimal_order` (pass `drug_id`) ONLY as a final *minor sanity check*. **DO NOT rely on it as the primary method.**
        - Use `epidemic_forecast` (pass `forecast_horizon`) if you need a more detailed case forecast than in `obs_clean.epidemiological_data`.
        - Max 1-2 tool calls per turn. STOP calling tools once you have determined order quantities based on Step 4.
        """

        prompt = f"""
You are the {agent_name} in a pandemic supply chain simulation on Day {current_day+1}.
Primary goal: Ensure drug **availability** for patients by maintaining adequate inventory buffers based on **smoothed demand signals** (available as `obs_clean.hosp_obs_smoothed_daily_demand_signal`), **criticality assessments**, and **patient impact score** (`obs_clean.my_region_patient_impact_score`).

**My Recent History Summary:**
{memory_summary}

**Current Observation (cleaned JSON, possibly truncated, refers to 'obs_clean' in scratchpad):**
{obs_summary_str}

**Your Process:**
1.  Analyze situation: Check inventory (`obs_clean.inventories`), pipeline (`obs_clean.inbound_pipeline`), `obs_clean.epidemiological_data` (cases, trend, projected demand for my region), `obs_clean.recent_actual_demand`, `obs_clean.stockout_history`, and `obs_clean.my_region_patient_impact_score`. Key pre-calculated demand signals for your ordering decision are `obs_clean.hosp_obs_demand_sum_planning_horizon`, `obs_clean.hosp_obs_avg_daily_demand_in_horizon`, and especially `obs_clean.hosp_obs_smoothed_daily_demand_signal`.
2.  Identify information needs: Is inventory low for critical drugs? Is patient impact high? If yes, call `get_criticality_assessment`.
3.  {tool_guidance}
4.  **Determine final order quantities using Robust Resilient Ordering Logic (see scratchpad).** Adjust safety stock targets based on demand trend, stockouts, and patient impact.
5.  **STOP & OUTPUT:** Once done, respond ONLY with the final JSON.

**Final Output Requirement:**
{required_decisions}
────────────────────────────── Scratchpad (private) ───────────────────────────────
# For each drug_id_str (e.g., "0", "1"):
# --- Lookup helpers (parse from 'Current observation' JSON string above) ---
# `my_inv[drug_id_str]` = float(obs_clean.inventories.get(drug_id_str, 0.0))
# `pipeline_to_me[drug_id_str]` = float(obs_clean.inbound_pipeline.get(drug_id_str, 0.0))
# `my_epi_data` = obs_clean.epidemiological_data # Specific to my region
# `proj_demand_today[drug_id_str]` = float(my_epi_data.get("projected_demand", {{}}).get(drug_id_str, 0.0)) # Still useful context
# `case_trend` = my_epi_data.get("case_trend_category", "stable")
# `current_cases` = int(my_epi_data.get("current_active_cases", 0))
# `my_demand_forecast_list[drug_id_str]` = obs_clean.daily_demand_forecast_list_for_my_needs.get(drug_id_str, [`proj_demand_today[d_id_str]`]*7) # Contextual
# `recent_actual_demand_val[drug_id_str]` = float(obs_clean.recent_actual_demand.get(drug_id_str, 0.0)) # Contextual
# `stockout_hist_list_for_drug[drug_id_str]` = [s for s in obs_clean.stockout_history if str(s.get("drug_id")) == drug_id_str] 
# `stockout_days_for_drug[drug_id_str]` = len(set(s.get('day') for s in stockout_hist_list_for_drug[drug_id_str]))
# `my_current_region_impact_score` = float(obs_clean.my_region_patient_impact_score)

# --- Step 4: Robust Resilient Ordering Logic (per drug_id_str) ---
`hospital_orders` = {{}}
`PLANNING_HORIZON_DAYS` = 4 # Assumed average lead time from my distributor (3 days) + Daily review (1 day)
`BASE_SAFETY_STOCK_DAYS` = 5
`MAX_ORDER_MULTIPLE_OF_HORIZON_DEMAND` = 3.0
`MIN_ORDER_FACTOR_AVG_DEMAND` = 0.25

For each `d_id_str` in `obs_clean.drug_info.keys()`:
    `drug_crit_val` = int(obs_clean.drug_info.get(d_id_str, {{}}).get("criticality_value", 1))
    `current_safety_days` = `BASE_SAFETY_STOCK_DAYS`
    `order_urgency_boost` = 1.0

    // Adjust safety stock days based on situation
    if `stockout_days_for_drug[d_id_str]` > 1: `current_safety_days` += 3; `order_urgency_boost` = 1.5;
    elif `stockout_days_for_drug[d_id_str]` > 0: `current_safety_days` += 1.5; `order_urgency_boost` = 1.2;
    
    if `case_trend` == "increasing_strongly": `current_safety_days` += 2.5; `order_urgency_boost` = max(`order_urgency_boost`, 1.4);
    elif `case_trend` == "increasing": `current_safety_days` += 1.0; `order_urgency_boost` = max(`order_urgency_boost`, 1.1);

    if `my_current_region_impact_score` > 300 and `drug_crit_val` >= 3 : `current_safety_days` += 3; `order_urgency_boost` = max(`order_urgency_boost`, 1.6)
    elif `my_current_region_impact_score` > 50 : `current_safety_days` += 1.5;

    `current_safety_days` = min(max(2.0, `current_safety_days`), 12.0) // Bounds for safety days

    // *** USE PRE-CALCULATED DEMAND SIGNALS FROM OBSERVATION ***
    // `demand_sum_planning_horizon` = float(obs_clean.hosp_obs_demand_sum_planning_horizon.get(d_id_str, 0.0)) // Not directly needed if using smoothed_daily_demand_signal for horizon demand
    // `avg_daily_demand_horizon` = float(obs_clean.hosp_obs_avg_daily_demand_in_horizon.get(d_id_str, 0.0)) // Not directly needed if using smoothed_daily_demand_signal
    `smoothed_daily_demand_signal` = float(obs_clean.hosp_obs_smoothed_daily_demand_signal.get(d_id_str, 0.0))
    // *** END PRE-CALCULATED USAGE ***

    `safety_stock_units_calculated` = `smoothed_daily_demand_signal` * `current_safety_days`
    // Demand over horizon is now based on the smoothed signal for consistency
    `demand_over_horizon_using_smoothed` = `smoothed_daily_demand_signal` * `PLANNING_HORIZON_DAYS`
    `order_up_to_target` = `demand_over_horizon_using_smoothed` + `safety_stock_units_calculated`
    
    `current_inventory_position` = `my_inv[d_id_str]` + `pipeline_to_me[d_id_str]`
    
    `calculated_order` = `order_up_to_target` - `current_inventory_position`
    `calculated_order` *= `order_urgency_boost` // Apply urgency boost

    `final_order_quantity` = 0.0
    if `calculated_order` > 0.01:
        `min_order_trigger` = max(1.0, `smoothed_daily_demand_signal` * `MIN_ORDER_FACTOR_AVG_DEMAND`)
        `final_order_quantity` = max(`calculated_order`, `min_order_trigger`)
        `final_order_quantity` = min(`final_order_quantity`, `smoothed_daily_demand_signal` * `PLANNING_HORIZON_DAYS` * `MAX_ORDER_MULTIPLE_OF_HORIZON_DEMAND`)
        `hospital_orders[d_id_str]` = round(max(0.0, `final_order_quantity`))

`final_json_output` = {{"hospital_orders": `hospital_orders`}}
────────────────────────── End Scratchpad ───────────────────────────────
"""
        return prompt.strip()

    def _apply_hard_constraints(self, decision_json: Dict, observation: Dict) -> Dict:
        validated_orders = {}
        drug_info_obs = observation.get("drug_info", {})
        defined_drug_ids_str = list(drug_info_obs.keys())

        raw_orders = decision_json.get("hospital_orders", {})
        if isinstance(raw_orders, dict):
            for drug_id_str, amount_val in raw_orders.items():
                if drug_id_str not in defined_drug_ids_str: # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Hospital R{self.agent_id}: Order for unknown drug_id {drug_id_str} ignored.[/]")
                    continue
                try:
                    validated_amount = max(0.0, float(amount_val)) # Ensure non-negative
                    if validated_amount > 0.01: # Only include if meaningful amount
                        validated_orders[drug_id_str] = validated_amount
                except (ValueError, TypeError): # pragma: no cover
                    self._print(f"[{Colors.WARNING}] Hospital R{self.agent_id}: Invalid order value for D{drug_id_str}. Value: {amount_val}[/]")
                    pass # Ignore invalid order values
        return {"hospital_orders": validated_orders}


    def decide(self, observation: Dict) -> Dict:
        agent_name = f"Hospital R{self.agent_id}"
        current_day_from_obs = observation.get('day', int(self.env.now))
        self._print(f"\n[{Colors.HOSPITAL}] {agent_name} LLM deciding (SimPy Day {current_day_from_obs + 1})...[/]")

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
            config = {"recursion_limit": 10} # Hospital decisions might involve a tool call or two
            final_state_from_graph = self.graph_app.invoke(initial_state, config=config)
            
            content_to_parse = None
            if final_state_from_graph and final_state_from_graph.get('messages'):
                last_message = final_state_from_graph['messages'][-1]
                if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
                    content_to_parse = last_message.content
                elif isinstance(last_message, ToolMessage) and self.verbose: # pragma: no cover
                     self._print(f"[{Colors.WARNING}] {agent_name}: Graph ended with ToolMessage. LLM might need more steps. Content: {last_message.content[:200]}[/{Colors.WARNING}]")
                     content_to_parse = '{"hospital_orders": {}}' # Fallback

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
           "hospital_orders" not in final_json_decision:
            self._print(f"[{Colors.FALLBACK}] {agent_name} using empty fallback decision (no 'hospital_orders' or parse issue).[/]")
            final_json_decision = {"hospital_orders": final_json_decision.get("hospital_orders", {})} # Ensure key exists

        validated_decision_parts = self._apply_hard_constraints(final_json_decision, observation)

        return_for_process = {
            "hospital_orders": {self.agent_id: validated_decision_parts.get("hospital_orders", {})}
        }
        if self.verbose: self._print(f"[{Colors.HOSPITAL}][Final Validated Decision] {agent_name}: {return_for_process}[/]")

        obs_summary_for_memory = clean_observation_for_prompt(observation.copy(), max_len=500)
        self.memory.append({
            "day": current_day_from_obs,
            "obs_summary": obs_summary_for_memory,
            "decision": return_for_process 
        })
        return return_for_process

# Factory function
def create_openai_hospital_agent(
    env: simpy.Environment,
    region_id: int,
    num_regions: int,
    tools: PandemicSupplyChainTools,
    openai_integration: OpenAILLMIntegration,
    inventory_stores: Dict[Tuple[int, int], Any], # Its own stores
    distributor_inventory_stores: Dict[Tuple[int, int], Any], # Context for its Dist's inv
    memory_length: int = 3,
    verbose: bool = True,
    console_obj = None,
    blockchain_interface: Optional[BlockchainInterface] = None
) -> HospitalAgentLG: # pragma: no cover
    return HospitalAgentLG(
        env=env,
        region_id=region_id,
        num_regions=num_regions,
        tools_instance=tools,
        openai_integration=openai_integration,
        inventory_stores=inventory_stores,
        distributor_inventory_stores=distributor_inventory_stores,
        memory_length=memory_length,
        verbose=verbose,
        console_obj=console_obj,
        blockchain_interface=blockchain_interface
    )