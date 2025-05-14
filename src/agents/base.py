# src/agents/base.py

from typing import List, Dict, TypedDict, Annotated, Sequence, Optional, Any
import operator
import json
import numpy as np
import time # For unique tool_call_id fallback

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Import tools and necessary interfaces
from src.tools import PandemicSupplyChainTools
# Import the production tool function directly
try:
    from src.tools.production import calculate_target_production_quantities_tool
except ImportError: # Should not happen if file structure is correct
    if hasattr(PandemicSupplyChainTools, 'calculate_target_production_quantities_tool'): # pragma: no cover
        calculate_target_production_quantities_tool = PandemicSupplyChainTools.calculate_target_production_quantities_tool
    else: # Fallback dummy if truly missing # pragma: no cover
        def calculate_target_production_quantities_tool(*args, **kwargs): 
            if console: 
                console.print("CRITICAL WARNING: calculate_target_production_quantities_tool not found! Using dummy.", style="bold red")
            return {"error": "Tool not implemented"}

try:
    from src.blockchain.interface import BlockchainInterface
except ImportError: # pragma: no cover
    BlockchainInterface = None

from src.llm.openai_integration import OpenAILLMIntegration

# --- Console and Colors ---
try:
    from config import console, Colors
except ImportError: # pragma: no cover
    import sys
    class MockConsole:
        def print(self, *args, **kwargs): print(*args, file=sys.stderr)
    console = MockConsole()
    class Colors:
        TOOL_OUTPUT = "yellow"; RED = "red"; ERROR = "bold red"
        YELLOW = "yellow"; DIM = "dim"; GREEN = "green"
        CYAN = "cyan"; GRAPH_FACTORY = "magenta"; WARNING = "yellow"; LLM = "cyan" # Added WARNING & LLM
# --- End Console and Colors ---


# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    observation: Dict 
    agent_type: str
    agent_id: Any 
    num_regions: int
    verbose: bool # For conditional logging within tool execution and other parts
    final_decision: Optional[Dict]

# --- Tool Executor Node Logic ---
def execute_tool(state: AgentState, tool_invocation: Dict, tools_instance: PandemicSupplyChainTools, bc_interface: Optional[BlockchainInterface]) -> ToolMessage:
    tool_name = tool_invocation.get("name")
    tool_args_from_llm = tool_invocation.get("args", {}) 
    tool_id = tool_invocation.get("id")

    color_yellow = getattr(Colors, "YELLOW", "yellow")
    if not tool_id: # pragma: no cover
         if console: console.print(f"[{color_yellow}]Warning: Tool invocation missing 'id'. Generating fallback. Tool: {tool_name}[/{color_yellow}]")
         tool_id = f"missing_id_{tool_name}_{time.time_ns()}"

    agent_type = state.get("agent_type", "UnknownAgent")
    agent_id_val = state.get("agent_id", "UnknownID")
    observation = state.get("observation", {}) 
    num_regions_from_state = state.get("num_regions", 0)
    current_day_from_state = observation.get('day', 0)
    is_verbose_from_state = state.get("verbose", False) # Get verbose flag

    output_str = f"Error: Tool '{tool_name}' execution failed or tool not found."
    output_content_for_message = {"error": output_str} 

    tool_map = {
        "get_blockchain_regional_cases": tools_instance.get_blockchain_regional_cases_tool,
        "epidemic_forecast": tools_instance.epidemic_forecast_tool,
        "predict_disruptions": tools_instance.disruption_prediction_tool,
        "get_criticality_assessment": tools_instance.criticality_assessment_tool,
        "calculate_optimal_order": tools_instance.optimal_order_quantity_tool,
        "calculate_allocation_priority": tools_instance.allocation_priority_tool,
        "calculate_target_production_quantities": calculate_target_production_quantities_tool,
    }
    func_to_call = tool_map.get(tool_name)

    if func_to_call:
        try:
            actual_tool_args_to_pass = tool_args_from_llm.copy()

            if tool_name == "get_blockchain_regional_cases":
                if agent_type != "manufacturer": # pragma: no cover
                    output_content_for_message = {"error": "Only Manufacturer can call get_blockchain_regional_cases."}
                elif not bc_interface: # pragma: no cover
                    output_content_for_message = {"error": "Blockchain interface not available."}
                else:
                    output_content_for_message = func_to_call(blockchain_interface=bc_interface, num_regions=num_regions_from_state)
            
            elif tool_name == "epidemic_forecast":
                days_to_forecast_llm = actual_tool_args_to_pass.get("forecast_horizon", 14)
                current_cases_val = 0.0
                case_history_list = []

                if agent_type == "manufacturer": # pragma: no cover
                    # For Manufacturer, LLM would need to specify which region's forecast via tool args.
                    # Current schema only has 'forecast_horizon'.
                    # If Manu calls this, it needs to be clear which region it's for or if it's an aggregate.
                    # Assuming the tool is for a *specific* region's forecast.
                    # This part might need schema adjustment for Manu to specify region_id.
                    output_content_for_message = {"error": "Epidemic forecast by Manufacturer needs a target region_id (not in current tool schema)."}
                else: # Distributor or Hospital
                    epi_data_for_agent = observation.get("epidemiological_data", {})
                    current_cases_val = float(epi_data_for_agent.get("current_active_cases", 0.0))
                    raw_hist_list = epi_data_for_agent.get("historical_active_cases_list", [])
                    case_history_list = [float(c) for c in raw_hist_list if isinstance(c, (int, float))]
                
                if agent_type != "manufacturer":
                     output_content_for_message = func_to_call(
                        current_cases=current_cases_val,
                        case_history=case_history_list,
                        days_to_forecast=int(days_to_forecast_llm)
                    )

            elif tool_name == "predict_disruptions":
                look_ahead_llm = actual_tool_args_to_pass.get("look_ahead_days", 14)
                hist_disruptions_from_obs = observation.get("all_scenario_disruptions", [])
                
                output_content_for_message = func_to_call(
                    historical_disruptions=hist_disruptions_from_obs,
                    current_day=int(current_day_from_state),
                    look_ahead_days=int(look_ahead_llm)
                )

            elif tool_name == "get_criticality_assessment":
                drug_id_arg_from_llm = tool_args_from_llm.get("drug_id")
                # LLM might optionally pass regional_patient_impact_score
                llm_provided_impact_score = tool_args_from_llm.get("regional_patient_impact_score")

                if drug_id_arg_from_llm is not None:
                    try:
                        drug_id = int(drug_id_arg_from_llm)
                        drug_id_str = str(drug_id)
                        drug_info_for_tool = observation.get("drug_info", {}).get(drug_id_str, {})
                        
                        if not drug_info_for_tool: # pragma: no cover
                            output_content_for_message = {"error": f"No drug_info found for drug_id {drug_id}."}
                        else:
                            stock_hist_to_use, demand_hist_to_use = [], []
                            # Fetch patient impact score from observation by default
                            obs_derived_regional_impact_score = None

                            if agent_type == "hospital":
                                stock_hist_to_use = observation.get("stockout_history", []) 
                                demand_hist_to_use = observation.get("demand_history", [])
                                obs_derived_regional_impact_score = observation.get("my_region_patient_impact_score")
                            elif agent_type == "distributor":
                                stock_hist_to_use = observation.get("downstream_hospital_stockout_history", [])
                                demand_hist_to_use = observation.get("downstream_hospital_demand_history", [])
                                obs_derived_regional_impact_score = observation.get("my_region_patient_impact_score")

                            stockout_history_for_this_drug = [s for s in stock_hist_to_use if s.get("drug_id") == drug_id]
                            unfulfilled_for_this_drug = sum(s.get("unfulfilled", 0.0) for s in stockout_history_for_this_drug)
                            total_demand_for_this_drug = sum(d.get("demand", 0.0) for d in demand_hist_to_use if d.get("drug_id") == drug_id)

                            tool_func_args_dict = {
                                "drug_info": drug_info_for_tool,
                                "stockout_history": stockout_history_for_this_drug,
                                "unfulfilled_demand": unfulfilled_for_this_drug,
                                "total_demand": max(1.0, total_demand_for_this_drug)
                            }
                            
                            # Decide which patient impact score to use: LLM provided or observation derived
                            final_impact_score_to_use = llm_provided_impact_score if llm_provided_impact_score is not None else obs_derived_regional_impact_score

                            if "regional_patient_impact_score" in func_to_call.__code__.co_varnames and final_impact_score_to_use is not None:
                                tool_func_args_dict["regional_patient_impact_score"] = float(final_impact_score_to_use)
                            
                            output_content_for_message = func_to_call(**tool_func_args_dict)
                    except (ValueError, TypeError) as e: # pragma: no cover
                        output_content_for_message = {"error": f"Data processing error for criticality assessment (Drug ID: {drug_id_arg_from_llm}). Details: {e}"}
                else: # pragma: no cover
                    output_content_for_message = {"error": "Missing 'drug_id' for get_criticality_assessment."}

            elif tool_name == "calculate_optimal_order":
                drug_id_arg_from_llm = tool_args_from_llm.get("drug_id")
                if drug_id_arg_from_llm is not None:
                    try:
                        drug_id = int(drug_id_arg_from_llm)
                        drug_id_str = str(drug_id)
                        inventory = float(observation.get("inventories", {}).get(drug_id_str, 0.0))
                        pipeline = float(observation.get("inbound_pipeline", {}).get(drug_id_str, 0.0))
                        
                        demand_forecast_list = []
                        if agent_type == "distributor":
                            demand_forecast_list = observation.get("downstream_hospital_demand_forecast", {}).get(drug_id_str, [])
                        elif agent_type == "hospital":
                            demand_forecast_list = observation.get("daily_demand_forecast_list_for_my_needs", {}).get(drug_id_str, [])
                            if not demand_forecast_list: # Fallback
                                hosp_proj_demand_today = float(observation.get("epidemiological_data", {}).get("projected_demand", {}).get(drug_id_str, 0.0))
                                demand_forecast_list = [hosp_proj_demand_today] * 7 

                        calculated_order_qty = func_to_call(
                            inventory_level=inventory,
                            pipeline_quantity=pipeline,
                            daily_demand_forecast=demand_forecast_list
                        )
                        output_content_for_message = {
                            "tool_name": tool_name,
                            "drug_id": drug_id,
                            "calculated_optimal_order": round(calculated_order_qty, 2)
                        }
                    except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                        output_content_for_message = {"error": f"Data processing error for optimal order (Drug ID: {drug_id_arg_from_llm}). Details: {e}"}
                else: # pragma: no cover
                    output_content_for_message = {"error": "Missing 'drug_id' for calculate_optimal_order."}
                    
            elif tool_name == "calculate_allocation_priority":
                if agent_type == "manufacturer":
                    # Arguments the LLM is REQUIRED to provide as per the updated schema
                    required_args_by_schema = ["drug_id", "region_requests", "available_inventory", "drug_info"]
                    missing_args = [arg for arg in required_args_by_schema if arg not in actual_tool_args_to_pass]

                    if not missing_args:
                        try:
                            # 1. Extract 'drug_info' directly. This will be the first positional argument.
                            tool_drug_info = actual_tool_args_to_pass["drug_info"] 
                            if not isinstance(tool_drug_info, dict): # Basic validation
                                raise TypeError(f"drug_info from LLM must be a dict, got {type(tool_drug_info)}")

                            # 2. Extract and process 'region_requests'.
                            # Ensure keys are int and values are float.
                            tool_region_requests_raw = actual_tool_args_to_pass["region_requests"]
                            if not isinstance(tool_region_requests_raw, dict): # Basic validation
                                raise TypeError(f"region_requests from LLM must be a dict, got {type(tool_region_requests_raw)}")
                            tool_region_requests = {
                                int(k): float(v) for k, v in tool_region_requests_raw.items()
                            }

                            # 3. Extract and process 'available_inventory'.
                            tool_available_inventory = float(actual_tool_args_to_pass["available_inventory"])

                            # 4. Extract and process 'region_cases' (optional in schema).
                            # If not provided by LLM, fetch from observation for all regions.
                            tool_region_cases = {}
                            if "region_cases" in actual_tool_args_to_pass and \
                               isinstance(actual_tool_args_to_pass["region_cases"], dict):
                                tool_region_cases = {
                                    int(k): int(v) for k, v in actual_tool_args_to_pass["region_cases"].items()
                                }
                                if is_verbose_from_state and console: # pragma: no cover
                                    color_info = getattr(Colors, "DIM", "grey50")
                                    console.print(f"[{color_info}][GRAPH_TOOL_EXEC] calculate_allocation_priority: Using region_cases provided by LLM.[/{color_info}]")
                            else:
                                # Fallback: if LLM doesn't provide region_cases, tool needs it.
                                # Fetch current cases for all regions from the observation.
                                for r_idx in range(num_regions_from_state):
                                    # Observation structure for Manufacturer: obs['epidemiological_data']['region_id_str']['current_active_cases']
                                    region_epi_data = observation.get("epidemiological_data", {}).get(str(r_idx), {})
                                    tool_region_cases[r_idx] = int(region_epi_data.get("current_active_cases", 0))
                                
                                if is_verbose_from_state and console: # pragma: no cover
                                    color_info = getattr(Colors, "DIM", "grey50")
                                    console.print(f"[{color_info}][GRAPH_TOOL_EXEC] calculate_allocation_priority: region_cases not provided by LLM, fetched from observation: {tool_region_cases}[/{color_info}]")
                            
                            # The 'drug_id' from LLM args can be used for logging or validation if needed,
                            # but it's not passed directly to the Python function if drug_info contains it.
                            # llm_provided_drug_id = int(actual_tool_args_to_pass["drug_id"])
                            # if int(tool_drug_info.get("id", -1)) != llm_provided_drug_id:
                            #     # Log a warning about potential mismatch if desired
                            #     pass

                            # Call the Python function with arguments in the correct positional order
                            output_content_for_message = func_to_call(
                                tool_drug_info,           # drug_info (dict)
                                tool_region_requests,     # region_requests (dict {int: float})
                                tool_region_cases,        # region_cases (dict {int: int})
                                tool_available_inventory  # available_inventory (float)
                            )
                        except (ValueError, TypeError, KeyError) as e: # pragma: no cover
                             output_content_for_message = {"error": f"Data type, key, or conversion error for allocation tool args: {type(e).__name__} - {e}. Args from LLM: {actual_tool_args_to_pass}"}
                    else: # pragma: no cover (LLM should be prompted to provide these as per schema)
                        output_content_for_message = {"error": f"Allocation priority tool for Manufacturer missing required arguments defined in schema: {', '.join(missing_args)}."}
                else: # pragma: no cover
                    output_content_for_message = {"error": "Allocation priority tool only for Manufacturer."}


            elif tool_name == "calculate_target_production_quantities":
                if agent_type == "manufacturer": 
                    simulation_context_arg = {
                        "num_regions": num_regions_from_state, "current_sim_day": int(current_day_from_state)
                    }
                    actual_tool_args_to_pass["simulation_context"] = simulation_context_arg
                    if "observation_subset" not in actual_tool_args_to_pass: # pragma: no cover
                        actual_tool_args_to_pass["observation_subset"] = {
                            "production_capacity": observation.get("production_capacity", {}),
                            "drug_info": observation.get("drug_info", {})
                        }
                        if console and is_verbose_from_state: # Use verbose flag from state
                             color_warning = getattr(Colors, "WARNING", "yellow")
                             console.print(f"[{color_warning}][GRAPH_TOOL_EXEC] WARNING: LLM did not provide 'observation_subset' for production tool. Using derived subset.[/{color_warning}]")
                    output_content_for_message = func_to_call(**actual_tool_args_to_pass)
                else: # pragma: no cover
                    output_content_for_message = {"error": "Production tool only for Manufacturer."}
            else: # pragma: no cover (This case should ideally not be reached if tool_map is complete)
                output_content_for_message = func_to_call(**actual_tool_args_to_pass)

            if isinstance(output_content_for_message, (dict, list)):
                 output_str = json.dumps(output_content_for_message, default=str)
            elif isinstance(output_content_for_message, float):
                 output_str = f"{output_content_for_message:.2f}"
            elif output_content_for_message is None: 
                 output_str = json.dumps({"error": "Tool executed but returned None."})
            else: 
                 output_str = str(output_content_for_message)

        except TypeError as te: 
            color_red = getattr(Colors, "RED", "red")
            error_detail = f"TypeError: {te}. Args LLM provided: {tool_args_from_llm}"
            if console: # pragma: no cover
                 console.print(f"[bold {color_red}][GRAPH_TOOL_EXEC] Error in tool '{tool_name}': {error_detail}[/bold {color_red}]")
            output_content_for_message = {"error": f"Tool '{tool_name}' call failed due to argument mismatch.", "details": error_detail}
            output_str = json.dumps(output_content_for_message)
        except Exception as e: 
            color_red = getattr(Colors, "RED", "red")
            error_detail = f"{type(e).__name__}: {e}"
            if console: # pragma: no cover
                 console.print(f"[bold {color_red}][GRAPH_TOOL_EXEC] Critical Error during tool '{tool_name}': {error_detail}. Args: {str(tool_args_from_llm)[:100]}...[/bold {color_red}]")
            output_content_for_message = {"error": f"Internal execution error in {tool_name}.", "details": error_detail}
            output_str = json.dumps(output_content_for_message)
    else: 
        color_red = getattr(Colors, "RED", "red")
        output_str = json.dumps({"error": f"Tool '{tool_name}' not found."})
        output_content_for_message = {"error": f"Tool '{tool_name}' not found."}
        if console: # pragma: no cover
            console.print(f"[{color_red}][GRAPH_TOOL_EXEC] {output_str}[/{color_red}]")

    color_tool_out = getattr(Colors, "TOOL_OUTPUT", "yellow")
    color_dim = getattr(Colors, "DIM", "dim")
    
    if console and is_verbose_from_state: # Use verbose flag from state
       console_msg_str = output_str
       if len(output_str) > 150: console_msg_str = output_str[:147] + "..." # pragma: no cover
       console.print(f"[{color_tool_out}][GRAPH_TOOL_RESULT] Tool {tool_name} for {agent_type}-{agent_id_val} (Day {current_day_from_state}) -> {console_msg_str}[/{color_tool_out}]", style=color_dim)

    return ToolMessage(content=output_str, name=tool_name, tool_call_id=tool_id)


# --- Graph Nodes ---
def call_model_node(state: AgentState, llm_integration: OpenAILLMIntegration, tools_schema: List[Dict]) -> Dict:
    messages = state['messages']
    response = llm_integration.invoke_llm_once(messages, tools=tools_schema)
    # Add response to state's messages
    return {"messages": [response]}

def call_tool_node(state: AgentState, tools_instance: PandemicSupplyChainTools, bc_interface: Optional[BlockchainInterface]) -> Dict:
    last_message = state['messages'][-1] if state['messages'] else None
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []} # No tools to call or last message not AIMessage with tool_calls

    tool_messages: List[BaseMessage] = []
    for tool_call in last_message.tool_calls:
        # Ensure tool_call is a valid dictionary structure before passing
        if isinstance(tool_call, dict) and "name" in tool_call and "args" in tool_call and "id" in tool_call:
             tool_message_obj = execute_tool(state, tool_call, tools_instance, bc_interface)
             tool_messages.append(tool_message_obj)
        else: # pragma: no cover
             # Handle malformed tool calls more gracefully
             error_tool_id = tool_call.get("id", f"malformed_tc_{time.time_ns()}") if isinstance(tool_call, dict) else f"malformed_tc_{time.time_ns()}"
             malformed_content = {"error": "Malformed tool_call structure received from LLM.", "received": str(tool_call)[:200]}
             if console and state.get("verbose", False):
                 color_red = getattr(Colors, "RED", "red")
                 console.print(f"[{color_red}][GRAPH_TOOL_NODE] Malformed tool call: {str(tool_call)[:100]}. ID: {error_tool_id}[/{color_red}]")
             tool_messages.append(ToolMessage(content=json.dumps(malformed_content), name="error_tool", tool_call_id=error_tool_id))
    return {"messages": tool_messages}

# --- Conditional Edge Logic ---
def should_continue_edge(state: AgentState) -> str:
    last_message = state['messages'][-1] if state['messages'] else None
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls and len(last_message.tool_calls) > 0:
            return "continue_to_tools"
        else: # LLM provided a response without tool calls, consider it final
            return "end_process" 
    elif isinstance(last_message, ToolMessage): # After tools, go back to LLM
        return "continue_to_llm" 
    elif isinstance(last_message, HumanMessage): # Should only be the first message
        return "continue_to_llm" # pragma: no cover
    
    # Fallback or if state is unexpected
    if console and state.get("verbose", False): # pragma: no cover
        color_yellow = getattr(Colors, "YELLOW", "yellow")
        console.print(f"[{color_yellow}][GRAPH_EDGE_WARN] Unexpected state in should_continue_edge. Last message: {type(last_message)}. Ending process.[/{color_yellow}]")
    return "end_process" # pragma: no cover

# --- Graph Factory Function ---
def create_agent_graph(llm_integration: OpenAILLMIntegration, tools_instance: PandemicSupplyChainTools, bc_interface: Optional[BlockchainInterface]):
    tools_schema = tools_instance.get_openai_tool_definitions()
    
    if not tools_schema and console: # pragma: no cover
        color_yellow = getattr(Colors, "YELLOW", "yellow")
        console.print(f"[{color_yellow}][GRAPH_FACTORY] Warning: No tool schemas defined. LLM cannot call tools.[/]")

    graph_builder = StateGraph(AgentState)
    # Bind the llm_integration and tools_schema to the call_model_node partial
    graph_builder.add_node("llm", lambda state_arg: call_model_node(state_arg, llm_integration, tools_schema))
    # Bind tools_instance and bc_interface to the call_tool_node partial
    graph_builder.add_node("tools", lambda state_arg: call_tool_node(state_arg, tools_instance, bc_interface))
    
    graph_builder.set_entry_point("llm")

    graph_builder.add_conditional_edges(
        "llm", # Source node
        should_continue_edge, # Function to decide next step
        { # Mapping from function's return value to node names
            "continue_to_tools": "tools",
            "end_process": END, # LangGraph's predefined end state
            "continue_to_llm": "llm" # This path should ideally not be hit from llm directly
        }
    )
    graph_builder.add_edge("tools", "llm") # After tools node, always go back to llm node

    try:
        agent_graph = graph_builder.compile()
        if console: # pragma: no cover
            color_graph = getattr(Colors, "GRAPH_FACTORY", "magenta") # Use a color for graph factory messages
            console.print(f"[{color_graph}][GRAPH_FACTORY] Agent graph compiled successfully.[/]")
        return agent_graph
    except Exception as e: # pragma: no cover
        if console:
            color_error = getattr(Colors, "ERROR", "bold red")
            console.print(f"[{color_error}][GRAPH_FACTORY] CRITICAL: Failed to compile agent graph: {type(e).__name__} - {e}[/{color_error}]")
            # For more detailed debugging of graph compilation issues:
            # console.print_exception(show_locals=True) 
        raise