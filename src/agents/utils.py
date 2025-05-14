# --- START OF src/agents/utils.py (REVISED) ---

import json
from typing import Dict, Any
import numpy as np

def clean_observation_for_prompt(observation: Dict, max_len: int = 3500) -> str:
    """
    Creates a cleaned, truncated string representation of the observation
    suitable for LLM prompts. Preserves critical fields for agent decision-making
    and simplifies or removes others.

    Args:
        observation: The agent's observation dictionary.
        max_len: Maximum length of the output string.

    Returns:
        A cleaned and potentially truncated JSON string representation.
    """
    cleaned_obs = {}
    try:
        # Attempt a deep copy to avoid modifying the original dict
        # This is important so we don't alter the original observation object
        # that might be used elsewhere in the simulation.
        cleaned_obs = json.loads(json.dumps(observation))
    except TypeError as e:
        # Fallback to shallow copy if deep copy fails
        # This is less safe as changes might propagate, but better than crashing.
        print(f"[yellow]Warning: Observation deep copy failed ({e}). Using shallow copy for cleaning.[/yellow]")
        cleaned_obs = observation.copy() # Use .copy() for a shallow copy

    # --- Preserve or carefully modify specific critical fields ---

    # Preserve 'epidemiological_data' structure needed by agents
    # (Hospital, Distributor, and Manufacturer prompts all expect specific structures here)
    if 'epidemiological_data' in cleaned_obs and isinstance(cleaned_obs['epidemiological_data'], dict):
        # For Distributor and Hospital, 'epidemiological_data' often has a top-level 'projected_demand' dict.
        # For Manufacturer, 'epidemiological_data' is often a dict of region_ids, each with a 'projected_demand' dict.
        # This logic aims to preserve these structures without over-simplifying if they are already correct.
        # No specific removal needed here if the structures from _get_..._observation are correct.
        pass # Keep as is, rounding will handle numerics.
    elif 'epidemiological_data' in cleaned_obs: # If it exists but isn't a dict, remove to be safe
        cleaned_obs.pop('epidemiological_data', None)


    # Simplify 'drug_info' but ensure 'demand_factor' (or 'base_demand') is present
    if 'drug_info' in cleaned_obs and isinstance(cleaned_obs['drug_info'], dict):
        new_drug_info = {}
        for k, v_dict in cleaned_obs['drug_info'].items():
            if isinstance(v_dict, dict):
                # Prioritize 'demand_factor', fallback to 'base_demand', then to 0 if neither exists
                demand_val = v_dict.get('demand_factor', v_dict.get('base_demand', 0.0))
                new_drug_info[k] = {
                    'name': v_dict.get('name'),
                    'crit': v_dict.get('criticality_value'),
                    'demand_factor': demand_val # Ensure this key is consistently named
                }
        if new_drug_info:
            cleaned_obs['drug_info'] = new_drug_info
        else:
            cleaned_obs.pop('drug_info', None)

    # Keep these fields as they are crucial and already somewhat summarized by _get_..._observation methods
    # Their numerical values will be handled by the rounding logic later.
    # - inventories
    # - inbound_pipeline
    # - outbound_pipeline (for Distributor)
    # - hospital_stockout_summary (for Distributor)
    # - downstream_stockout_summary (for Manufacturer)
    # - downstream_projected_demand_summary (for Manufacturer)
    # - production_capacity (for Manufacturer)
    # - warehouse_inventories (for Manufacturer)

    # Simplify history fields by limiting the number of recent items
    # The prompts generally look at the last 2-3 days from these.
    history_limit = 5 # Keep up to 5 recent entries for context
    history_keys_to_trim = [
        'recent_orders',
        'recent_allocations',
        'demand_history',      # Hospital uses this
        'stockout_history',    # Hospital uses this
        'pending_releases'     # Manufacturer uses this
    ]
    for key in history_keys_to_trim:
        if key in cleaned_obs and isinstance(cleaned_obs[key], list):
            cleaned_obs[key] = cleaned_obs[key][-history_limit:]
        elif key in cleaned_obs and not isinstance(cleaned_obs[key], list):
            # If it's not a list but exists, remove it to avoid type errors in prompts
            cleaned_obs.pop(key, None)


    # Fields to completely remove if they are too verbose or not directly used by current prompts
    # (This list can be adjusted based on prompt needs)
    # Example: If 'region_info' or 'disruptions' (full list) become too verbose
    # fields_to_remove = ['disruptions_full_log', 'detailed_pipeline_history']
    # for key_to_remove in fields_to_remove:
    #    cleaned_obs.pop(key_to_remove, None)


    # --- Round all float values to 1 decimal place for conciseness ---
    def round_nested_floats(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: round_nested_floats(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [round_nested_floats(elem) for elem in item]
        elif isinstance(item, (float, np.floating)):
            try:
                return round(item, 1)
            except (ValueError, TypeError): # Handle potential non-numeric floats if any
                return item
        elif isinstance(item, (int, np.integer)): # Ensure integers are preserved as integers
             return int(item)
        # Keep other types (like strings, booleans) as they are
        return item

    cleaned_obs = round_nested_floats(cleaned_obs)

    # --- Convert to JSON string and handle truncation ---
    try:
        # Using sort_keys can help with consistency if LLM responses are cached
        # or for easier diffing of prompts, but not strictly necessary.
        json_string = json.dumps(cleaned_obs, indent=None, separators=(',', ':'), default=str, sort_keys=True)

        if len(json_string) > max_len:
            truncated_json = json_string[:max_len]
            # Attempt to find a good truncation point to keep JSON somewhat valid
            last_comma = truncated_json.rfind(',')
            last_brace = truncated_json.rfind('}')
            last_bracket = truncated_json.rfind(']')
            
            # Choose the latest of these separators, if found reasonably close to end
            cut_point = max(last_comma, last_brace, last_bracket)

            if cut_point > max_len * 0.8: # Only if a separator is found in the last 20%
                # Ensure we don't cut in the middle of a key or string value by finding the preceding quote
                final_cut = truncated_json[:cut_point].rfind('"')
                if final_cut != -1 and final_cut < cut_point -5 : # Heuristic: ensure it's not cutting too close
                    return truncated_json[:final_cut-1] + '"...[TRUNCATED]"}' # Try to close quotes and object
                else:
                    return truncated_json[:cut_point] + ' ...[TRUNCATED]"}'
            else: # If no good separator found, just hard truncate
                return truncated_json + '..."}' # Add a closing brace for some validity
        return json_string
    except TypeError as e:
        print(f"[red]Could not serialize cleaned observation to JSON: {e}. Using basic string representation.[/]")
        # Fallback to a very simple string representation if JSON serialization fails
        fallback_items = []
        if "day" in observation: fallback_items.append(f'"day":{observation["day"]}')
        if "inventories" in observation: fallback_items.append(f'"inventories":"{str(observation["inventories"])[:100]}..."') # Truncate inv display
        fallback_str = ",".join(fallback_items)
        return "{" + fallback_str[:max_len-5] + "...}" # Ensure it fits and indicates truncation

# --- END OF src/agents/utils.py (REVISED) ---