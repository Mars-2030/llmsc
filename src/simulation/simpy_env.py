"""
Core SimPy environment for the pandemic supply chain simulation.
Integrates with LangGraph agents and optional blockchain for trusted data.
"""

import simpy
import numpy as np
import time
from typing import Dict, List, Optional, Any

from collections import defaultdict
import json # For deep copying history if needed for results

# Import from other project modules
from config import (
    console as global_console_config, Colors, # Rename to avoid conflict
    SIMPY_DEFAULT_WAREHOUSE_RELEASE_DELAY, SIMPY_DEFAULT_ALLOCATION_BATCH_FREQUENCY,
    COST_HOLDING_PER_UNIT_DAY, COST_BACKLOG_PER_UNIT, COST_BACKLOG_CRITICALITY_MULTIPLIER
)

# Import simulation process functions and helpers
from .processes import (
    manufacturer_process, distributor_process, hospital_process,
    warehouse_release_process, epidemic_and_demand_process,
    blockchain_daily_update_process
    # transport_process is usually called by other processes, not started independently here
)
from .helpers import (
    initialize_simpy_stores as calculate_initial_inventory_levels, # Renamed for clarity
    format_observation_for_agent,
    create_simpy_stores
)

# Import scenario and agent components
from src.scenario.generator import PandemicScenarioGenerator
from src.tools import PandemicSupplyChainTools
from src.llm.openai_integration import OpenAILLMIntegration

# Import agent factories
from src.agents import (
    create_openai_manufacturer_agent,
    create_openai_distributor_agent,
    create_openai_hospital_agent
)

# Optional import for blockchain
try:
    from src.blockchain.interface import BlockchainInterface
    from src.tools.allocation import allocation_priority_tool # For local fallback
except ImportError:
    BlockchainInterface = None
    allocation_priority_tool = None

class PandemicSupplyChainSimulation:
    """
    SimPy-based pandemic supply chain simulation.
    """

    def __init__(
        self,
        scenario_generator: PandemicScenarioGenerator,
        openai_integration: OpenAILLMIntegration,
        tools_instance: PandemicSupplyChainTools,
        blockchain_interface: Optional[BlockchainInterface] = None,
        use_blockchain: bool = False,
        num_regions: int = 3,
        num_drugs: int = 3,
        duration_days: int = 30,
        console_obj = None, # Use this for instance-specific console
        holding_cost_per_unit_day: float = COST_HOLDING_PER_UNIT_DAY,
        backlog_cost_per_unit: float = COST_BACKLOG_PER_UNIT,
        backlog_crit_multiplier: float = COST_BACKLOG_CRITICALITY_MULTIPLIER,
        warehouse_release_delay: float = SIMPY_DEFAULT_WAREHOUSE_RELEASE_DELAY,
        allocation_batch_frequency: int = SIMPY_DEFAULT_ALLOCATION_BATCH_FREQUENCY,
        verbose: bool = True,
    ):
        self.env = simpy.Environment()
        self.console = console_obj if console_obj else global_console_config
        self.scenario = scenario_generator
        self.openai_integration = openai_integration
        self.tools_instance = tools_instance # Store tools instance
        self.blockchain = blockchain_interface
        self.use_blockchain = use_blockchain and self.blockchain is not None
        if self.use_blockchain and allocation_priority_tool is None:
            self.console.print(f"[{Colors.WARNING}]Blockchain is enabled, but local allocation_priority_tool for fallback is not available.[/]")

        self.num_regions = num_regions
        self.num_drugs = num_drugs
        self.duration_days = duration_days
        self.verbose = verbose

        self.holding_cost_per_unit_day = holding_cost_per_unit_day
        self.backlog_cost_per_unit = backlog_cost_per_unit
        self.backlog_crit_multiplier = max(1.0, backlog_crit_multiplier)

        self.warehouse_release_delay = warehouse_release_delay
        self.allocation_batch_frequency = max(1, allocation_batch_frequency)

        # Calculate initial levels for stores
        calculated_initial_levels = calculate_initial_inventory_levels(self.scenario)

        # Create SimPy Stores (Containers) with initial levels
        stores = create_simpy_stores(
            self.env, self.num_regions, self.num_drugs, calculated_initial_levels
        )
        self.manu_usable_inv_stores = stores["manufacturer_usable"]
        self.warehouse_inv_stores = stores["warehouse"]
        self.dist_inv_stores = stores["distributor"]
        self.hosp_inv_stores = stores["hospital"]

        # For tracking items in transit: list of tuples
        # (drug_id, quantity, expected_arrival_simpy_time, destination_type_str, destination_region_id, origin_node_id_for_log)
        # destination_type_str: "distributor" or "hospital"
        # destination_region_id: region_id for the distributor or hospital
        # self.active_transports: List[Tuple[int, float, float, str, int, Any]] = []
        self.active_transports: List[Dict[str, Any]] = []
        # Each entry could be:
        # {
        #     "transport_id": unique_id, # e.g., generated with uuid or a simple counter
        #     "drug_id": int,
        #     "quantity": float,
        #     "departure_time": float,
        #     "expected_arrival_time": float,
        #     "destination_type": str, # "distributor", "hospital"
        #     "destination_entity_id": int, # region_id for distributor, hospital_id (which is also region_id in your model)
        #     "origin_entity_id": Any, # manufacturer_id (0), distributor_id (region_id+1)
        #     "status": "in_transit" # Could be "arrived", "delayed" etc. if more detail needed
        # }
        self.next_transport_id = 0 # Simple counter for unique IDs

        # Metrics & History
        self.stockouts = defaultdict(lambda: defaultdict(int))
        self.unfulfilled_demand_units = defaultdict(lambda: defaultdict(float))
        self.total_demand_units = defaultdict(lambda: defaultdict(float))
        self.patient_impact_score = defaultdict(float)

        self.demand_history = []
        self.order_history = []
        self.production_history = [] # Entries: {"day": int, "drug_id": int, "amount_ordered": float, "amount_produced": float, "capacity_at_time": float, "released": bool, "release_day": Optional[int]}
        self.allocation_history = []
        self.cost_history = [] # Entries: {"day": int, "holding_cost": float, "backlog_cost": float}
        self.inventory_history = {} # {day_int: {store_key_tuple_str: level_float}}
        self.blockchain_tx_log = []

        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0

        self.stockout_history = []

        self.current_regional_cases = {r: 0 for r in range(self.num_regions)}
        self.current_regional_projected_demand = {
            r: {d: 0.0 for d in range(self.num_drugs)} for r in range(self.num_regions)
        }
        self.manufacturer_pending_batch_allocations = defaultdict(lambda: defaultdict(float))

        self._initialize_agents() # Call after stores are created
        self._initialize_processes()
        self._record_daily_inventory_state() # Record state for day 0 (after init)

        if self.use_blockchain: # Initialize criticalities on blockchain after scenario is parsed
            self._initialize_blockchain_criticalities()


        if self.console:
            self.console.print(f"[{Colors.SIMPY}]SimPy simulation initialized.[/]")

    def _initialize_agents(self): # Renamed from _initialize_state_and_agents
        """Initialize LangGraph agents, passing SimPy env and store references."""
        # Create Agents
        self.manufacturer_agent_instance = create_openai_manufacturer_agent(
            env=self.env,
            tools=self.tools_instance,
            openai_integration=self.openai_integration,
            num_regions=self.num_regions,
            verbose=self.verbose, console_obj=self.console,
            blockchain_interface=self.blockchain,
            usable_inventory_stores=self.manu_usable_inv_stores,
            warehouse_inventory_stores=self.warehouse_inv_stores
        )
        self.distributor_agent_instances = [
            create_openai_distributor_agent(
                env=self.env, region_id=r, tools=self.tools_instance,
                openai_integration=self.openai_integration, num_regions=self.num_regions,
                verbose=self.verbose, console_obj=self.console, blockchain_interface=self.blockchain,
                inventory_stores=self.dist_inv_stores,
                manufacturer_usable_stores = self.manu_usable_inv_stores
            ) for r in range(self.num_regions)
        ]
        self.hospital_agent_instances = [
            create_openai_hospital_agent(
                env=self.env, region_id=r, tools=self.tools_instance,
                openai_integration=self.openai_integration, num_regions=self.num_regions,
                verbose=self.verbose, console_obj=self.console, blockchain_interface=self.blockchain,
                inventory_stores=self.hosp_inv_stores,
                distributor_inventory_stores=self.dist_inv_stores
            ) for r in range(self.num_regions)
        ]
        if self.console:
            self.console.print(f"[{Colors.SIMPY}]Agents initialized.[/]")

    def _initialize_blockchain_criticalities(self):
        # ... (same as previous version, ensures it's called after scenario is available) ...
        if not self.use_blockchain or not self.blockchain: return
        if self.console: self.console.print(f"[{Colors.BLOCKCHAIN}]Setting initial drug criticalities on blockchain...[/]")
        success = True
        for drug_data in self.scenario.drugs:
            drug_id = drug_data['id']
            crit_val = drug_data.get('criticality_value', 1)
            try:
                tx_receipt_dict = self.blockchain.set_drug_criticality(drug_id, crit_val)
                status = 'error'
                receipt_obj = None
                if tx_receipt_dict:
                    status = tx_receipt_dict.get('status', 'error')
                    receipt_obj = tx_receipt_dict.get('receipt')

                if status == 'success':
                    if self.verbose: self.console.print(f"  [{Colors.BLOCKCHAIN}]Drug {drug_id} criticality set to {crit_val}.[/]")
                else:
                    self.console.print(f"  [{Colors.ERROR}]Failed to set BC criticality for Drug {drug_id}. Status: {status}[/]")
                    success = False
                self.blockchain_tx_log.append({
                    "day": int(self.env.now), "type": "set_criticality", "drug_id": drug_id,
                    "status": status, "receipt": receipt_obj
                })
            except Exception as e:
                self.console.print(f"  [{Colors.ERROR}]Error setting BC criticality for Drug {drug_id}: {e}[/]")
                success = False
        if success and self.console: self.console.print(f"[{Colors.SUCCESS}]✓ Drug criticalities set on blockchain.[/]")


    def _initialize_processes(self):
        """Initialize and start all SimPy processes."""
        self.env.process(self._simulation_day_loop()) # Main loop for daily records
        self.env.process(epidemic_and_demand_process(self.env, self))
        self.env.process(warehouse_release_process(self.env, self))
        if self.use_blockchain:
            self.env.process(blockchain_daily_update_process(self.env, self))
        
        # Pass the imported allocation_priority_tool function as an argument
        # Ensure 'allocation_priority_tool' is defined (imported above)
        if allocation_priority_tool is None and not self.use_blockchain: # pragma: no cover
             self.console.print(f"[{Colors.WARNING}] CRITICAL: allocation_priority_tool is not available for manufacturer's local allocation logic, and blockchain is not used. Allocations may fail or use basic fallback.[/]")
        
        self.env.process(manufacturer_process(self.env, self, self.manufacturer_agent_instance, allocation_priority_tool_func=allocation_priority_tool))

        for r_id in range(self.num_regions):
            self.env.process(distributor_process(self.env, self, self.distributor_agent_instances[r_id], r_id))
            self.env.process(hospital_process(self.env, self, self.hospital_agent_instances[r_id], r_id))

        if self.console: self.console.print(f"[{Colors.SIMPY}]All SimPy processes initialized and started.[/]")

    def _simulation_day_loop(self):
        """The main daily loop that advances simulation time and triggers daily records."""
        for day_num_1_indexed in range(1, self.duration_days + 1):
            current_sim_time = self.env.now # Should be integer day number (0, 1, 2...)
            if self.console:
                self.console.rule(f"[{Colors.DAY_HEADER}] Global Start of Day {int(current_sim_time) + 1}/{self.duration_days} (SimTime: {current_sim_time:.2f})", style=Colors.DAY_HEADER)

            # Processes like epidemic_and_demand_process, agent processes, etc.,
            # are expected to run their logic for the current day and then yield self.env.timeout(1).
            # This loop ensures that after all those processes have conceptually finished their "day",
            # we record the state.

            # Yield a very small amount of time to ensure all processes that should run at env.now have run.
            # This is a bit of a subtlety in SimPy. If multiple processes yield timeout(0) or are
            # scheduled for the same time, their execution order is not guaranteed.
            # By yielding a tiny bit, we let the event queue process for the current integer time.
            yield self.env.timeout(0.001) # Let other processes for current integer day run

            # --- Actions that happen logically "at the end of the day" ---
            # The cost calculation and inventory recording should reflect the state
            # after all decisions and movements for the current day have been simulated.
            # The `epidemic_and_demand_process` is now responsible for daily backlog cost and appending to cost_history.
            # This function will update that entry with holding costs.
            self._calculate_and_record_daily_holding_cost()
            self._record_daily_inventory_state()

            # Now, advance SimPy time by the remainder of the day to trigger next day's events.
            # Since other processes yield timeout(1), this main loop also needs to advance.
            if day_num_1_indexed < self.duration_days: # Don't yield beyond the last day
                 yield self.env.timeout(1 - 0.001) # Advance to the next integer day
            else: # Last day, simulation ends
                 pass


        if self.console: self.console.print(f"[{Colors.SIMPY}]Simulation day loop finished after {self.duration_days} days (SimTime: {self.env.now:.2f}).[/]")


    def _record_daily_inventory_state(self):
        """Records inventory snapshots for the current day FROM SIMPY CONTAINERS."""
        day = int(self.env.now) # Current day (0-indexed)
        self.inventory_history[day] = {}
        try:
            for drug_id, store in self.manu_usable_inv_stores.items():
                self.inventory_history[day][(f"manu_usable_D{drug_id}")] = store.level
            for drug_id, store in self.warehouse_inv_stores.items():
                self.inventory_history[day][(f"warehouse_D{drug_id}")] = store.level
            for (r_id, drug_id), store in self.dist_inv_stores.items():
                self.inventory_history[day][(f"dist_R{r_id}_D{drug_id}")] = store.level
            for (r_id, drug_id), store in self.hosp_inv_stores.items():
                self.inventory_history[day][(f"hosp_R{r_id}_D{drug_id}")] = store.level
        except Exception as e:
            if self.console: self.console.print(f"[{Colors.ERROR}] Error recording inventory state on day {day}: {e}")


    def _calculate_and_record_daily_holding_cost(self):
        """Calculates daily holding costs and updates the cost_history entry for the day."""
        day = int(self.env.now)
        daily_holding_cost = 0.0

        for stores_dict_list in [self.manu_usable_inv_stores, self.warehouse_inv_stores, self.dist_inv_stores, self.hosp_inv_stores]:
            for store_instance in stores_dict_list.values(): # Iterate through store instances
                daily_holding_cost += max(0, store_instance.level) * self.holding_cost_per_unit_day
        
        self.total_holding_cost += daily_holding_cost

        # Find the cost_history entry for today (should have been added by epidemic_and_demand_process with backlog cost)
        # and update its holding_cost field.
        updated_today_cost_entry = False
        for entry in reversed(self.cost_history): # More efficient to check recent entries
            if entry["day"] == day:
                entry["holding_cost"] = daily_holding_cost
                updated_today_cost_entry = True
                break
        
        if not updated_today_cost_entry:
            # This case means epidemic_and_demand_process hasn't run for today yet to create the entry,
            # or it's day 0 before that process runs.
            # This implies a potential ordering issue in processes or this function is called too early.
            # For robustness, create the entry if missing, but this signals a deeper look might be needed.
            self.cost_history.append({
                "day": day,
                "holding_cost": daily_holding_cost,
                "backlog_cost": 0 # Will be filled/overwritten by epidemic_and_demand_process if it runs later for this day
            })
            if self.verbose and day > 0: # Don't warn for day 0 initialization
                 self.console.print(f"[{Colors.WARNING}] Day {day}: Cost history entry for holding cost created; backlog cost pending from demand process.[/]")


    def _calculate_final_results(self) -> Dict:
        # ... (Same as previous version, ensure all history lists are correctly populated) ...
        final_stockouts = {str(k): dict(v) for k, v in self.stockouts.items()}
        final_unfulfilled = {str(k): dict(v) for k, v in self.unfulfilled_demand_units.items()}
        final_total_demand = {str(k): dict(v) for k, v in self.total_demand_units.items()}
        final_patient_impact = {str(k): v for k, v in self.patient_impact_score.items()}

        # Ensure inventory history keys are strings for JSON serialization if complex tuples were used.
        # My _record_daily_inventory_state now uses stringified tuple keys.
        serializable_inv_history = {}
        for day, day_data in self.inventory_history.items():
            serializable_inv_history[day] = {str(key_tuple): val for key_tuple, val in day_data.items()}


        results = {
            "total_stockouts": final_stockouts,
            "total_unfulfilled_demand": final_unfulfilled,
            "patient_impact": final_patient_impact,
            "total_demand": final_total_demand,
            "scenario_regions": self.scenario.regions,
            "scenario_drugs": self.scenario.drugs,
            "config_warehouse_delay": self.warehouse_release_delay,
            "config_allocation_frequency": self.allocation_batch_frequency,
            "total_holding_cost": self.total_holding_cost,
            "total_backlog_cost": self.total_backlog_cost,
            "environment_history": {
                "demand_history": self.demand_history,
                "order_history": self.order_history,
                "production_history": self.production_history,
                "allocation_history": self.allocation_history,
                "cost_history": self.cost_history,
                "inventory_history": serializable_inv_history, # Use serializable version
                "stockout_history": self.stockout_history,
                "scenario_length": self.duration_days
            },
            "blockchain_tx_log": self.blockchain_tx_log
        }
        return results

    def run(self) -> Dict:
        """Run the SimPy simulation environment."""
        if self.console: self.console.print(f"[{Colors.SIMPY}]Starting SimPy environment run until day {self.duration_days}...[/]")
        # The processes are already started in __init__. env.run() executes them.
        self.env.run(until=self.duration_days) # Run for specified number of time units (days)
        if self.console: self.console.print(f"[{Colors.SIMPY}]SimPy environment run finished at time {self.env.now:.2f}.[/]")
        return self._calculate_final_results()