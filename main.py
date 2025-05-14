#!/usr/bin/env python3
"""
Pandemic Supply Chain Simulation - Main Entry Point (SimPy + LangGraph + Blockchain)
"""

import argparse
import time
import datetime
import os
import numpy as np # For results display
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Configuration and utilities
from config import (
    console, Colors, ensure_folder_exists, save_console_html, OPENAI_API_KEY,
    NODE_URL, CONTRACT_ADDRESS, CONTRACT_ABI_PATH, BLOCKCHAIN_PRIVATE_KEY,
    check_blockchain_config
)

# Scenario generator
from src.scenario.generator import PandemicScenarioGenerator

# Visualization (Assuming these are updated to work with SimPy simulation results)
from src.visualization.metrics_viz import (
    track_service_levels, visualize_service_levels, visualize_performance,
    visualize_inventory_levels, visualize_blockchain_performance,
    calculate_bullwhip_effect, visualize_costs
)
from src.visualization.scenario_viz import (
    visualize_epidemic_curves, visualize_drug_demand, visualize_disruptions,
    visualize_sir_components, visualize_sir_simulation
)

# LLM Integration and Tools
from src.tools import PandemicSupplyChainTools # Import the class
from src.llm.openai_integration import OpenAILLMIntegration

# SimPy Simulation Core Class
from src.simulation.simpy_env import PandemicSupplyChainSimulation

# Blockchain (optional)
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None


def run_simpy_simulation_entry( # Renamed to avoid conflict if old main had same name
    console: Console,
    openai_api_key: str,
    num_regions: int = 3,
    num_drugs: int = 3,
    simulation_days: int = 30,
    pandemic_severity: float = 0.8,
    disruption_probability: float = 0.1,
    warehouse_release_delay: float = 1.0, # SimPy uses float for time
    allocation_batch_frequency: int = 1,
    model_name: str = "gpt-4o",
    visualize: bool = True,
    verbose: bool = True,
    output_folder: str = "output_simpy", # Default output folder for SimPy version
    blockchain_interface = None, # Passed in instance
    use_blockchain: bool = False,
    holding_cost_per_unit_day: float = 0.005,
    backlog_cost_per_unit: float = 5.0,
    backlog_crit_multiplier: float = 2.0
):
    """Run simulation with SimPy + LangGraph agents."""

    sim_type = "SimPy + LangGraph" + (" + Blockchain" if use_blockchain else "")
    console.print(f"[bold]Initializing {sim_type} pandemic supply chain simulation...[/]")

    # --- Scenario and Core Components Setup ---
    scenario_generator = PandemicScenarioGenerator(
        console=console, num_regions=num_regions, num_drugs=num_drugs,
        scenario_length=simulation_days, pandemic_severity=pandemic_severity,
        disruption_probability=disruption_probability,
    )

    try:
        openai_integration = OpenAILLMIntegration(openai_api_key, model_name, console=console)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize OpenAI Integration: {e}. Aborting simulation.[/]")
        return None

    tools_instance = PandemicSupplyChainTools()

    # --- Create and Run SimPy Simulation ---
    try:
        simulation = PandemicSupplyChainSimulation(
            scenario_generator=scenario_generator,
            openai_integration=openai_integration,
            tools_instance=tools_instance, # Pass tools instance
            blockchain_interface=blockchain_interface,
            use_blockchain=use_blockchain,
            num_regions=num_regions,
            num_drugs=num_drugs,
            duration_days=simulation_days,
            console_obj=console,
            warehouse_release_delay=warehouse_release_delay,
            allocation_batch_frequency=allocation_batch_frequency,
            holding_cost_per_unit_day=holding_cost_per_unit_day,
            backlog_cost_per_unit=backlog_cost_per_unit,
            backlog_crit_multiplier=backlog_crit_multiplier,
            verbose=verbose,
            # model_name is part of openai_integration now
        )
        console.print(f"[bold]Running SimPy simulation for {simulation_days} days using {model_name}...[/]")
        start_time = time.time()

        # The run method of PandemicSupplyChainSimulation executes env.run() and returns results
        results = simulation.run()

        end_time = time.time()
        console.rule(f"\n[bold]Simulation complete. Total time: {end_time - start_time:.2f} seconds.[/]")

    except Exception as e:
         console.print(f"[bold red]Error creating/running SimPy simulation: {e}. Aborting.[/]")
         console.print_exception()
         return None


    # --- Generate Visualizations (If Requested) ---
    # Visualization functions now need to work with the 'simulation' object or its 'results'
    if visualize:
        console.print("[bold]Generating visualizations...[/]")
        try:
            # Scenario visualizations (use scenario_generator)
            visualize_epidemic_curves(scenario_generator, output_folder, console=console)
            visualize_drug_demand(scenario_generator, output_folder, console=console)
            visualize_disruptions(scenario_generator, output_folder, console=console)
            visualize_sir_components(scenario_generator, output_folder, console=console)
            for region_id in range(min(3, num_regions)): # Limit detailed SIR plots
                 if scenario_generator.epidemic_curves.get(region_id) is not None:
                    visualize_sir_simulation(scenario_generator, region_id, output_folder, console=console)

            # Performance visualizations (pass the simulation object or its results structure)
            # Assuming visualization functions are updated to handle the new data source
            visualize_performance(simulation, output_folder, console=console) # Pass simulation object
            visualize_inventory_levels(simulation, output_folder, console=console) # Pass simulation object
            visualize_service_levels(simulation, output_folder, console=console) # Pass simulation object

            # Cost and Bullwhip visualization (pass results['environment_history'] or similar from SimPy results)
            if "environment_history" in results: # Check if this key exists in SimPy results
                visualize_costs(results["environment_history"], output_folder, console)
                # Bullwhip might also need data from results["environment_history"]
                bullwhip_metrics = calculate_bullwhip_effect(results["environment_history"])
                # Display bullwhip_metrics if needed (see old main.py for example)
                if bullwhip_metrics: # Check if metrics were successfully calculated
                    console.print("\n[bold cyan]Bullwhip Effect Ratios (Variance of Orders / Variance of Downstream Demand/Orders):[/]")
                    # Create a Rich Table for bullwhip metrics
                    bw_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE, padding=(0, 1))
                    bw_table.add_column("Echelon Comparison", style="cyan", min_width=35)
                    bw_table.add_column("Bullwhip Ratio", style="white", justify="right", min_width=15)
                    bw_table.add_column("Interpretation", style="dim", min_width=40)

                    for key, value in bullwhip_metrics.items():
                        # Determine color and interpretation based on the ratio value
                        # A ratio > 1 indicates amplification (bullwhip effect)
                        # A ratio < 1 indicates smoothing / dampening
                        # A ratio = 1 indicates perfect matching (no amplification or smoothing)
                        color = "default"
                        interpretation_text = ""
                        if value > 1.15: # Significantly amplified
                            color = "bold red"
                            interpretation_text = "Significant amplification (Bullwhip)"
                        elif value > 1.05: # Slightly amplified
                            color = "yellow"
                            interpretation_text = "Slight amplification"
                        elif value < 0.95: # Smoothing effect
                            color = "cyan"
                            interpretation_text = "Smoothing effect (Dampening)"
                        elif value < 0.85: # Significant smoothing
                            color = "bold cyan"
                            interpretation_text = "Significant smoothing"
                        else: # Close to 1
                            color = "green"
                            interpretation_text = "Variances closely matched"
                        
                        # Add row to the table
                        bw_table.add_row(key, f"[{color}]{value:.2f}[/]", interpretation_text)
                    
                    console.print(bw_table)
                    console.print(f"  [{Colors.DIM}]Note: Ratios > 1 indicate the bullwhip effect (order variance is higher than demand variance).[/]")
                    console.print(f"  [{Colors.DIM}]Ratios < 1 indicate a dampening of variability.[/]")
                else:
                    # This 'else' corresponds to 'if bullwhip_metrics:'
                    console.print(f"\n[{Colors.YELLOW}]Could not calculate Bullwhip Effect (insufficient data or error in calculation).[/]")
            else:
        # This 'else' corresponds to 'if "environment_history" in sim_results:'
                console.print(f"\n[{Colors.YELLOW}]Skipping Bullwhip Effect calculation: 'environment_history' not found in simulation results.[/]")

            # Blockchain visualization
            if use_blockchain and blockchain_interface:
                visualize_blockchain_performance(blockchain_interface, output_folder, console=console)

        except Exception as e:
            console.print(f"[red]Error during visualization: {e}[/]")
            console.print_exception(show_locals=False)

    return results, simulation # Return both results dict and simulation object for full data access

# --- Main Execution Block ---
if __name__ == "__main__":
    openai_key = OPENAI_API_KEY
    if not openai_key or openai_key == "YOUR_OPENAI_API_KEY":
         console.print("[bold red]Error: OPENAI_API_KEY not found or not set in .env file. Exiting.[/]")
         exit(1)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Run SimPy-based pandemic supply chain simulation")
    # Arguments (largely same as before)
    parser.add_argument("--regions", type=int, default=3, help="Number of regions")
    parser.add_argument("--drugs", type=int, default=3, help="Number of drugs")
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--severity", type=float, default=0.8, help="Pandemic severity (0-1)")
    parser.add_argument("--disrupt-prob", type=float, default=0.1, help="Base disruption probability factor")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--no-viz", action="store_false", dest="visualize", help="Disable visualizations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Less verbose output (Agent internal logs)")
    # parser.add_argument("--no-colors", action="store_false", dest="use_colors", help="Disable colored output") # Rich handles this via NO_COLOR env var
    parser.add_argument("--folder", type=str, default="output_simpy", help="Base folder for simulation output") # Changed default folder
    parser.add_argument("--warehouse-delay", type=float, default=1.0, help="Warehouse release delay (days) - can be float for SimPy")
    parser.add_argument("--allocation-batch", type=int, default=1, help="Allocation batch frequency (days, 1=daily)")
    parser.add_argument("--use-blockchain", action="store_true", default=False, help="Enable blockchain integration")
    parser.add_argument("--holding-cost", type=float, default=0.005, help="Cost per unit per day for holding inventory")
    parser.add_argument("--backlog-cost", type=float, default=5.0, help="Base cost per unit for unfulfilled demand (backlog)")
    parser.add_argument("--backlog-crit-multiplier", type=float, default=2.0, help="Multiplier for backlog cost based on drug criticality")
    args = parser.parse_args()

    # Create timestamped output folder
    output_folder_path = f"{args.folder}_{timestamp}_regions{args.regions}_drugs{args.drugs}_days{args.days}"
    if args.use_blockchain: output_folder_path += "_blockchain"
    console.print(Panel("[bold white]🦠 PANDEMIC SUPPLY CHAIN SIMULATION (SimPy + LangGraph Agents) 🦠[/]", border_style="blue", expand=False, padding=(1,2)))

    # Config Table (same as before)
    config_table = Table(title="Simulation Configuration", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan"); config_table.add_column("Value", style="white")
    config_table.add_row("Regions", str(args.regions)); config_table.add_row("Drugs", str(args.drugs))
    config_table.add_row("Simulation Days", str(args.days)); config_table.add_row("Pandemic Severity", f"{args.severity:.2f}")
    config_table.add_row("Disruption Probability Factor", f"{args.disrupt_prob:.2f}")
    config_table.add_row("Warehouse Delay", f"{args.warehouse_delay} days")
    config_table.add_row("Allocation Batch Frequency", f"{args.allocation_batch} days" if args.allocation_batch > 1 else "Daily")
    config_table.add_row("LLM Model", args.model)
    config_table.add_row("Visualizations", "Enabled" if args.visualize else "Disabled")
    config_table.add_row("Verbose Output", "Enabled" if args.verbose else "Disabled")
    config_table.add_row("Output Folder", output_folder_path)
    config_table.add_row("Blockchain", "[bold green]Enabled[/]" if args.use_blockchain else "Disabled")
    if args.use_blockchain:
         config_table.add_row("  Node URL", NODE_URL)
         config_table.add_row("  Contract Address", CONTRACT_ADDRESS or "[red]Not Set[/]")
         config_table.add_row("  ABI Path", CONTRACT_ABI_PATH)
         config_table.add_row("  Signer Key Loaded", "[green]Yes[/]" if BLOCKCHAIN_PRIVATE_KEY else "[red]No[/]")
    config_table.add_row("Holding Cost (/unit/day)", f"${args.holding_cost:.4f}")
    config_table.add_row("Backlog Cost (/unit, base)", f"${args.backlog_cost:.2f}")
    config_table.add_row("Backlog Criticality Multiplier", f"{args.backlog_crit_multiplier:.1f}x")
    console.print(config_table); console.print()

    output_folder = ensure_folder_exists(console, output_folder_path)

    # --- INITIALIZE BLOCKCHAIN INTERFACE ---
    blockchain_interface_instance = None
    actual_use_blockchain_flag = False
    if args.use_blockchain:
        console.print("\n[bold cyan]Attempting Blockchain Integration...[/]")
        if BlockchainInterface is None:
             console.print("[bold red]❌ Blockchain support not available (missing dependencies like web3?). Halting.[/]")
             exit(1)
        if not check_blockchain_config():
             console.print("[bold red]❌ Blockchain configuration incomplete in .env or ABI file missing. Halting.[/]")
             exit(1)
        try:
            blockchain_interface_instance = BlockchainInterface(
                node_url=NODE_URL, contract_address=CONTRACT_ADDRESS,
                contract_abi_path=CONTRACT_ABI_PATH, private_key=BLOCKCHAIN_PRIVATE_KEY
            )
            actual_use_blockchain_flag = True
            console.print(f"[bold green]✓ Connected to Ethereum node and loaded contract.[/]")
        except Exception as e:
            console.print(f"[bold red]❌ FATAL ERROR: Could not initialize Blockchain Interface: {e}[/]")
            console.print_exception()
            try: save_console_html(console, output_folder=output_folder, filename="simulation_error_report_simpy.html")
            except Exception as save_e: console.print(f"[red]Could not save error report: {save_e}[/]")
            exit(1)
    else:
        actual_use_blockchain_flag = False
        console.print("\n[yellow]Blockchain integration disabled by command-line argument.[/]")
    console.print("-" * 30)

    # --- Run Simulation ---
    # run_simpy_simulation_entry returns (results_dict, simulation_object)
    sim_results, simulation_object = run_simpy_simulation_entry(
        console=console,
        openai_api_key=openai_key,
        num_regions=args.regions, num_drugs=args.drugs, simulation_days=args.days,
        pandemic_severity=args.severity, disruption_probability=args.disrupt_prob,
        warehouse_release_delay=args.warehouse_delay, allocation_batch_frequency=args.allocation_batch,
        model_name=args.model, visualize=args.visualize, verbose=args.verbose,
        output_folder=output_folder,
        blockchain_interface=blockchain_interface_instance,
        use_blockchain=actual_use_blockchain_flag,
        holding_cost_per_unit_day=args.holding_cost,
        backlog_cost_per_unit=args.backlog_cost,
        backlog_crit_multiplier=args.backlog_crit_multiplier
    )

    if sim_results is None:
         console.print("[bold red]Simulation run failed to produce results. Check logs above for errors.[/]")
         html_filename_err = "simulation_report_simpy_FAILED" + ("_blockchain" if actual_use_blockchain_flag else "") + ".html"
         save_console_html(console, output_folder=output_folder, filename=html_filename_err)
         exit(1)


    # --- Display Results (Adapted from old main.py) ---
    console.print(Panel("[bold white]Simulation Results Summary (SimPy)[/]", border_style="green", expand=False))
    drug_names = {d['id']: d['name'] for d in sim_results.get('scenario_drugs', [])}
    region_names = {r['id']: r['name'] for r in sim_results.get('scenario_regions', [])}

    # Stockouts Summary
    console.print("\n[bold red]Total Stockout Days by Drug and Region:[/]")
    stockout_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    stockout_table_summary.add_column("Drug", style="cyan", min_width=10)
    stockout_table_summary.add_column("Region", style="magenta", min_width=10)
    stockout_table_summary.add_column("Stockout Days", style="white", justify="right", min_width=15)
    total_stockout_days = 0
    if "total_stockouts" in sim_results and isinstance(sim_results["total_stockouts"], dict):
        for drug_id_str, regions_data in sim_results["total_stockouts"].items():
            try: drug_id = int(drug_id_str)
            except ValueError: continue
            drug_name = drug_names.get(drug_id, f"Drug {drug_id}")
            if isinstance(regions_data, dict):
                for region_id_str, count in regions_data.items():
                    try: region_id = int(region_id_str)
                    except ValueError: continue
                    region_name = region_names.get(region_id, f"Region {region_id}")
                    if count > 0:
                        color = "red" if count > (args.days * 0.3) else "yellow"
                        stockout_table_summary.add_row(drug_name, region_name, f"[{color}]{count}[/]")
                        total_stockout_days += count
    if total_stockout_days == 0:
         console.print("[green]✓ No stockout days recorded across all regions and drugs.[/]")
    else:
         console.print(stockout_table_summary)
         stockout_severity_threshold = (args.days * args.regions * args.drugs) * 0.05
         color = "red" if total_stockout_days > stockout_severity_threshold * 2 else "yellow" if total_stockout_days > 0 else "green"
         console.print(f"  Total stockout days across system: [bold {color}]{total_stockout_days}[/]")

    # Unfulfilled Demand Summary
    total_unfulfilled = sum(sum(drug_data.values()) for drug_data in sim_results.get("total_unfulfilled_demand", {}).values() if isinstance(drug_data, dict))
    total_demand_all = sum(sum(drug_data.values()) for drug_data in sim_results.get("total_demand", {}).values() if isinstance(drug_data, dict))
    percent_unfulfilled_str = f" ({ (total_unfulfilled / total_demand_all * 100) if total_demand_all > 0 else 0 :.1f}%)" if total_demand_all > 0 else ""
    color = "red" if total_unfulfilled > total_demand_all * 0.1 else "yellow" if total_unfulfilled > 0 else "green"
    console.print(f"\n[bold]Total Unfulfilled Demand (units): [{color}]{total_unfulfilled:.1f}[/{color}]{percent_unfulfilled_str}")

    # Patient Impact Summary
    console.print("\n[bold red]Patient Impact Score by Region:[/]")
    impact_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    impact_table_summary.add_column("Region", style="magenta", min_width=10)
    impact_table_summary.add_column("Impact Score", style="white", justify="right", min_width=15)
    total_impact = sum(sim_results.get("patient_impact", {}).values())
    if isinstance(sim_results.get("patient_impact"), dict):
        for region_id_str, impact in sim_results["patient_impact"].items():
            try: region_id = int(region_id_str)
            except ValueError: continue
            region_name = region_names.get(region_id, f"Region {region_id}")
            impact_color = "red" if impact > 1000 * args.days / 30 else "yellow" if impact > 100 * args.days / 30 else "green"
            impact_table_summary.add_row(region_name, f"[{impact_color}]{impact:.1f}[/]")
    console.print(impact_table_summary)
    color = "red" if total_impact > 5000 * args.days / 30 else "yellow" if total_impact > 500 * args.days / 30 else "green"
    console.print(f"  Total Patient Impact Score: [bold {color}]{total_impact:.1f}[/]")

    # Service Level Summary
    console.print("\n[bold cyan]Service Level Performance (% Demand Met):[/]")
    service_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    service_table_summary.add_column("Metric", style="cyan", min_width=15)
    service_table_summary.add_column("Service Level", style="white", justify="right", min_width=15)
    service_levels_data = track_service_levels(simulation_object) # Pass SimPy sim object
    if service_levels_data:
         avg_service = np.mean([item["service_level"] for item in service_levels_data]) if service_levels_data else 0
         min_service = min(item["service_level"] for item in service_levels_data) if service_levels_data else 0
         final_service = service_levels_data[-1]["service_level"] if service_levels_data else 0
         def get_service_color(level): return "green" if level >= 95 else "cyan" if level >= 90 else "yellow" if level >= 80 else "red"
         service_table_summary.add_row("Average", f"[{get_service_color(avg_service)}]{avg_service:.1f}%[/]")
         service_table_summary.add_row("Minimum", f"[{get_service_color(min_service)}]{min_service:.1f}%[/]")
         service_table_summary.add_row("Final Day", f"[{get_service_color(final_service)}]{final_service:.1f}%[/]")
         console.print(service_table_summary)
    else:
         console.print("[yellow]No service level data calculated.[/]")

    # Cost Summary
    console.print("\n[bold cyan]Cost Summary:[/]")
    cost_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", style="white", justify="right")
    total_holding = sim_results.get("total_holding_cost", 0.0)
    total_backlog = sim_results.get("total_backlog_cost", 0.0)
    total_system_cost = total_holding + total_backlog
    cost_table.add_row("Total Holding Cost", f"${total_holding:,.2f}")
    cost_table.add_row("Total Backlog Cost", f"${total_backlog:,.2f}")
    cost_table.add_row("[bold]Total System Cost[/]", f"[bold]${total_system_cost:,.2f}[/]")
    console.print(cost_table)

    # Bullwhip Effect
    if "environment_history" in sim_results:
        bullwhip_metrics = calculate_bullwhip_effect(sim_results["environment_history"])
        if bullwhip_metrics:
            console.print("\n[bold cyan]Bullwhip Effect Ratios (Var(Orders)/Var(Demand)):[/]")
            bw_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            bw_table.add_column("Level", style="cyan")
            bw_table.add_column("Ratio", style="white", justify="right")
            for key, value in bullwhip_metrics.items():
                color = "red" if value > 1.5 else "yellow" if value > 1.1 else "green"
                bw_table.add_row(key, f"[{color}]{value:.2f}[/]")
            console.print(bw_table)
        else:
            console.print("\n[yellow]Could not calculate Bullwhip Effect (insufficient history/data).[/]")


    # Blockchain Performance Metrics (same as before)
    if actual_use_blockchain_flag and blockchain_interface_instance:
        console.print(Panel("[bold white]Blockchain Performance Metrics[/]", border_style=Colors.BLOCKCHAIN, expand=False))
        bc_metrics = blockchain_interface_instance.get_performance_metrics()
        perf_table = Table(title="Blockchain Interaction Summary", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        perf_table.add_column("Metric", style="cyan", min_width=25); perf_table.add_column("Value", style="white", min_width=20)
        # ... (populate table as in old main.py) ...
        perf_table.add_row("  Attempted Count", str(bc_metrics['tx_sent_count']))
        perf_table.add_row("  Successful Count", str(bc_metrics['tx_success_count']))
        # ... Add all rows for blockchain perf table
        console.print(perf_table)
        # Visualize Blockchain Performance already called if visualize is True

    # Query Final Blockchain State (same as before)
    if actual_use_blockchain_flag and blockchain_interface_instance:
        console.print("\n[bold cyan]Querying Final Blockchain State...[/]")
        try:
            blockchain_interface_instance.print_contract_state(num_regions=args.regions, num_drugs=args.drugs)
        except Exception as e:
            console.print(f"[red]Error querying final blockchain state: {e}[/]")

    # --- Save Console Output ---
    html_filename = "simulation_report_simpy" + ("_blockchain" if actual_use_blockchain_flag else "") + ".html"
    save_console_html(console, output_folder=output_folder, filename=html_filename)
    console.print(f"\n[green]Visualizations and SimPy report saved to folder: '{output_folder}'[/]")