"""
Visualization functions for pandemic scenarios with SIR model.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.integrate import solve_ivp


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.scenario.generator import PandemicScenarioGenerator # For type hinting scenario
    from rich.console import Console # For type hinting console

# Attempt to import Colors from config, with a fallback
try:
    from config import Colors
except ImportError:
    class Colors: # Minimal fallback for colors used in this file
        YELLOW = "yellow"
        RED = "red"
        GREEN = "green" # If used by console.print success




def visualize_epidemic_curves(scenario, output_folder="output", console=None):
    """Visualize epidemic curves by region."""
    output_path = os.path.join(output_folder, 'epidemic_curves.png')
    plt.figure(figsize=(12, 8))
    for region_id, curve in scenario.epidemic_curves.items():
        # Check if curve is valid and has data
        if curve is not None and len(curve) > 0:
            region_name = scenario.regions[region_id]["name"]
            plt.plot(curve, label=region_name)
        else:
            if console: console.print(f"[yellow]Warning: Skipping visualization for empty epidemic curve for region {region_id}[/]")

    plt.xlabel('Day'); plt.ylabel('Active Cases')
    plt.title('SIR Model: Active Cases by Region'); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Epidemic curves visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving epidemic curves visualization: {e}[/]")


def visualize_drug_demand(scenario, output_folder="output", console = None):
    """Visualize drug demand by region."""
    output_path = os.path.join(output_folder, 'drug_demand.png')
    num_drugs = len(scenario.drugs); num_regions = len(scenario.regions)
    fig, axes = plt.subplots(num_drugs, 1, figsize=(12, 4 * num_drugs), sharex=True, squeeze=False) # Ensure axes is always 2D
    for drug_id, drug in enumerate(scenario.drugs):
        ax = axes[drug_id, 0] # Access subplot correctly
        for region_id in range(num_regions):
            region_name = scenario.regions[region_id]["name"]
            demand_curve = [scenario.get_daily_drug_demand(day, region_id, drug_id) for day in range(scenario.scenario_length)]
            ax.plot(demand_curve, label=region_name)
        ax.set_title(f'Daily Demand for {drug["name"]} (Crit: {drug["criticality_name"]})')
        ax.set_ylabel('Units Required'); ax.grid(True, alpha=0.3); ax.legend()
    plt.xlabel('Day'); plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Drug demand visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving drug demand visualization: {e}[/]")

def visualize_sir_components(scenario, output_folder="output", console=None):
    """Visualize SIR model components (S, I, R) for each region."""
    num_regions = len(scenario.regions)
    
    # Create a figure with subplots for each region
    fig, axes = plt.subplots(num_regions, 1, figsize=(12, 5 * num_regions), sharex=True)
    
    # If there's only one region, wrap the axis in a list for consistent indexing
    if num_regions == 1:
        axes = [axes]
    
    for region_id, region in enumerate(scenario.regions):
        # Get the active cases from the scenario's epidemic curve
        active_cases = scenario.epidemic_curves[region_id]
        
        # Plot active cases
        axes[region_id].plot(active_cases, 'r-', linewidth=2, label='Active Cases')
        
        # Add region information to the title
        population = region["population"]
        region_name = region["name"]
        region_type = region["type"]
        
        axes[region_id].set_title(f'Region {region_name} ({region_type}, Pop: {population:,})')
        axes[region_id].set_ylabel('Cases')
        axes[region_id].grid(True, alpha=0.3)
        axes[region_id].legend()
    
    # Set common x-label for the bottom subplot
    axes[-1].set_xlabel('Day')
    
    # Add an overall title
    plt.suptitle('SIR Model: Active Cases by Region', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_folder, 'sir_components.png')
    try:
        plt.savefig(output_path)
        plt.close()
        if console: console.print(f"[bold green]✓ SIR components visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving SIR components visualization: {e}[/]")



def visualize_sir_simulation(
    scenario: 'PandemicScenarioGenerator',
    selected_region_id: int = 0,
    output_folder: str = "output",
    console: Optional['Console'] = None
):
    """
    Visualize detailed SIR simulation (S, I, R components and new cases)
    for a selected region by re-simulating a basic SIR model based on
    the pre-generated active cases curve from the scenario.

    This re-simulation is an approximation for visualization purposes and uses
    estimated/fixed parameters (R0, infectious_period). It does not use
    the exact parameters that might have generated complex multi-wave scenarios.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Validate selected_region_id
    if not (0 <= selected_region_id < len(scenario.regions)):
        if console:
            console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Warning: Region ID {selected_region_id} is out of range (0-{len(scenario.regions)-1}). Visualizing Region 0 instead.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        selected_region_id = 0
        if not scenario.regions: # No regions defined at all
            if console: console.print(f"[{getattr(Colors, 'RED', 'red')}]Error: No regions defined in the scenario. Cannot visualize SIR simulation.[/{getattr(Colors, 'RED', 'red')}]")
            return

    region = scenario.regions[selected_region_id]
    region_name = region.get("name", f"Region {selected_region_id}")
    population = float(region.get("population", 100000)) # Default if missing

    # Get the pre-generated active cases curve for this region
    active_cases_scenario = scenario.epidemic_curves.get(selected_region_id)

    # Create figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'SIR Model Visualization for {region_name} (Population: {population:,.0f})', fontsize=16)

    # --- Plot 1: Active Cases from Scenario ---
    if active_cases_scenario is not None and len(active_cases_scenario) > 0:
        days_scenario = np.arange(len(active_cases_scenario))
        axes[0].plot(days_scenario, active_cases_scenario, 'r-', linewidth=2, label='Active Cases (from Scenario)')
    else:
        axes[0].text(0.5, 0.5, "No active cases data from scenario", ha='center', va='center', transform=axes[0].transAxes)
        if console: console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Warning: No active cases data in scenario for Region {selected_region_id} to plot.[/{getattr(Colors, 'YELLOW', 'yellow')}]")

    axes[0].set_ylabel('Number of Cases')
    axes[0].set_title('Active Cases Over Time (Scenario Data)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # --- SIR Re-simulation for S, I, R Components Visualization ---
    if active_cases_scenario is None or len(active_cases_scenario) == 0:
        if console: console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Skipping S,I,R re-simulation plot for Region {selected_region_id} due to missing scenario active cases data.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        axes[1].text(0.5, 0.5, "S,I,R Plot Skipped (No Scenario Data)", ha='center', va='center', transform=axes[1].transAxes)
        axes[2].text(0.5, 0.5, "New Cases Plot Skipped (No Scenario Data)", ha='center', va='center', transform=axes[2].transAxes)
    else:
        # Parameters for the visualization's SIR model (these are illustrative)
        r0_viz = 2.5  # Illustrative R0 for this visualization
        infectious_period_viz = 10.0  # Illustrative infectious period in days
        gamma_viz = 1.0 / infectious_period_viz
        beta_viz = r0_viz * gamma_viz

        # Initial conditions for the visualization's SIR model
        # Try to base initial infected on the first day of scenario's active cases
        initial_infected_viz = float(active_cases_scenario[0]) if active_cases_scenario[0] > 0 else max(1.0, population * 0.00001) # At least one, or small fraction
        initial_infected_viz = min(initial_infected_viz, population * 0.9) # Cap initial infected to avoid S < 0
        
        initial_susceptible_viz = population - initial_infected_viz
        initial_recovered_viz = 0.0
        if initial_susceptible_viz < 0: # Ensure S is not negative
            initial_susceptible_viz = 0.0
            initial_infected_viz = population # If S becomes negative, assume all are infected initially

        y0_viz = [initial_susceptible_viz, initial_infected_viz, initial_recovered_viz]

        # Time span and evaluation points for the visualization's SIR model
        t_eval_viz = days_scenario # Match the scenario's time points
        t_span_viz = [t_eval_viz[0], t_eval_viz[-1]] if len(t_eval_viz) > 0 else [0,0]


        def sir_model_ode_func(t, y, N_pop, beta, gamma):
            S, I, R = y
            # N_pop is total population, passed as an argument
            if N_pop == 0: N_pop = 1 # Avoid division by zero
            dSdt = -beta * S * I / N_pop
            dIdt = beta * S * I / N_pop - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]

        if t_span_viz[1] > t_span_viz[0] and population > 0: # Ensure valid span and population
            solution = solve_ivp(
                sir_model_ode_func,
                t_span_viz,
                y0_viz,
                args=(population, beta_viz, gamma_viz), # Pass N, beta, gamma
                t_eval=t_eval_viz,
                method='RK45',
                dense_output=True # Useful for smooth curves if t_eval is sparse
            )

            if console:
                console.print(f"[DEBUG VIZ_SIR] Re-sim for Region {selected_region_id} - solve_ivp completed.")
                console.print(f"  solution.success: {solution.success}, solution.status: {solution.status}")
                if hasattr(solution, 'y') and solution.y is not None:
                    console.print(f"  solution.y type: {type(solution.y)}, shape: {solution.y.shape if isinstance(solution.y, np.ndarray) else 'N/A (not ndarray)'}")

            if solution.success and hasattr(solution, 'y') and isinstance(solution.y, np.ndarray) and solution.y.shape[0] >= 3:
                S_viz = solution.y[0]
                I_viz = solution.y[1]
                R_viz = solution.y[2]

                # --- Plot 2: S, I, R Components (from re-simulation) ---
                axes[1].plot(t_eval_viz, S_viz / population * 100, 'b-', label='Susceptible (Viz.)')
                axes[1].plot(t_eval_viz, I_viz / population * 100, 'm-', label='Infected (Viz.)') # Changed color for distinction
                axes[1].plot(t_eval_viz, R_viz / population * 100, 'g-', label='Recovered (Viz.)')
                axes[1].set_ylabel('Percentage of Population')
                axes[1].set_title(f'SIR Model Components (Illustrative Re-simulation, R0≈{r0_viz:.1f})')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc='upper right')

                # --- Plot 3: New Cases per Day (from re-simulation) ---
                new_cases_viz = np.zeros_like(t_eval_viz, dtype=float)
                if len(t_eval_viz) > 0: new_cases_viz[0] = initial_infected_viz # Approx. for day 0
                for i in range(1, len(t_eval_viz)):
                    new_cases_viz[i] = max(0, S_viz[i-1] - S_viz[i]) # Change in Susceptible
                
                axes[2].bar(t_eval_viz, new_cases_viz, alpha=0.7, color='orange', label='New Cases (Viz.)')
                axes[2].set_xlabel('Day')
                axes[2].set_ylabel('Number of New Cases')
                axes[2].set_title('Estimated New Cases per Day (Illustrative Re-simulation)')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc='upper right')
            else:
                if console: console.print(f"[{getattr(Colors, 'RED', 'red')}]Error: solve_ivp failed or returned unexpected S,I,R data for Region {selected_region_id}. Msg: {solution.message}[/{getattr(Colors, 'RED', 'red')}]")
                axes[1].text(0.5, 0.5, "S,I,R Re-simulation Failed", ha='center', va='center', transform=axes[1].transAxes)
                axes[2].text(0.5, 0.5, "New Cases Re-simulation Failed", ha='center', va='center', transform=axes[2].transAxes)
        else:
            if console: console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Skipping S,I,R re-simulation for Region {selected_region_id} due to invalid time span or zero population.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
            axes[1].text(0.5, 0.5, "S,I,R Plot Skipped (Invalid Sim Params)", ha='center', va='center', transform=axes[1].transAxes)
            axes[2].text(0.5, 0.5, "New Cases Plot Skipped (Invalid Sim Params)", ha='center', va='center', transform=axes[2].transAxes)


    # Finalize and save plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    output_path = os.path.join(output_folder, f'sir_detailed_region_{selected_region_id}.png') # Changed filename slightly
    try:
        plt.savefig(output_path)
        if console: console.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ SIR detailed visualization for Region {selected_region_id} saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e_save:
        if console: console.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving SIR detailed visualization: {e_save}[/bold {getattr(Colors, 'RED', 'red')}]")
    finally:
        plt.close(fig) # Ensure figure is always closed


def visualize_disruptions(scenario, output_folder="output", console=None):
    """Visualize manufacturing and transportation disruptions."""
    output_path = os.path.join(output_folder, 'disruptions.png')
    manufacturing_disruptions = [d for d in scenario.disruptions if d["type"] == "manufacturing"]
    transportation_disruptions = [d for d in scenario.disruptions if d["type"] == "transportation"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    if manufacturing_disruptions:
        for i, disruption in enumerate(manufacturing_disruptions):
            drug_name = scenario.drugs[disruption["drug_id"]]["name"]
            start = disruption["start_day"]
            end = disruption["end_day"]
            severity = disruption["severity"]
            ax1.barh(i, end - start + 1, left=start, height=0.8, color=plt.cm.Reds(severity), alpha=0.7) # Added +1 to duration for visualization
            ax1.text(start + (end - start + 1) / 2, i, f"{severity:.2f}", ha='center', va='center', color='black')
        ax1.set_yticks(range(len(manufacturing_disruptions)))
        ax1.set_yticklabels([scenario.drugs[d["drug_id"]]["name"] for d in manufacturing_disruptions])
    ax1.set_title('Manufacturing Disruptions'); ax1.set_xlabel('Day')
    if transportation_disruptions:
        for i, disruption in enumerate(transportation_disruptions):
            region_name = scenario.regions[disruption["region_id"]]["name"]
            start = disruption["start_day"]
            end = disruption["end_day"]
            severity = disruption["severity"]
            ax2.barh(i, end - start + 1, left=start, height=0.8, color=plt.cm.Blues(severity), alpha=0.7) # Added +1 to duration for visualization
            ax2.text(start + (end - start + 1) / 2, i, f"{severity:.2f}", ha='center', va='center', color='black')
        ax2.set_yticks(range(len(transportation_disruptions)))
        ax2.set_yticklabels([scenario.regions[d["region_id"]]["name"] for d in transportation_disruptions])
    ax2.set_title('Transportation Disruptions'); ax2.set_xlabel('Day')
    plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Disruptions visualization saved to '{output_path}'[/]")
    except Exception as e:
         if console: console.print(f"[bold red]Error saving disruptions visualization: {e}[/]")