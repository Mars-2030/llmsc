# src/visualization/metrics_viz.py

"""
Metrics tracking and visualization for the pandemic supply chain.
Adapted to work with the SimPy-based PandemicSupplyChainSimulation object.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from collections import defaultdict
from typing import Dict, List, TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from src.simulation.simpy_env import PandemicSupplyChainSimulation # For type hinting
    from rich.console import Console # For type hinting

# Attempt to import Colors from config, with a fallback
try:
    from config import Colors
except ImportError:
    class Colors: # Minimal fallback for colors used in this file
        YELLOW = "yellow"; RED = "red"; GREEN = "green"; BLOCKCHAIN = "grey"


def track_service_levels(sim_object: 'PandemicSupplyChainSimulation') -> List[Dict[str, Any]]:
    """Track percentage of demand met over time using data from the simulation object."""
    service_levels = []
    if not hasattr(sim_object, 'demand_history') or not hasattr(sim_object, 'stockout_history'):
        if hasattr(sim_object, 'console') and sim_object.console:
            sim_object.console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Service Level: Missing demand or stockout history.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        return service_levels

    demands_by_day = defaultdict(list)
    for d_event in sim_object.demand_history:
        if isinstance(d_event, dict) and "day" in d_event:
            demands_by_day[d_event["day"]].append(d_event)

    stockouts_by_day = defaultdict(list)
    for s_event in sim_object.stockout_history:
        if isinstance(s_event, dict) and "day" in s_event:
            stockouts_by_day[s_event["day"]].append(s_event)

    # Consider all days present in either history, up to simulation duration
    all_days_in_data = set(demands_by_day.keys()) | set(stockouts_by_day.keys())
    if not all_days_in_data and sim_object.duration_days > 0 : # if no events but sim ran
        sim_days_range = range(sim_object.duration_days)
    elif not all_days_in_data:
        sim_days_range = []
    else:
        sim_days_range = range(max(all_days_in_data) + 1 if all_days_in_data else 0)


    for day in sim_days_range:
        day_demands_events = demands_by_day.get(day, [])
        day_stockouts_events = stockouts_by_day.get(day, [])

        total_demand_for_day = sum(d.get("demand", 0.0) for d in day_demands_events)
        total_unfulfilled_for_day = sum(s.get("unfulfilled", 0.0) for s in day_stockouts_events)

        if total_demand_for_day > 1e-6: # Avoid division by zero or near-zero
            level = 100.0 * (1.0 - total_unfulfilled_for_day / total_demand_for_day)
            service_level_val = max(0.0, min(100.0, level))
        else:
            service_level_val = 100.0 # 100% service if no demand

        service_levels.append({"day": day, "service_level": service_level_val})

    return service_levels

def visualize_service_levels(
    sim_object: 'PandemicSupplyChainSimulation',
    output_folder: str = "output",
    console: Optional['Console'] = None
):
    """Visualize service levels over time."""
    service_levels_data = track_service_levels(sim_object)
    console_to_use = console if console else getattr(sim_object, 'console', None)

    if not service_levels_data:
        if console_to_use: console_to_use.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]No service level data available to visualize.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        return

    output_path = os.path.join(output_folder, 'service_levels.png')
    days = [item["day"] for item in service_levels_data]
    service_values = [item["service_level"] for item in service_levels_data]

    plt.figure(figsize=(12, 6))
    plt.plot(days, service_values, marker='.', linestyle='-', color='blue', markersize=4)
    plt.axhline(y=95, color='green', linestyle='--', label='Target (95%)')
    plt.axhline(y=90, color='orange', linestyle='--', label='Acceptable (90%)')
    plt.axhline(y=80, color='red', linestyle='--', label='Critical Low (80%)')
    plt.xlabel('Day'); plt.ylabel('Service Level (%)')
    plt.title('Daily Service Level Over Time (% of Demand Met)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.ylim(0, 105)
    plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ Service levels visualization saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e:
         if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving service levels visualization: {e}[/bold {getattr(Colors, 'RED', 'red')}]")


def visualize_performance(
    sim_object: 'PandemicSupplyChainSimulation',
    output_folder: str = "output",
    console: Optional['Console'] = None
):
    """Visualize supply chain performance metrics using data from PandemicSupplyChainSimulation."""
    output_path = os.path.join(output_folder, 'supply_chain_performance.png')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    console_to_use = console if console else getattr(sim_object, 'console', None)

    # 1. Stockouts heatmap
    stockout_data_list = []
    if hasattr(sim_object, 'scenario') and hasattr(sim_object, 'stockouts'):
        for drug_id_int in range(sim_object.num_drugs):
            drug_id_str = str(drug_id_int)
            for region_id_int in range(sim_object.num_regions):
                region_id_str = str(region_id_int)
                stockout_count = sim_object.stockouts.get(drug_id_str, {}).get(region_id_str, 0)
                stockout_data_list.append({
                    "Drug": sim_object.scenario.drugs[drug_id_int].get("name", f"D-{drug_id_int}"),
                    "Region": sim_object.scenario.regions[region_id_int].get("name", f"R-{region_id_int}"),
                    "Stockouts": stockout_count
                })

    if stockout_data_list:
        stockout_df = pd.DataFrame(stockout_data_list)
        try:
             stockout_pivot = stockout_df.pivot(index="Drug", columns="Region", values="Stockouts")
             sns.heatmap(stockout_pivot, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0], linewidths=.5)
             axes[0].set_title("Stockout Days by Drug and Region")
        except Exception as e: # Catch more general errors for pivot
             axes[0].text(0.5, 0.5, 'Stockout data pivot failed.', ha='center', va='center')
             axes[0].set_title("Stockout Days by Drug and Region (Error)")
             if console_to_use: console_to_use.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Could not generate stockout heatmap: {e}[/{getattr(Colors, 'YELLOW', 'yellow')}]")
    else:
        axes[0].text(0.5, 0.5, 'No stockout data available.', ha='center', va='center')
        axes[0].set_title("Stockout Days by Drug and Region")

    # 2. Unfulfilled demand bar chart
    unfulfilled_data_list = []
    if hasattr(sim_object, 'scenario') and hasattr(sim_object, 'unfulfilled_demand_units') and hasattr(sim_object, 'total_demand_units'):
        for drug_id_int in range(sim_object.num_drugs):
            drug_id_str = str(drug_id_int)
            for region_id_int in range(sim_object.num_regions):
                region_id_str = str(region_id_int)
                unfulfilled_val = sim_object.unfulfilled_demand_units.get(drug_id_str, {}).get(region_id_str, 0.0)
                total_val = sim_object.total_demand_units.get(drug_id_str, {}).get(region_id_str, 0.0)
                percent_unfulfilled = (unfulfilled_val / total_val) * 100.0 if total_val > 1e-6 else 0.0
                unfulfilled_data_list.append({
                    "Drug": sim_object.scenario.drugs[drug_id_int].get("name", f"D-{drug_id_int}"),
                    "Region": sim_object.scenario.regions[region_id_int].get("name", f"R-{region_id_int}"),
                    "Percent Unfulfilled": percent_unfulfilled
                })

    if unfulfilled_data_list:
         unfulfilled_df = pd.DataFrame(unfulfilled_data_list)
         sns.barplot(data=unfulfilled_df, x="Drug", y="Percent Unfulfilled", hue="Region", ax=axes[1], errorbar=None)
         axes[1].set_title("Percentage of Unfulfilled Demand by Drug and Region")
         axes[1].set_ylabel("Percent Unfulfilled (%)"); axes[1].tick_params(axis='x', rotation=45)
         axes[1].legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left'); axes[1].grid(True, axis="y", alpha=0.3)
    else:
         axes[1].text(0.5, 0.5, 'No unfulfilled demand data.', ha='center', va='center')
         axes[1].set_title("Percentage of Unfulfilled Demand by Drug and Region")

    # 3. Patient impact bar chart
    impact_data_list = []
    if hasattr(sim_object, 'scenario') and hasattr(sim_object, 'patient_impact_score'):
        for r_int in range(sim_object.num_regions):
            r_str = str(r_int)
            impact_val = sim_object.patient_impact_score.get(r_str, 0.0)
            impact_data_list.append({
                "Region": sim_object.scenario.regions[r_int].get("name", f"R-{r_int}"),
                "Patient Impact": impact_val
            })
    if impact_data_list:
         impact_df = pd.DataFrame(impact_data_list)
         sns.barplot(x="Region", y="Patient Impact", data=impact_df, ax=axes[2], palette="viridis", hue="Region", legend=False) # hue="Region" with legend=False for color consistency by Region
         axes[2].set_title("Patient Impact Score by Region")
         axes[2].set_ylabel("Impact Score (higher is worse)"); axes[2].grid(True, axis="y", alpha=0.3)
    else:
         axes[2].text(0.5, 0.5, 'No patient impact data.', ha='center', va='center')
         axes[2].set_title("Patient Impact Score by Region")

    plt.tight_layout()
    try:
         plt.savefig(output_path); plt.close(fig)
         if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ Supply chain performance visualization saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e:
          if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving performance visualization: {e}[/bold {getattr(Colors, 'RED', 'red')}]")


def visualize_inventory_levels(
    sim_object: 'PandemicSupplyChainSimulation',
    output_folder: str = "output",
    console: Optional['Console'] = None
):
    """Visualize inventory levels including warehouse, using data from PandemicSupplyChainSimulation."""
    output_path = os.path.join(output_folder, 'inventory_levels.png')
    console_to_use = console if console else getattr(sim_object, 'console', None)

    if not hasattr(sim_object, 'inventory_history') or not sim_object.inventory_history:
        if console_to_use: console_to_use.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]No inventory history data available for visualization.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        return

    days = sorted(sim_object.inventory_history.keys())
    if not days: return

    fig, axes = plt.subplots(sim_object.num_drugs, 1, figsize=(14, 5 * sim_object.num_drugs), sharex=True, squeeze=False)
    plot_colors = {'warehouse': 'cyan', 'manufacturer': 'blue', 'distributor': 'green', 'hospital': 'magenta', 'total_system': 'black'}

    for drug_id_int in range(sim_object.num_drugs):
        ax = axes[drug_id_int, 0]
        drug_info = sim_object.scenario.drugs[drug_id_int]
        drug_name = drug_info.get("name", f"Drug-{drug_id_int}")
        crit_name = drug_info.get("criticality_name", "N/A")

        manu_inv_levels = [sim_object.inventory_history.get(d, {}).get((f"manu_usable_D{drug_id_int}"), 0.0) for d in days]
        wh_inv_levels = [sim_object.inventory_history.get(d, {}).get((f"warehouse_D{drug_id_int}"), 0.0) for d in days]
        
        dist_total_inv_levels = [0.0] * len(days)
        hosp_total_inv_levels = [0.0] * len(days)

        for r_id in range(sim_object.num_regions):
            dist_levels_region = [sim_object.inventory_history.get(d, {}).get((f"dist_R{r_id}_D{drug_id_int}"), 0.0) for d in days]
            hosp_levels_region = [sim_object.inventory_history.get(d, {}).get((f"hosp_R{r_id}_D{drug_id_int}"), 0.0) for d in days]
            dist_total_inv_levels = [x + y for x, y in zip(dist_total_inv_levels, dist_levels_region)]
            hosp_total_inv_levels = [x + y for x, y in zip(hosp_total_inv_levels, hosp_levels_region)]

        ax.plot(days, wh_inv_levels, label="Warehouse", color=plot_colors['warehouse'], linestyle="-", lw=2)
        ax.plot(days, manu_inv_levels, label="Manufacturer (Usable)", color=plot_colors['manufacturer'], linestyle="-", lw=2)
        ax.plot(days, dist_total_inv_levels, label="All Distributors", color=plot_colors['distributor'], linestyle="-", lw=2)
        ax.plot(days, hosp_total_inv_levels, label="All Hospitals", color=plot_colors['hospital'], linestyle="-", lw=2)
        
        total_system_inv = [w + m + d + h for w,m,d,h in zip(wh_inv_levels, manu_inv_levels, dist_total_inv_levels, hosp_total_inv_levels)]
        ax.plot(days, total_system_inv, label="Total System", color=plot_colors['total_system'], linestyle=":", lw=2.5)

        ax.set_title(f"Inventory: {drug_name} (Crit: {crit_name})")
        ax.set_ylabel("Units"); ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
        current_y_lim = ax.get_ylim()
        ax.set_ylim(bottom=0, top=max(10, current_y_lim[1])) # Ensure some y-range even if inv is 0

    if axes.size > 0 : axes[-1,0].set_xlabel("Day") # Set xlabel on the last subplot
    plt.suptitle("Supply Chain Inventory Levels Over Time", fontsize=16, y=1.00) # Adjusted y
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    try:
         plt.savefig(output_path); plt.close(fig)
         if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ Inventory levels visualization saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e:
         if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving inventory levels visualization: {e}[/bold {getattr(Colors, 'RED', 'red')}]")

    # Assuming visualize_warehouse_flow is still relevant and works with sim_object.production_history
    # visualize_warehouse_flow(sim_object, output_folder, console=console_to_use)


def visualize_blockchain_performance(
    blockchain_interface: Optional['BlockchainInterface'], # Now directly takes the interface
    output_folder="output",
    console: Optional['Console']=None
):
    """Visualize blockchain interaction performance metrics."""
    if not blockchain_interface or not hasattr(blockchain_interface, 'get_performance_metrics'):
        if console: console.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Blockchain interface not available or no performance metrics method. Skipping viz.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        return
    output_path = os.path.join(output_folder, 'blockchain_performance.png')

    bc_metrics = blockchain_interface.get_performance_metrics()
    tx_latencies = blockchain_interface.tx_latencies if hasattr(blockchain_interface, 'tx_latencies') else []
    read_latencies = blockchain_interface.read_latencies if hasattr(blockchain_interface, 'read_latencies') else []

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ... (Plotting logic for tx_latencies, read_latencies, tx_outcomes, read_outcomes) ...
    # This part of the function remains largely the same as your original,
    # as it directly uses the metrics from the blockchain_interface.
    # Ensure all keys accessed in bc_metrics are present.

    # Example for Transaction Latency Plot (Plot 1)
    ax = axes[0, 0]
    if tx_latencies:
        ax.hist(tx_latencies, bins=max(1,len(tx_latencies)//5 if len(tx_latencies)>10 else 10), color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(bc_metrics.get('tx_latency_avg_s',0), color='red', linestyle='dashed', linewidth=1, label=f"Avg: {bc_metrics.get('tx_latency_avg_s',0):.3f}s")
        if tx_latencies: ax.axvline(np.median(tx_latencies), color='orange', linestyle='dotted', linewidth=1, label=f"Median: {np.median(tx_latencies):.3f}s")
        ax.axvline(bc_metrics.get('tx_latency_p95_s',0), color='purple', linestyle='dashdot', linewidth=1, label=f"P95: {bc_metrics.get('tx_latency_p95_s',0):.3f}s")
        ax.set_title('Transaction Receipt Latency Distribution')
        ax.set_xlabel('Latency (seconds)'); ax.set_ylabel('Frequency')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Transaction Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Transaction Receipt Latency Distribution')

    # Read Call Latency (Plot 2)
    ax = axes[0, 1]
    if read_latencies:
        ax.hist(read_latencies, bins=max(1,len(read_latencies)//5 if len(read_latencies)>10 else 10), color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(bc_metrics.get('read_latency_avg_s',0), color='red', linestyle='dashed', linewidth=1, label=f"Avg: {bc_metrics.get('read_latency_avg_s',0):.3f}s")
        if read_latencies: ax.axvline(np.median(read_latencies), color='orange', linestyle='dotted', linewidth=1, label=f"Median: {np.median(read_latencies):.3f}s")
        ax.axvline(bc_metrics.get('read_latency_p95_s',0), color='purple', linestyle='dashdot', linewidth=1, label=f"P95: {bc_metrics.get('read_latency_p95_s',0):.3f}s")
        ax.set_title('Read Call Latency Distribution')
        ax.set_xlabel('Latency (seconds)'); ax.set_ylabel('Frequency')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Read Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Read Call Latency Distribution')

    # Transaction Outcomes (Plot 3)
    ax = axes[1,0]
    tx_total = bc_metrics.get('tx_sent_count',0)
    if tx_total > 0:
        counts = [bc_metrics.get('tx_success_count',0), bc_metrics.get('tx_failure_count',0)]
        labels = ['Succeeded', 'Failed']
        ax.barh(labels, counts, color=['mediumseagreen', 'salmon'], edgecolor='black')
        ax.set_title(f'Transaction Outcomes (Total: {tx_total})'); ax.set_xlabel('Count')
        rate = bc_metrics.get('tx_success_rate', "N/A")
        rate_str = f"{rate:.1f}% Success" if isinstance(rate, float) else str(rate)
        ax.text(0.95, 0.05, rate_str, ha='right', va='bottom', transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    else:
        ax.text(0.5,0.5, "No Transactions Sent", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Transaction Outcomes')

    # Read Call Outcomes (Plot 4)
    ax = axes[1,1]
    read_total = bc_metrics.get('read_call_count',0)
    if read_total > 0:
        errors = bc_metrics.get('read_error_count',0)
        counts = [read_total - errors, errors]
        labels = ['Succeeded', 'Failed']
        ax.barh(labels, counts, color=['cornflowerblue', 'lightgrey'], edgecolor='black')
        ax.set_title(f'Read Call Outcomes (Total: {read_total})'); ax.set_xlabel('Count')
        rate = bc_metrics.get('read_success_rate', "N/A")
        rate_str = f"{rate:.1f}% Success" if isinstance(rate, float) else str(rate)
        ax.text(0.95, 0.05, rate_str, ha='right', va='bottom', transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    else:
        ax.text(0.5,0.5, "No Read Calls", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Read Call Outcomes')


    plt.suptitle("Blockchain Interaction Performance Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(output_path); plt.close(fig)
        if console: console.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ Blockchain performance visualization saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e:
         if console: console.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving blockchain performance visualization: {e}[/bold {getattr(Colors, 'RED', 'red')}]")


def calculate_bullwhip_effect(environment_history: Dict[str, Any], num_regions: int) -> Optional[Dict[str, float]]: # num_regions IS a parameter
    """
    Calculates bullwhip effect. Takes the 'environment_history' dictionary and num_regions.
    """
    demand_hist = environment_history.get("demand_history", [])
    order_hist = environment_history.get("order_history", [])
    prod_hist = environment_history.get("production_history", [])
    sim_length = environment_history.get("scenario_length", 0)

    if not demand_hist or not order_hist or sim_length < 5:
        return None

    daily_demand = defaultdict(float)
    daily_hosp_orders = defaultdict(float)
    daily_dist_orders = defaultdict(float)
    daily_manu_prod = defaultdict(float)

    for event in demand_hist:
        if isinstance(event, dict): daily_demand[event.get('day',-1)] += event.get('demand',0.0)

    for event in order_hist:
        if isinstance(event, dict):
            if event.get('to_id') == 0: # Order TO Manufacturer (from Distributor)
                daily_dist_orders[event.get('day',-1)] += event.get('amount',0.0)
            # *** CORRECTED LINE: Use the 'num_regions' parameter ***
            elif event.get('from_id',-1) > num_regions and event.get('to_id',-1) <= num_regions and event.get('to_id',-1) != 0 : # Order TO Distributor (from Hospital)
                # from_id > num_regions means it's a hospital (e.g. num_regions+1 to 2*num_regions)
                # to_id <= num_regions AND !=0 means it's a distributor (1 to num_regions)
                daily_hosp_orders[event.get('day',-1)] += event.get('amount',0.0)

    for event in prod_hist:
        if isinstance(event, dict): daily_manu_prod[event.get('day',-1)] += event.get('amount_produced',0.0)

    days_range = range(sim_length)
    df = pd.DataFrame(index=days_range)
    df['HospDemand'] = [daily_demand.get(d, 0) for d in days_range]
    df['HospOrders'] = [daily_hosp_orders.get(d, 0) for d in days_range]
    df['DistOrders'] = [daily_dist_orders.get(d, 0) for d in days_range]
    df['ManuProd'] = [daily_manu_prod.get(d, 0) for d in days_range]

    epsilon = 1e-9
    # Ensure variance is calculated correctly and handle potential NaN or zero variance
    var_demand_val = df['HospDemand'].var(); var_demand = epsilon if pd.isna(var_demand_val) or var_demand_val <= epsilon else var_demand_val
    var_hosp_orders_val = df['HospOrders'].var(); var_hosp_orders = epsilon if pd.isna(var_hosp_orders_val) or var_hosp_orders_val <= epsilon else var_hosp_orders_val
    var_dist_orders_val = df['DistOrders'].var(); var_dist_orders = epsilon if pd.isna(var_dist_orders_val) or var_dist_orders_val <= epsilon else var_dist_orders_val
    var_manu_prod_val = df['ManuProd'].var(); var_manu_prod = epsilon if pd.isna(var_manu_prod_val) or var_manu_prod_val <= epsilon else var_manu_prod_val

    # Check again after epsilon application if any became None (unlikely but safety)
    if var_demand is None or var_hosp_orders is None or var_dist_orders is None or var_manu_prod is None:
        return None

    # Avoid division by zero if a variance is still effectively zero after epsilon
    bw_hosp_orders_demand = var_hosp_orders / var_demand if var_demand > epsilon else float('inf')
    bw_dist_orders_hosp = var_dist_orders / var_hosp_orders if var_hosp_orders > epsilon else float('inf')
    bw_manu_prod_dist = var_manu_prod / var_dist_orders if var_dist_orders > epsilon else float('inf')
    
    return {
        "Hospital Orders / Demand": bw_hosp_orders_demand,
        "Distributor Orders / Hosp Orders": bw_dist_orders_hosp,
        "Manufacturer Prod / Dist Orders": bw_manu_prod_dist,
    }

def visualize_costs(environment_history: Dict[str, Any], output_folder="output", console=None):
    """Visualize daily and cumulative costs over time from environment_history."""
    # This function's internal logic should remain the same, as it operates
    # on the 'cost_history' list provided in environment_history.
    # Just ensure 'cost_history' is correctly populated by the SimPy simulation.

    cost_hist_list = environment_history.get("cost_history", [])
    output_path = os.path.join(output_folder, 'costs_over_time.png')
    console_to_use = console

    if not cost_hist_list:
        if console_to_use: console_to_use.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]No cost history data available for visualization.[/{getattr(Colors, 'YELLOW', 'yellow')}]")
        return

    df = pd.DataFrame(cost_hist_list)
    if 'day' not in df.columns or 'holding_cost' not in df.columns or 'backlog_cost' not in df.columns:
        if console_to_use: console_to_use.print(f"[{getattr(Colors, 'YELLOW', 'yellow')}]Cost history missing required columns (day, holding_cost, backlog_cost).[/]")
        return

    df['cumulative_holding'] = df['holding_cost'].cumsum()
    df['cumulative_backlog'] = df['backlog_cost'].cumsum()
    df['cumulative_total'] = df['cumulative_holding'] + df['cumulative_backlog']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df['day'], df['holding_cost'], label='Daily Holding Cost', color='orange', alpha=0.8, marker='.', markersize=3)
    axes[0].plot(df['day'], df['backlog_cost'], label='Daily Backlog Cost', color='red', alpha=0.8, marker='.', markersize=3)
    axes[0].set_ylabel('Cost ($)'); axes[0].set_title('Daily Supply Chain Costs')
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_ylim(bottom=0)


    axes[1].plot(df['day'], df['cumulative_holding'], label='Cumulative Holding', color='darkorange', lw=2)
    axes[1].plot(df['day'], df['cumulative_backlog'], label='Cumulative Backlog', color='darkred', lw=2)
    axes[1].plot(df['day'], df['cumulative_total'], label='Cumulative Total Cost', color='black', linestyle='--', lw=2)
    axes[1].set_xlabel('Day'); axes[1].set_ylabel('Cumulative Cost ($)')
    axes[1].set_title('Cumulative Supply Chain Costs'); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1].set_ylim(bottom=0)


    plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close(fig)
        if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'GREEN', 'green')}]✓ Cost visualization saved to '{output_path}'[/bold {getattr(Colors, 'GREEN', 'green')}]")
    except Exception as e:
         if console_to_use: console_to_use.print(f"[bold {getattr(Colors, 'RED', 'red')}]Error saving cost visualization: {e}[/bold {getattr(Colors, 'RED', 'red')}]")