"""
Configuration settings and utilities for the pandemic supply chain simulation.
"""

import os
import numpy as np
from rich.console import Console
from rich.terminal_theme import MONOKAI # For HTML export theme
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# --- Rich Console ---
# record=True allows HTML export of console output
# width can be adjusted based on your terminal/preference
console = Console(record=True, width=120)

# --- Random Seed for Reproducibility ---
# Ensures that simulations with the same parameters produce the same random numbers
# (e.g., for demand fluctuations, disruption occurrences, if not overridden by scenario)
np.random.seed(42)
# random.seed(42) # If using Python's built-in random module elsewhere

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Blockchain Configuration ---
NODE_URL = os.getenv("NODE_URL", "http://127.0.0.1:8545") # Default to local Hardhat node
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS") # Populated by deploy script
# Default location where deploy script saves ABI (relative to this config.py file's directory)
# This assumes config.py is in the root of the project.
# Adjust if config.py is moved.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRACT_ABI_PATH = os.path.abspath(os.path.join(_BASE_DIR, 'src', 'blockchain', 'SupplyChainData.abi.json'))
# Account private key for sending transactions (e.g., owner role for contract interactions)
BLOCKCHAIN_PRIVATE_KEY = os.getenv("PRIVATE_KEY") # From Hardhat node output

# --- SimPy Configuration (Defaults) ---
# These can be overridden by command-line arguments in main.py
# SimPy time units are typically floats, representing days in this simulation.
SIMPY_DEFAULT_PRODUCTION_TIME = 1.0  # Default time (days) for a single unit/batch of production (can be refined)
SIMPY_DEFAULT_TRANSPORT_TIME_MANU_TO_DIST = 3.0  # Avg days for manufacturer to distributor shipment
SIMPY_DEFAULT_TRANSPORT_TIME_DIST_TO_HOSP = 1.0  # Avg days for distributor to hospital shipment
SIMPY_DEFAULT_WAREHOUSE_RELEASE_DELAY = 1.0  # Default delay (days) for warehouse releases to manufacturer usable stock
SIMPY_DEFAULT_ALLOCATION_BATCH_FREQUENCY = 1  # Default frequency (days) for manufacturer allocation batching (1 = daily)

# --- LLM Configuration (Defaults) ---
LLM_DEFAULT_MODEL = "gpt-4o" # Default model for agents
LLM_DEFAULT_TEMPERATURE = 0.2 # Controls randomness of LLM responses

# --- Cost Configuration (Defaults) ---
COST_HOLDING_PER_UNIT_DAY = 0.005
COST_BACKLOG_PER_UNIT = 5.0
COST_BACKLOG_CRITICALITY_MULTIPLIER = 2.0

# --- API Key and Blockchain Configuration Checks ---
def check_openai_api_key():
    """Checks if the OpenAI API key is set and prints an error if not."""
    if not OPENAI_API_KEY or OPENAI_API_KEY in ["YOUR_OPENAI_API_KEY", "Insert the API key", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]:
        console.print("[bold red]Error: 'OPENAI_API_KEY' is not set or is using a placeholder in the .env file.[/]")
        console.print("[bold red]Please obtain an API key from OpenAI and set it in your .env file.[/]")
        return False
    return True

def check_blockchain_config():
    """
    Checks essential blockchain configuration variables.
    Returns True if basic config is present, False otherwise.
    Prints warnings for missing optional components.
    """
    config_ok = True
    if not CONTRACT_ADDRESS:
        console.print("[bold yellow]Warning: CONTRACT_ADDRESS not found in .env. Blockchain features requiring a contract address will fail.[/]")
        # Not necessarily a fatal error for all modes, but important if use_blockchain is True.
        # Consider making this a hard fail if use_blockchain is True in main.py.
        config_ok = False # Let's make it a failure point for check_blockchain_config

    if not os.path.exists(CONTRACT_ABI_PATH):
         console.print(f"[bold red]Error: Contract ABI not found at expected path: {CONTRACT_ABI_PATH}.[/]")
         console.print(f"[bold red]Ensure the contract is compiled and the ABI is in the correct location.[/]")
         config_ok = False

    if not BLOCKCHAIN_PRIVATE_KEY:
         console.print("[bold yellow]Warning: BLOCKCHAIN_PRIVATE_KEY not found in .env. Blockchain write operations (e.g., updating cases, executing allocation) will fail.[/]")
         # Allow read-only operation? For now, just warn.
         # If use_blockchain implies writes, this could also be a hard fail point in main.py.

    if not NODE_URL:
        console.print("[bold yellow]Warning: NODE_URL not found in .env. Using default local node, but this might not be intended.[/]")
        # Not fatal, as it defaults.

    return config_ok


# --- Output Styling (Rich Colors) ---
class Colors:
    """Terminal color themes for the simulation for Rich console."""
    # Agent Types
    MANUFACTURER = "blue"
    DISTRIBUTOR = "green"
    HOSPITAL = "magenta"

    # Simulation Components
    SIMPY = "bright_green" # For SimPy specific messages
    BLOCKCHAIN = "bright_black" # For blockchain interaction messages
    LLM = "cyan" # For LLM interaction messages (can be more specific)

    # Message Categories
    TOOL_CALL = "yellow"
    TOOL_OUTPUT = "dark_khaki" # Differentiated from tool call
    DECISION = "cyan"
    REASONING = "italic yellow" # LLM reasoning steps
    FALLBACK = "bold orange1"
    ERROR = "bold red"
    WARNING = "bold yellow"
    SUCCESS = "bold green"
    INFO = "dim" # For less prominent informational messages
    DEBUG = "grey50" # For detailed debug messages

    # States & Metrics
    STOCKOUT = "red"
    IMPACT = "bold red"
    DAY_HEADER = "black on cyan bold"
    EPIDEMIC_STATE = "blue"

    # General Styles
    BOLD = "bold"
    ITALIC = "italic"
    DIM = "dim" # Already defined, but good to list
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BLACK = "black"

    @staticmethod
    def styled_text(text, style):
        """Helper to wrap text with Rich styling tags."""
        return f"[{style}]{text}[/]"

# --- Utility Functions ---
def save_console_html(console_obj: Console, filename="simulation_report.html", output_folder="output"):
    """Saves the recorded Rich console output to an HTML file."""
    output_path = os.path.join(output_folder, filename)
    try:
        # Ensure console has recorded content before exporting
        if not console_obj.record:
             console_obj.print(f"[{Colors.WARNING}]Console recording is disabled. Cannot export HTML.[/]")
             return

        console_obj.print(f"[{Colors.INFO}]Attempting to export Rich console log to HTML: {output_path}...[/]")
        html_content = console_obj.export_html(theme=MONOKAI, inline_styles=True) # MONOKAI is a good dark theme

        if not html_content or len(html_content) < 500: # Arbitrary small length check
             console_obj.print(f"[{Colors.WARNING}]Exported HTML content seems empty or too short. Length: {len(html_content)}[/]")

        # Customize title (optional)
        html_content = html_content.replace("<title>Rich", "<title>Pandemic Supply Chain Simulation Report (SimPy + LangGraph + Blockchain)</title>")

        with open(output_path, "w", encoding="utf-8") as file:
            bytes_written = file.write(html_content)
        console_obj.print(f"[{Colors.SUCCESS}]✓ Console output saved to HTML: '{output_path}' ({bytes_written} bytes written)[/]")

    except Exception as e:
        console_obj.print(f"[{Colors.ERROR}]Error saving console output to HTML: {e}[/]")
        # console_obj.print_exception(show_locals=False) # For more detailed debugging if needed

def ensure_folder_exists(console_obj: Console, folder_path: str):
    """Creates a folder if it doesn't exist. Optionally uses a console for logging."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if console_obj:
            console_obj.print(f"[{Colors.SUCCESS}]Created output folder: {folder_path}[/]")
    return folder_path


# --- Initial Checks on Load (Optional but good practice) ---
# check_openai_api_key() # You might call this in main.py instead to control exit behavior.
# If use_blockchain is enabled in main.py, then call check_blockchain_config() there.