# Pandemic Supply Chain Simulation with Autonomous LangGraph Agents and Blockchain

## Overview

This project simulates a multi-echelon (Manufacturer -> Distributor -> Hospital) supply chain for essential drugs during a synthetic pandemic scenario. It now utilizes **autonomous agents powered by Large Language Models (LLMs) via Langchain and LangGraph**, combined with specialized tools, to drive decision-making at each supply chain level.

Agents analyze their situation, determine if they need additional information or calculations (like forecasts, blockchain data, or criticality assessments), dynamically call available tools using OpenAI's function-calling capabilities orchestrated by LangGraph, and then formulate their final decisions (production, allocation, ordering).

Blockchain integration (simulated locally using Hardhat) remains a key feature, enhancing trust and automation:

1.  **Trusted Data Source (Accessed via Tool):** Regional case counts (from the simulation's SIR model) are written to the blockchain daily. The **Manufacturer agent's LLM can now *choose* to call a specific tool (`get_blockchain_regional_cases`)** when its reasoning determines it needs this verified data, typically for prioritizing scarce allocations.
2.  **On-Chain Logic Execution (Triggered by Environment):** The core logic for **fair allocation** of drugs from the manufacturer to distributors (`executeFairAllocation`) resides **on the smart contract**. The simulation environment triggers this function based on the Manufacturer agent's *final allocation request*, ensuring that the allocation adheres to the verifiable rules and on-chain data (cases, criticality).

This version focuses on increased agent autonomy, leveraging the LLM's reasoning to orchestrate tool use, while retaining the blockchain for critical data verification and process enforcement.

## Features

*   **Multi-Echelon Simulation:** Models Manufacturer, regional Distributors, and regional Hospitals.
*   **Pandemic Dynamics:** Uses an SIR (Susceptible-Infected-Recovered) model to generate regional epidemic curves influencing drug demand.
*   **Autonomous LLM Agents (LangGraph):**
    *   Agents use OpenAI models (e.g., GPT-4o) integrated via Langchain.
    *   **LangGraph State Machines:** Agent decision logic is defined as a graph where the LLM decides the flow, including when to call tools.
    *   **Dynamic Tool Use:** LLM analyzes observations and decides *if* and *which* tools (forecasting, blockchain query, criticality assessment, optimal order calculation) are needed.
    *   **Reduced Hardcoded Rules:** Python rules are minimized, primarily acting as *hard constraints* (e.g., inventory/capacity limits) applied *after* the LLM/graph produces a decision.
*   **Blockchain Integration (Hardhat/Solidity):**
    *   **Tool-Based Case Data Query:** Manufacturer LLM can invoke `get_blockchain_regional_cases_tool` to query verified cases.
    *   **On-Chain Fair Allocation Enforcement:** Environment triggers `executeFairAllocation` based on the Manufacturer agent's final request; the contract executes allocation using on-chain data.
    *   Daily case counts written to the blockchain by the environment.
    *   Initial drug criticalities set on-chain during deployment.
*   **Scenario Generation:** Configurable parameters for regions, drugs, pandemic severity, and disruptions.
*   **Metrics & Visualization:** Tracks KPIs (stockouts, service level, patient impact, costs, bullwhip effect) and generates plots using Matplotlib/Seaborn. Blockchain interaction metrics are also tracked.
*   **Configuration:** Uses a `.env` file for API keys and blockchain settings.
*   **Console Output:** Rich console output for detailed logging (including LangGraph steps) and reporting (savable as HTML).

## Architecture & Blockchain Interaction (LangGraph Model)

1.  **Environment (`PandemicSupplyChainEnvironment`):** Simulates the supply chain, pandemic spread (SIR model), updates blockchain cases daily, processes agent actions, and **triggers the `executeFairAllocation` blockchain transaction** based on the Manufacturer's final allocation decision.
2.  **Agents (Python - e.g., `ManufacturerAgentLG`):**
    *   Contain a compiled **LangGraph** application (`graph_app`).
    *   The `decide` method sets up the initial `AgentState` (with observation, messages) and invokes the graph.
    *   **LangGraph Nodes:**
        *   `llm`: Calls the OpenAI model via `OpenAILLMIntegration`.
        *   `tools`: Executes Python functions from `PandemicSupplyChainTools` based on LLM requests (using `execute_tool` mapping).
    *   **LangGraph Edges:** Control the flow between LLM calls and tool executions based on whether the LLM requested a tool.
    *   The graph returns the final LLM response (hopefully the decision JSON).
    *   The agent applies final *hard constraints* (e.g., inventory/capacity caps) to the LLM's output before returning the structured action to the environment.
3.  **Blockchain (Hardhat Node + `SupplyChainData.sol`):**
    *   **Initial Setup:** `deploy.js` sets drug criticalities.
    *   **Daily Case Update:** Environment calls `updateRegionalCaseCount`.
    *   **Manufacturer Case Query (Tool):** Manufacturer LLM requests `get_blockchain_regional_cases` -> LangGraph `tools` node executes the Python tool wrapper -> `BlockchainInterface` calls `getRegionalCaseCount` on the contract.
    *   **Fair Allocation Execution:** Environment receives Manufacturer's final allocation JSON -> calls `executeFairAllocation` via `BlockchainInterface` -> Contract executes logic using on-chain cases/criticality.
4.  **Tools (`PandemicSupplyChainTools`):**
    *   Provides Python functions for forecasting, assessment, order quantities, etc.
    *   Includes `get_openai_tool_definitions` to describe tools for the LLM.
    *   Includes the static wrapper `get_blockchain_regional_cases_tool` for the blockchain query.
5.  **LLM Integration (`OpenAILLMIntegration`):** Uses `langchain_openai` to handle single API calls, including binding tools for the LLM. Does *not* manage the tool-calling loop (LangGraph does).

## Prerequisites

*   **Python:** 3.9 or higher recommended (due to Langchain/LangGraph dependencies).
*   **Node.js & npm:** LTS version recommended. [Download Node.js](https://nodejs.org/)
*   **Git:** For cloning the repository.
*   **OpenAI API Key:** Requires access to models supporting function/tool calling (e.g., GPT-4, GPT-4o recommended). Obtain from [OpenAI](https://platform.openai.com/signup/).

## Setup Instructions (Local Hardhat)

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Set up Python Environment:**
    ```bash
    python -m venv venv
    # Activate (macOS/Linux)
    source venv/bin/activate
    # OR Activate (Windows CMD/PowerShell)
    # venv\Scripts\activate

    # Install dependencies (ensure requirements.txt includes langchain, langgraph)
    pip install -r requirements.txt --upgrade
    ```

3.  **Install Node.js Dependencies:**
    ```bash
    npm install
    ```

4.  **Configure Environment Variables (`.env`):**
    *   Create/edit `.env` in the project root.
    *   Add/update the following:

        ```dotenv
        # Required: Your OpenAI API Key (GPT-4o or similar recommended)
        OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # Required: Private key for LOCAL Hardhat node interaction.
        # Get this from the output of `npx hardhat node`. Use Account #0 for deployment.
        PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80" # Example

        # Recommended: Local node URL (Hardhat default)
        NODE_URL="http://127.0.0.1:8545"

        # Auto-filled by deploy script - leave blank or commented initially
        # CONTRACT_ADDRESS=""
        ```
    *   **IMPORTANT:** Use a private key from `npx hardhat node`. **Do not use real keys.**

5.  **Compile Smart Contract:**
    ```bash
    npx hardhat compile
    ```

6.  **(Crucial) Adjust Drug Criticality Setup in `deploy.js`:**
    *   The `executeFairAllocation` contract function **requires** drug criticalities to be set.
    *   The default `scripts/deploy.js` sets criticalities for Drugs 0, 1, and 2.
    *   **If you run the Python simulation with `--drugs N` where N > 3, you MUST modify `scripts/deploy.js`** to set criticalities for all drugs up to ID `N-1` *before* deploying.
    *   **To modify `deploy.js`:** Open it, find `// Example: Set criticalities...`, and adjust the `drugCriticalities` array or the loop logic to cover all drugs you intend to simulate with appropriate values (1-Low to 4-Critical).
    *   **Failure:** If criticality isn't set for a drug involved in an allocation, the `executeFairAllocation` transaction will likely **revert**.

## Running the Simulation (Local Hardhat)

1.  **Start Local Blockchain Node (Terminal 1):**
    *   Open a terminal, `cd` to the project directory.
    *   Run: `npx hardhat node`
    *   Leave this terminal running. Copy the `Private Key` for `Account #0` into your `.env` file if you haven't.

2.  **Deploy Smart Contract (Terminal 2):**
    *   Open a *new* terminal, `cd` to the project directory.
    *   Run (Node.js accessible, no Python venv): `npm run deploy:localhost`
    *   Verify success, criticality settings logged, and `CONTRACT_ADDRESS` updated in `.env`.

3.  **Run Python Simulation (Terminal 2 or 3):**
    *   **Activate Python virtual environment:** `source venv/bin/activate` (or `venv\Scripts\activate`).
    *   Run `main.py`, ensuring `--drugs` matches `deploy.js` configuration. Use `--use-blockchain`.
        ```bash
        # Example: 3 regions, 3 drugs (matching default deploy.js), 10 days, verbose
        # Use a capable model like gpt-4o
        python main.py --regions 3 --drugs 3 --days 10 --use-blockchain --verbose --model gpt-4o

        # Example: 5 regions, 5 drugs (requires deploy.js modification!), 20 days
        # python main.py --regions 5 --drugs 5 --days 20 --use-blockchain --verbose --model gpt-4o
        ```
    *   `--use-blockchain` enables interaction with Hardhat node/contract.
    *   `--verbose` shows detailed agent/graph/tool logs.

## Checking Results

1.  **Python Console Output (`main.py` run):**
    *   Look for successful blockchain initialization and agent creation.
    *   Observe `[GRAPH]` logs showing the flow (LLM -> Tools -> LLM -> END).
    *   Note which tools are called (`[TOOL]`) and their outputs.
    *   See the final parsed (`[LG Parsed JSON]`) and validated (`[Final Validated Decision]`) outputs for each agent.
    *   Check blockchain transaction confirmations (`✓ Tx Success!`) or failures (`❌ Tx Failed`).
    *   Review the final results summary (stockouts, impact, service level, costs, bullwhip).
    *   Check the final "Querying Final Blockchain State" section.
2.  **Hardhat Node Console Output (Terminal 1):**
    *   Observe `eth_` JSON-RPC calls and transaction mining/reverts. Look especially for `SupplyChainData.executeFairAllocation` calls and potential reverts if criticalities are missing.
3.  **Output Folder (`output_lg_<timestamp>_.../`):**
    *   **`simulation_report_langgraph*.html`:** Open in a browser for the complete, colorized simulation log.
    *   **`.png` files:** Analyze plots for supply chain performance, inventory, epidemic curves, costs, blockchain metrics, etc.

## Project Structure (Illustrative - Key Changes)

```
.
├── .env
├── README.md # <--- Updated Documentation
├── contracts/
│ └── SupplyChainData.sol # (Unchanged)
├── hardhat.config.js # (Unchanged)
├── main.py # <--- Uses new Agent Factories
├── node_modules/
├── output_lg*/ # <--- Default output folder prefix changed
├── package.json
├── requirements.txt # <--- Updated (langchain, langgraph, etc.)
├── scripts/  s
│ └── deploy.js # (Unchanged logic, but content may need editing for --drugs > 3)
├── src/
│ ├── agents/
│ │ ├── init.py # <--- Exports  factories
│ │ ├── base.py # <--- NEW (LangGraph state, nodes, graph factory)
│ │ ├── distributor.py # <--- NEW/REFACTORED
│ │ ├── hospital.py # <--- NEW/REFACTORED
│ │ └── manufacturer.py# <--- NEW/REFACTORED
│ ├── blockchain/ # (Unchanged)
│ │ ├── interface.py
│ │ └── SupplyChainData.abi.json
│ ├── environment/ # (Unchanged)
│ │ ├── metrics.py
│ │ └── supply_chain.py
│ ├── llm/
│ │ └── openai_integration.py # <--- REFACTORED (uses langchain, simpler invoke)
│ ├── scenario/ # (Unchanged)
│ │ ├── generator.py
│ │ └── visualizer.py
│ └── tools/
│ ├── init.py # <--- ADDED get_openai_tool_definitions()
│ ├── allocation.py # (Unchanged)
│ ├── assessment.py # (Unchanged)
│ └── forecasting.py # (Unchanged)
├── test/
└── venv/
```


## Future Improvements / Considerations

*   **Prompt Optimization:** Iteratively refine agent prompts for better reasoning and tool usage.
*   **Error Handling in Graph:** Implement more robust error handling within the LangGraph flow (e.g., what if a tool consistently fails?).
*   **More Sophisticated Tools:** Develop more advanced tools (e.g., complex inventory optimization models).
*   **Agent Memory:** Explore more sophisticated memory mechanisms beyond simple message history if needed for long-term planning.
*   **Evaluation:** Systematically evaluate performance against the previous rule-based/LLM hybrid and a pure rule-based baseline using the defined metrics.
*   **Cost/Latency:** Monitor OpenAI API costs and decision latency, optimizing if necessary.
*   **Testnet/Mainnet:** Adapt configuration for deployment to public networks.