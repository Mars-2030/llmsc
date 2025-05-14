# --- START OF (Modified) src/llm/openai_integration.py ---

import json
import time
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Assuming console is imported from config if needed
try:
    from config import console, Colors
except ImportError:
    console = None # Basic fallback

class OpenAILLMIntegration:
    """
    Integration class using Langchain's ChatOpenAI for the simulation.
    Focuses on single LLM invocations for LangGraph.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o", # Defaulting to a more capable model for tool use
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay: float = 5.0, # Consider shorter delay for simulation speed
        console=None
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        # max_retries is handled by ChatOpenAI client
        self.console = console

        try:
            self.client = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                max_retries=max_retries,
                # request_timeout=30 # Optional: set timeout
            )
            self._print(f"[green]Langchain ChatOpenAI client initialized successfully for model: {self.model_name}[/]")
        except Exception as e:
            self._print(f"[bold red]Error initializing ChatOpenAI client: {e}[/]")
            raise

    def _print(self, message):
        """Helper to safely print using the stored console."""
        if self.console:
            self.console.print(message)

    def invoke_llm_once(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None) -> AIMessage:
        """
        Makes a single call to the OpenAI API using ChatOpenAI client.
        Handles binding tools if provided. Does NOT handle the tool execution loop.

        Args:
            messages: List of Langchain message objects (HumanMessage, AIMessage, ToolMessage).
            tools: Optional list of OpenAI tool JSON schemas.

        Returns:
            AIMessage: The response message from the LLM, which might contain tool calls or an error message.
        """
        bound_client = self.client
        if tools:
            try:
                # Use bind_tools for proper tool integration with langchain client
                bound_client = self.client.bind_tools(tools)
            except Exception as e:
                 self._print(f"[red]Error binding tools to LLM client: {e}. Proceeding without tools bound explicitly (may affect tool calling).[/]")
                 # Continue with unbound client, might still work depending on model/version

        try:
            # Invoke the possibly tool-bound client
            response = bound_client.invoke(messages)

            if not isinstance(response, AIMessage):
                 self._print(f"[red]LLM invocation did not return an AIMessage. Got: {type(response)}[/]")
                 return AIMessage(content="Error: LLM did not provide a valid AIMessage response.")
            return response
        except Exception as e:
            self._print(f"[red]OpenAI API call failed during invoke: {e}[/]")
            # Return an AIMessage indicating failure
            return AIMessage(content=f"Error: API call failed - {e}")

# --- END OF (Modified) src/llm/openai_integration.py ---