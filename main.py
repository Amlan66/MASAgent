"""
Agent - Main Entry Point

Entry point for the Agent system.
Initializes all agents and orchestrates query processing.
"""
import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from mcp_servers.multiMCP import MultiMCP
from agents import (
    RetrieverAgent,
    PerceptionAgent,
    CriticAgent,
    DecisionAgent,
    ExecutorAgent,
    MemoryAgent
)
from orchestrator import AgentCoordinator

# Load environment variables
load_dotenv()

BANNER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔱  Amlan Agent  🔱
A reasoning-driven AI agent with memory and tool capabilities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type your question and press Enter.
Type 'exit' or 'quit' to leave.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def load_agent_config() -> Dict[str, Any]:
    """
    Load agent configuration from profiles.yaml.
    
    Returns:
        Dict with agent configuration parameters
    """
    config_path = Path("config/profiles.yaml")
    
    if not config_path.exists():
        print(f"⚠️  Warning: {config_path} not found. Using defaults.")
        return get_default_config()
    
    try:
        with open(config_path, "r") as f:
            profile = yaml.safe_load(f)
        
        config = {
            # Strategy parameters
            "planning_strategy": profile.get("strategy", {}).get("planning_mode", "exploratory"),
            "exploration_mode": profile.get("strategy", {}).get("exploration_mode", "parallel"),
            "max_steps": profile.get("strategy", {}).get("max_steps", 10),
            "max_retries_per_step": profile.get("strategy", {}).get("max_lifelines_per_step", 2),
            "memory_fallback_enabled": profile.get("strategy", {}).get("memory_fallback_enabled", True),
            
            # Memory parameters
            "memory_base_dir": profile.get("memory", {}).get("storage", {}).get("base_dir", "storage/session_logs"),
            "summarize_results": profile.get("memory", {}).get("summarize_tool_results", True),
            
            # LLM parameters
            "llm_provider": profile.get("llm", {}).get("text_generation", "gemini"),
            "embedding_model": profile.get("llm", {}).get("embedding", "nomic"),
            
            # Agent metadata
            "agent_name": profile.get("agent", {}).get("name", "Sudarshan Chakra"),
            "agent_id": profile.get("agent", {}).get("id", "chakra_001"),
        }
        
        print(f"✓ Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        print(f"⚠️  Error loading config: {e}. Using defaults.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration if profiles.yaml is missing or invalid.
    
    Returns:
        Dict with default configuration
    """
    return {
        "planning_strategy": "exploratory",
        "exploration_mode": "parallel",
        "max_steps": 10,
        "max_retries_per_step": 2,
        "memory_fallback_enabled": True,
        "memory_base_dir": "storage/session_logs",
        "summarize_results": True,
        "llm_provider": "gemini",
        "embedding_model": "nomic",
        "agent_name": "Sudarshan Chakra",
        "agent_id": "chakra_001"
    }

def load_mcp_config() -> list:
    """
    Load MCP server configuration from mcp_server_config.yaml.
    
    Returns:
        List of MCP server configurations
    """
    config_path = Path("config/mcp_server_config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    mcp_servers = config.get("mcp_servers", [])
    print(f"✓ Loaded {len(mcp_servers)} MCP server configurations")
    
    return mcp_servers

async def initialize_mcp(mcp_configs: list) -> MultiMCP:
    """
    Initialize MultiMCP with server configurations.
    
    Args:
        mcp_configs: List of MCP server configurations
    
    Returns:
        Initialized MultiMCP instance
    """
    print("\n🔧 Initializing MCP servers...")
    multi_mcp = MultiMCP(server_configs=mcp_configs)
    await multi_mcp.initialize()
    print(f"✓ {len(mcp_configs)} MCP servers initialized\n")
    
    return multi_mcp

def get_prompt_path(prompt_name: str) -> str:
    """
    Get path to prompt file.
    
    Args:
        prompt_name: Name of prompt file (without .txt extension)
    
    Returns:
        Full path to prompt file
    """
    # Try new location first
    new_path = Path(f"config/prompts/{prompt_name}.txt")
    if new_path.exists():
        return str(new_path)
    
    # Fallback to old location
    old_path = Path(f"prompts/{prompt_name}.txt")
    if old_path.exists():
        return str(old_path)
    
    raise FileNotFoundError(f"Prompt file not found: {prompt_name}.txt")

async def initialize_agents(multi_mcp: MultiMCP, config: Dict[str, Any]) -> AgentCoordinator:
    """
    Initialize all agents and create coordinator.
    
    Args:
        multi_mcp: Initialized MultiMCP instance
        config: Agent configuration dict
    
    Returns:
        Initialized AgentCoordinator
    """
    print("🤖 Initializing agents...\n")
    
    # Initialize each agent
    retriever = RetrieverAgent(config={
        "agent_name": "RetrieverAgent",
        "memory_base_dir": config["memory_base_dir"]
    })
    await retriever.initialize()
    print("  ✓ RetrieverAgent initialized")
    
    perception = PerceptionAgent(config={
        "agent_name": "PerceptionAgent",
        "prompt_path": get_prompt_path("perception_prompt"),
        "llm_provider": config["llm_provider"]
    })
    await perception.initialize()
    print("  ✓ PerceptionAgent initialized")
    
    critic = CriticAgent(config={
        "agent_name": "CriticAgent",
        "prompt_path": get_prompt_path("critic_prompt"),
        "llm_provider": config["llm_provider"]
    })
    await critic.initialize()
    print("  ✓ CriticAgent initialized")
    
    decision = DecisionAgent(
        multi_mcp=multi_mcp,
        config={
            "agent_name": "DecisionAgent",
            "prompt_path": get_prompt_path("decision_prompt"),
            "llm_provider": config["llm_provider"]
        }
    )
    await decision.initialize()
    print("  ✓ DecisionAgent initialized")
    
    executor = ExecutorAgent(
        multi_mcp=multi_mcp,
        config={
            "agent_name": "ExecutorAgent",
            "max_functions": 5,
            "timeout": 500
        }
    )
    await executor.initialize()
    print("  ✓ ExecutorAgent initialized")
    
    memory = MemoryAgent(config={
        "agent_name": "MemoryAgent",
        "storage_base_dir": config["memory_base_dir"]
    })
    await memory.initialize()
    print("  ✓ MemoryAgent initialized")
    
    # Create coordinator
    coordinator = AgentCoordinator(
        retriever=retriever,
        perception=perception,
        critic=critic,
        decision=decision,
        executor=executor,
        memory=memory,
        config=config
    )
    
    print(f"\n✓ All agents initialized successfully\n")
    
    return coordinator

async def process_query(coordinator: AgentCoordinator, query: str):
    """
    Process a single user query.
    
    Args:
        coordinator: Initialized AgentCoordinator
        query: User query string
    """
    try:
        session = await coordinator.handle_query(query)
        
        # Display result
        print("\n" + "="*60)
        print("📊 QUERY RESULT")
        print("="*60)
        
        if session.status == "completed" and session.solution_summary:
            print(f"✅ Status: SUCCESS")
            print(f"📝 Answer: {session.solution_summary}")
            if session.confidence:
                print(f"🎯 Confidence: {session.confidence:.2%}")
        else:
            print(f"⚠️  Status: {session.status.upper()}")
            if session.final_answer:
                print(f"📝 Partial Answer: {session.final_answer}")
        
        print(f"🔢 Steps executed: {len(session.executed_steps)}")
        print(f"⏱️  Session ID: {session.session_id}")
        print("="*60 + "\n")
        
        return session
        
    except Exception as e:
        print(f"\n❌ Error processing query: {e}\n")
        return None


async def interactive_mode(coordinator: AgentCoordinator):
    """
    Run agent in interactive mode with continuous query loop.
    
    Args:
        coordinator: Initialized AgentCoordinator
    """
    print(BANNER)
    
    while True:
        try:
            # Get user input
            query = input("🟢 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!\n")
                break
            
            # Process query
            await process_query(coordinator, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")



async def main():
    """
    Main entry point for Amlan Agent.
    """
    try:
        # Load configurations
        print("\n🔱 Starting Amlan Agent...\n")
        agent_config = load_agent_config()
        mcp_configs = load_mcp_config()
        
        # Initialize MCP
        multi_mcp = await initialize_mcp(mcp_configs)
        
        # Initialize agents
        coordinator = await initialize_agents(multi_mcp, agent_config)
        
        await interactive_mode(coordinator)
            
    except FileNotFoundError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Make sure config files exist in config/ directory\n")
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        print("   Check your configuration and try again\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
