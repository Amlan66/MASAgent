# MAS Agent 

## Overview

This Agent is an advanced AI reasoning system that processes user queries through a sophisticated multi-agent architecture with a state machine-driven workflow engine. The system is designed to handle complex tasks by breaking them down into manageable steps, with built-in validation, memory retrieval, and adaptive planning capabilities.

## Architecture

### Core Components

The system is built around three main architectural pillars:

1. **Multi-Agent System**: Six specialized agents handle different aspects of query processing
2. **Workflow State Engine**: A state machine that orchestrates the execution flow
3. **Execution Context**: Shared state management across all agents

## Agent System

### 1. Base Agent (`agents/base_agent.py`)

All agents inherit from the `BaseAgent` abstract class, providing:
- Consistent API across all agents (`process()` method)
- Built-in metrics tracking (calls, execution time, errors)
- Request validation and error handling
- Performance monitoring capabilities

### 2. Retriever Agent (`agents/retriever_agent.py`)

**Purpose**: Searches for relevant information from past successful sessions

**Key Features**:
- Uses RapidFuzz algorithm for fuzzy string matching
- Scoring formula: `0.5 * query_score + 0.4 * summary_score - 0.05 * length_penalty`
- Searches through stored session logs in `storage/session_logs/`
- Returns top 3 most relevant results
- Supports multiple retrieval sources (memory, future RAG integration)

**Methods**:
- `process()`: Main entry point for retrieval requests
- `_search_memory()`: Core memory search functionality
- `_load_memory_entries()`: Loads session data from disk
- `_extract_entry()`: Extracts relevant information from sessions

### 3. Perception Agent (`agents/perception_agent.py`)

**Purpose**: Analyzes user queries and execution results using LLM (Gemini 2.0 Flash)

**Key Features**:
- Two analysis modes: `QUERY_ANALYSIS` and `RESULT_ANALYSIS`
- ERORLL format output (Entities, Requirements, Original/Local goals, Reasoning, Learning)
- Memory context integration
- Structured JSON parsing with fallback handling

**Analysis Modes**:
- **QUERY_ANALYSIS**: Interprets initial user queries, extracts entities and requirements
- **RESULT_ANALYSIS**: Evaluates step execution results and learning outcomes

**Methods**:
- `process()`: Main processing method
- `_analyze_query()`: Analyzes user queries
- `_analyze_result()`: Analyzes execution results
- `_build_query_prompt()`: Constructs LLM prompts
- `_parse_llm_output()`: Parses structured LLM responses

### 4. Critic Agent (`agents/critic_agent.py`)

**Purpose**: Validates queries, plans, and results before and after execution

**Key Features**:
- Three validation modes: `QUERY_VALIDATION`, `PLAN_VALIDATION`, `RESULT_VALIDATION`
- Pre-execution validation (prevents invalid plans from running)
- Post-execution evaluation (determines success/failure)
- Rules-based validation engine
- LLM-powered validation for complex scenarios

**Validation Modes**:
- **QUERY_VALIDATION**: Checks if queries are clear and answerable
- **PLAN_VALIDATION**: Validates plans before execution (tool availability, syntax, logic)
- **RESULT_VALIDATION**: Evaluates if steps succeeded and goals were achieved

**Methods**:
- `process()`: Main validation entry point
- `_validate_query()`: Query validation logic
- `_validate_plan()`: Plan validation logic
- `_validate_result()`: Result validation logic
- `_check_tools_available()`: Verifies tool availability
- `_check_past_failures()`: Learns from previous failures

### 5. Decision Agent (`agents/decision_agent.py`)

**Purpose**: Creates and revises execution plans using LLM

**Key Features**:
- Three decision modes: `INITIAL_PLAN`, `NEXT_STEP`, `REPLAN`
- Integrates available MCP tools into planning
- Supports conservative and exploratory strategies
- Aggressive intra-step chaining capabilities
- JSON parsing with robust fallback handling

**Decision Modes**:
- **INITIAL_PLAN**: Creates first plan from user query
- **NEXT_STEP**: Continues with next step in current plan
- **REPLAN**: Revises plan after failure or new information

**Methods**:
- `process()`: Main decision processing
- `_create_initial_plan()`: Initial plan generation
- `_generate_next_step()`: Next step determination
- `_replan()`: Plan revision logic
- `_build_planning_context()`: Context preparation for LLM

### 6. Executor Agent (`agents/executor_agent.py`)

**Purpose**: Executes Python code with MCP tool integration in a sandboxed environment

**Key Features**:
- AST transformations for code safety (`KeywordStripper`, `AwaitTransformer`)
- Sandboxed execution with restricted builtins
- MCP tool proxy integration
- Function call counting and timeout protection
- Safe globals environment construction

**Security Features**:
- Removes dangerous keywords (`import`, `exec`, `eval`, etc.)
- Transforms async/await syntax
- Limits function calls (MAX_FUNCTIONS = 5)
- Timeout protection (TIMEOUT_PER_FUNCTION = 500ms)

**Methods**:
- `process()`: Main execution entry point
- `_execute_code()`: Core code execution logic
- `_count_function_calls()`: Monitors function usage
- `_build_safe_globals()`: Creates secure execution environment
- `_make_tool_proxy()`: Creates MCP tool proxies

### 7. Memory Agent (`agents/memory_agent.py`)

**Purpose**: Manages session persistence and retrieval

**Key Features**:
- JSON-based session storage
- Date-based directory structure (`YYYY/MM/DD/<uuid>.json`)
- Session model serialization/deserialization
- Automatic session path generation

**Methods**:
- `save_session()`: Persists SessionModel to disk
- `load_session()`: Loads SessionModel from disk
- `_get_session_path()`: Generates file paths

## Workflow State Engine

### State Machine Overview

The Workflow Engine (`orchestrator/workflow_engine.py`) implements a state machine that controls flow with explicit state handlers. Each state represents a distinct phase of query processing.

### Workflow States

```
INIT
  ↓
RETRIEVE (search memory, retriever agent)
  ↓
ANALYZE_QUERY (perception agent)
  ↓
VALIDATE_QUERY (critic agent)
  ↓
  ├→ If answer in memory → COMPLETE
  └→ If need planning → PLAN (decision agent)
       ↓
     VALIDATE_PLAN (criti agent)
       ↓
       ├→ If rejected → PLAN (retry) (decision agent)
       └→ If approved → EXECUTE_STEP (executor agent)
            ↓
          ANALYZE_RESULT (perception agent)
            ↓
          VALIDATE_RESULT (critic agent)
            ↓
            ├→ If goal achieved → COMPLETE
            ├→ If step ok → DECIDE_NEXT (executor agent)
            └→ If step failed → PLAN (replan) (decision agent)
```



#### State Definitions

1. **INIT**: Initial state when query is received
   - Sets up ExecutionContext
   - Prepares for retrieval
   - **Next**: RETRIEVE

2. **RETRIEVE**: Search for relevant memories
   - **Agent**: RetrieverAgent
   - Searches past successful sessions
   - **Next**: ANALYZE_QUERY

3. **ANALYZE_QUERY**: Understand the user's query
   - **Agent**: PerceptionAgent (QUERY_ANALYSIS mode)
   - Extracts entities, intent, requirements
   - **Next**: VALIDATE_QUERY

4. **VALIDATE_QUERY**: Validate if query is clear and answerable
   - **Agent**: CriticAgent (QUERY_VALIDATION mode)
   - **Next**: 
     - COMPLETE (if answer found in memory)
     - PLAN (if need to execute steps)
     - FAILED (if query is unclear/unanswerable)

5. **PLAN**: Create or revise execution plan
   - **Agent**: DecisionAgent (INITIAL_PLAN or REPLAN mode)
   - **Next**: VALIDATE_PLAN

6. **VALIDATE_PLAN**: Validate plan before execution 
   - **Agent**: CriticAgent (PLAN_VALIDATION mode)
   - **Next**:
     - EXECUTE_STEP (if plan approved)
     - PLAN (if plan rejected - retry)
     - FAILED (if unrecoverable)

7. **EXECUTE_STEP**: Execute current step's code
   - **Agent**: ExecutorAgent
   - Runs Python code with MCP tools
   - **Next**: ANALYZE_RESULT

8. **ANALYZE_RESULT**: Analyze execution result
   - **Agent**: PerceptionAgent (RESULT_ANALYSIS mode)
   - Extracts learning and outcomes
   - **Next**: VALIDATE_RESULT

9. **VALIDATE_RESULT**: Validate step success and goal achievement
   - **Agent**: CriticAgent (RESULT_VALIDATION mode)
   - **Next**:
     - COMPLETE (if original goal achieved)
     - DECIDE_NEXT (if step succeeded but goal not achieved)
     - PLAN (if step failed - replan)

10. **DECIDE_NEXT**: Decide what to do after successful step
    - **Agent**: DecisionAgent (NEXT_STEP mode)
    - **Next**:
      - EXECUTE_STEP (if more steps in plan)
      - COMPLETE (if all steps done)
      - PLAN (if need replanning)

11. **COMPLETE**: Terminal success state
    - Query successfully answered
    - Session saved to memory

12. **FAILED**: Terminal failure state
    - Max retries exceeded or unrecoverable error
    - Session saved with failure status

### State Transitions

The state machine enforces valid transitions through the `WorkflowState.get_allowed_transitions()` method:

```python
transitions = {
    INIT: {RETRIEVE},
    RETRIEVE: {ANALYZE_QUERY},
    ANALYZE_QUERY: {VALIDATE_QUERY},
    VALIDATE_QUERY: {COMPLETE, PLAN, FAILED},
    PLAN: {VALIDATE_PLAN},
    VALIDATE_PLAN: {EXECUTE_STEP, PLAN, FAILED},
    EXECUTE_STEP: {ANALYZE_RESULT},
    ANALYZE_RESULT: {VALIDATE_RESULT},
    VALIDATE_RESULT: {COMPLETE, DECIDE_NEXT, PLAN},
    DECIDE_NEXT: {EXECUTE_STEP, COMPLETE, PLAN},
    COMPLETE: set(),  # Terminal
    FAILED: set()     # Terminal
}
```

### State Handlers

Each state has a dedicated handler class that:
- Processes the current state
- Calls appropriate agents
- Determines the next state
- Validates state transitions
- Records transition history

Example state handler:
```python
class ExecuteStepStateHandler(StateHandler):
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        # Execute current step using ExecutorAgent
        execution_request = ExecutionRequest(...)
        execution_response = await coordinator.execute_step(execution_request)
        
        # Update context with results
        context.add_step_result(execution_response)
        
        # Transition to next state
        return self.transition(context, WorkflowState.ANALYZE_RESULT, "Step executed")
```

## Orchestration Layer

### Agent Coordinator (`orchestrator/coordinator.py`)

The AgentCoordinator manages all agents and orchestrates query processing:

**Responsibilities**:
1. Initialize all agents with configuration
2. Create ExecutionContext for each query
3. Delegate execution to WorkflowEngine
4. Provide agent wrapper methods for state handlers
5. Save completed sessions via MemoryAgent

**Key Methods**:
- `handle_query()`: Main entry point for query processing
- `retrieve()`: Wrapper for RetrieverAgent
- `analyze_query()`: Wrapper for PerceptionAgent (query analysis)
- `validate_query()`: Wrapper for CriticAgent (query validation)
- `create_plan()`: Wrapper for DecisionAgent (planning)
- `execute_step()`: Wrapper for ExecutorAgent
- `save_session()`: Wrapper for MemoryAgent

### Execution Context (`core/context.py`)

Shared state object that travels through all states:

```python
class ExecutionContext:
    session_id: str
    original_query: str
    retrieved_memories: List[MemoryResult]
    query_analysis: Optional[PerceptionResponse]
    current_plan: Optional[PlanModel]
    executed_steps: List[StepModel]
    step_failures: List[Dict]
    config: Dict[str, Any]
    # ... and more
```

## Configuration System

### Profiles Configuration (`config/profiles.yaml`)

The system loads configuration from `profiles.yaml`:

```yaml
agent:
  name: "Amlan Agent"
  id: "amlan_agent_v1"

strategy:
  planning_mode: "exploratory"  # or "conservative"
  exploration_mode: "parallel"  # or "sequential"
  max_steps: 10
  max_lifelines_per_step: 2

memory:
  storage:
    base_dir: "storage/session_logs"
    structure: "date_based"  # YYYY/MM/DD

llm:
  text_generation: "gemini"
  embedding: "nomic"

agents:
  retriever:
    top_k: 3
    caching_enabled: true
  perception:
    confidence_threshold: 0.7
  critic:
    pre_validation_enabled: true
    post_validation_enabled: true
  decision:
    aggressive_chaining: true
  executor:
    max_functions: 5
    timeout: 500
  memory:
    auto_save: true
```

### MCP Server Configuration (`config/mcp_server_config.yaml`)

Configures Model Context Protocol servers for tool integration:

1. **Math Server**: Mathematical operations (add, subtract, multiply, etc.)
2. **Documents Server**: Document processing and RAG
3. **Web Search Server**: DuckDuckGo search and web scraping
4. **Mixed Server**: Additional utility functions

## Data Models (`core/models.py`)

The system uses Pydantic models for type safety and validation:

**Key Model Categories**:
- **Request/Response Models**: For agent communication
- **Domain Models**: StepModel, PlanModel, SessionModel
- **Configuration Models**: AgentConfig, CoordinatorConfig
- **Enums**: WorkflowState, AnalysisMode, CriticMode, DecisionMode


## Key Features

### 1. **Pre-validation** 
- Plans are validated before execution
- Prevents wasted computation on invalid plans
- Checks tool availability and code syntax

### 2. **Adaptive Replanning**
- System can revise plans based on execution results
- Learns from failures and adjusts strategy
- Maintains plan history for debugging

### 3. **Memory Integration**
- Searches past successful sessions
- Provides context for similar queries
- Learns from previous solutions

### 4. **Robust Error Handling**
- Multiple retry mechanisms
- Graceful degradation on failures
- Comprehensive error logging

### 5. **Tool Integration**
- Seamless MCP tool integration
- Safe code execution environment
- Function call monitoring and limits

### 6. **Metrics and Monitoring**
- Per-agent performance metrics
- Execution time tracking
- Success/failure rates
- Error categorization

## File Structure

```
amlan_agent/
├── main.py                     # Entry point
├── agents/                     # Agent implementations
│   ├── base_agent.py          # Abstract base class
│   ├── retriever_agent.py     # Memory search
│   ├── perception_agent.py    # Query/result analysis
│   ├── critic_agent.py        # Validation (NEW!)
│   ├── decision_agent.py      # Planning
│   ├── executor_agent.py      # Code execution
│   └── memory_agent.py        # Session persistence
├── orchestrator/               # Orchestration layer
│   ├── coordinator.py         # Agent coordination
│   └── workflow_engine.py     # State machine
├── core/                      # Core data structures
│   ├── models.py             # Pydantic models
│   ├── context.py            # Execution context
│   └── states.py             # State definitions
├── mcp_servers/              # MCP tool servers
│   ├── multiMCP.py           # Tool registry
│   ├── mcp_server_1.py       # Math tools
│   ├── mcp_server_2.py       # Document tools
│   ├── mcp_server_3.py       # Web search tools
│   └── mcp_server_4.py       # Mixed tools
├── config/                   # Configuration files
│   ├── profiles.yaml         # Agent configuration
│   ├── mcp_server_config.yaml # MCP configuration
│   └── prompts/              # LLM prompts
└── storage/                  # Runtime storage
    └── session_logs/         # Session persistence

