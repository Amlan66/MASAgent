# Configuration Files

This directory contains all configuration files for the Sudarshan Chakra Agent system.

## Files Overview

### 1. profiles.yaml
**Purpose**: Main agent configuration and behavior settings

**Key Sections**:
- **agent**: Agent metadata (name, id, description)
- **strategy**: Planning and execution strategy
  - `planning_mode`: conservative or exploratory
  - `exploration_mode`: parallel or sequential
  - `max_steps`: Maximum execution steps (10)
  - `max_lifelines_per_step`: Retry attempts per step (2)
- **memory**: Memory and storage configuration
  - `base_dir`: storage/session_logs
  - `structure`: date-based (YYYY/MM/DD)
- **llm**: Language model settings
  - `text_generation`: gemini (default)
  - `embedding`: nomic
- **agents**: Individual agent configurations
  - Retriever: top_k=3, caching enabled
  - Perception: confidence_threshold=0.7
  - Critic: pre/post validation enabled
  - Decision: aggressive_chaining=true
  - Executor: max_functions=5, timeout=500ms
  - Memory: auto_save enabled
- **workflow**: Workflow engine settings
- **coordinator**: Coordinator behavior

### 2. mcp_server_config.yaml
**Purpose**: MCP (Model Context Protocol) server configurations

**Configured Servers**:
1. **math**: Mathematical operations
   - Tools: add, subtract, multiply, divide, factorial, trigonometry, etc.
   - Script: mcp_server_1.py
   
2. **documents**: Document processing
   - Tools: search_stored_documents_rag, convert_webpage_url_into_markdown, extract_pdf
   - Script: mcp_server_2.py
   
3. **websearch**: Web search and retrieval
   - Tools: duckduckgo_search_results, download_raw_html_from_url
   - Script: mcp_server_3.py
   
4. **mixed**: Additional utilities
   - Tools: basic math operations
   - Script: mcp_server_4.py

**Important**: Update the `cwd` paths to match your installation directory.

### 3. prompts/perception_prompt.txt
**Purpose**: System prompt for the Perception Agent

**Modes**:
- QUERY_ANALYSIS: Analyze initial user queries
- RESULT_ANALYSIS: Evaluate execution step results

**Output Format**: JSON with entities, reasoning, confidence, goal achievement status

**Key Responsibilities**:
- Extract entities and requirements
- Assess query complexity
- Evaluate progress toward goals
- Identify tool failures
- Determine if original goal is achieved

### 4. prompts/decision_prompt.txt
**Purpose**: System prompt for the Decision Agent

**Modes**:
- INITIAL_PLAN: Create first execution plan
- REPLAN: Revise plan based on feedback
- NEXT_STEP: Generate next step in sequence

**Output Format**: Two-part response
1. Natural language plan outline
2. Structured JSON step object

**Step Types**:
- CODE: Execute Python with tool calls
- CONCLUDE: Provide final answer
- NOP: Request clarification

**Key Responsibilities**:
- Create multi-step plans
- Generate executable code
- Chain tool operations
- Handle failures and replanning

### 5. prompts/critic_prompt.txt
**Purpose**: System prompt for the Critic Agent

**Modes**:
- QUERY_VALIDATION: Validate user queries
- PLAN_VALIDATION: Pre-execution plan validation
- RESULT_VALIDATION: Post-execution result validation

**Output Format**: JSON with approval, issues, recommendations, goal status

**Key Responsibilities**:
- Verify tool availability
- Check code syntax
- Identify past failures
- Assess goal achievement
- Recommend next actions

## Configuration Loading

The main.py file loads these configurations in this order:

1. Load `profiles.yaml` → Agent behavior settings
2. Load `mcp_server_config.yaml` → Tool server configurations
3. Initialize MCP servers with loaded configs
4. Initialize agents with configurations from profiles.yaml
5. Load prompts from `config/prompts/` directory

## Customization

### Changing Planning Strategy
Edit `profiles.yaml`:
```yaml
strategy:
  planning_mode: conservative  # or exploratory
  exploration_mode: sequential  # or parallel
```

### Adjusting Step Limits
Edit `profiles.yaml`:
```yaml
strategy:
  max_steps: 15  # increase from 10
  max_lifelines_per_step: 3  # increase retries
```

### Changing LLM Provider
Edit `profiles.yaml`:
```yaml
llm:
  text_generation: phi4  # or gemma3:12b, qwen2.5:32b-instruct-q4_0
```

### Adding New MCP Servers
Edit `mcp_server_config.yaml`:
```yaml
mcp_servers:
  - id: new_server
    script: mcp_server_5.py
    cwd: /path/to/mcp_servers
    description: "Description of new server"
    capabilities:
      - tool1
      - tool2
```

### Modifying Prompts
Directly edit the prompt files in `config/prompts/`:
- Keep the core structure (modes, output format)
- Maintain JSON output schemas
- Add examples as needed
- Update field descriptions

## Environment Variables

Create a `.env` file in the project root for sensitive configurations:
```
GEMINI_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key
# Add other API keys as needed
```

## Validation

After modifying configurations, test with:
```bash
python -c "import yaml; yaml.safe_load(open('config/profiles.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/mcp_server_config.yaml'))"
```

## Troubleshooting

**Issue**: Configuration not loading
- Check YAML syntax (proper indentation)
- Verify file paths are correct
- Ensure files are in correct location

**Issue**: MCP servers not initializing
- Verify `cwd` paths in mcp_server_config.yaml
- Check that server scripts exist
- Ensure Python environment has required dependencies

**Issue**: Prompts not found
- Verify prompts are in `config/prompts/` directory
- Check file names match exactly (case-sensitive)
- Ensure .txt extension is present

## Best Practices

1. **Backup configurations** before making changes
2. **Test changes incrementally** - modify one setting at a time
3. **Use version control** for configuration files
4. **Document custom changes** in comments
5. **Keep prompts focused** - avoid overly verbose instructions
6. **Monitor performance** after configuration changes

## Default Behavior

If configuration files are missing or invalid, the system uses these defaults:
- Planning mode: exploratory
- Max steps: 10
- Max retries: 2
- LLM provider: gemini
- Memory base: storage/session_logs

## Additional Resources

- Main implementation: `../main.py`
- Agent implementations: `../agents/`
- Workflow engine: `../orchestrator/workflow_engine.py`
- Architecture docs: `../ARCHITECTURE_SUMMARY.md`

