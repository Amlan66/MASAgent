"""
DecisionAgent: Creates and revises execution plans using LLM.

Uses Gemini LLM to generate plans and determine next executable steps.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from agents import BaseAgent

from core import (
    DecisionRequest,
    DecisionResponse,
    DecisionMode,
    ActionStep,
    StepType
)

class DecisionAgent(BaseAgent):
    """
    DecisionAgent: Creates plans and determines next steps.
    
    Features:
    - Three decision modes: INITIAL_PLAN, NEXT_STEP, REPLAN
    - Integrates available tools into planning
    - Supports conservative and exploratory strategies
    - Aggressive intra-step chaining
    - JSON parsing with fallback handling
    
    Decision Modes:
        INITIAL_PLAN: Create first plan from user query
        NEXT_STEP: Continue with next step in current plan
        REPLAN: Revise plan after failure or new information
    
    Usage:
        decision = DecisionAgent(
            config={"agent_name": "decision"},
            prompt_path="config/prompts/decision_prompt.txt",
            multi_mcp=multi_mcp_instance,
            api_key="your-api-key"
        )
        await decision.initialize()
        
        request = DecisionRequest(
            mode=DecisionMode.INITIAL_PLAN,
            query="What is 2+2?",
            perception_analysis=perception_response,
            strategy="exploratory",
            context_id=ctx.context_id
        )
        
        response = await decision.execute(request)
        print(response.plan_outline)
        print(response.next_step.code)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_path: str = "config/prompts/decision_prompt.txt",
        multi_mcp = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize DecisionAgent.
        
        Args:
            config: Configuration dict (from profiles.yaml)
            prompt_path: Path to decision prompt template
            multi_mcp: MultiMCP instance for tool descriptions
            api_key: Optional Gemini API key (defaults to env var)
        """

        super().__init__(config)
        load_dotenv()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.prompt_path = Path(prompt_path)
        self.model_name = config.get("llm_model", "gemini-2.0-flash")
        self.multi_mcp = multi_mcp

        self.client: Optional[genai.Client] = None
        self.prompt_template: Optional[str] = None

    async def initialize(self) -> None:
        """
        Initialize the decision agent.
        Loads prompt template and creates gemini client.
        """

        try:
            # Verify API key
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or config")
            
            # Initialize Gemini client
            self.client = genai.Client(api_key=self.api_key)
            
            # Load prompt template
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt not found: {self.prompt_path}")
            
            self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

            # Verify MultiMCP is available
            if not self.multi_mcp:
                print("⚠️ Warning: MultiMCP not provided, tool descriptions unavailable")
            
            self.is_initialized = True
            print(f"✅ DecisionAgent initialized (model: {self.model_name})")
            
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise

    async def process(self, request: DecisionRequest) -> DecisionResponse:
        """
        Process decision request and return plan with next step.
        
        Args:
            request: DecisionRequest with mode, query, context
        
        Returns:
            DecisionResponse with plan and executable step
        """
        start_time = time.perf_counter()

        try:
            
            if request.mode == DecisionMode.INITIAL_PLAN:
                result = await self._create_initial_plan(request)
            elif request.mode == DecisionMode.NEXT_STEP:
                result = await self._generate_next_step(request)
            elif request.mode == DecisionMode.REPLAN:
                result = await self._replan(request)
            else:
                raise ValueError(f"Invalid decision mode: {request.mode}")
            
            print(f"🔧 Decision result: {result}")
            processing_time = time.perf_counter() - start_time
        
            response = DecisionResponse(
                request_id=request.request_id,
                plan_outline=result["plan_text"],
                next_step=ActionStep(
                    step_index=result["step_index"],
                    description=result["description"],
                    type=StepType(result["type"]),
                    code=result.get("code"),
                    conclusion=result.get("conclusion"),
                    rationale=result.get("rationale", result["description"])
                ),
                strategy_used=request.strategy,
                estimated_complexity=self._estimate_complexity(result),
                tools_required=self._extract_tools_from_code(result.get("code", "")),
                planning_rationale=result.get("rationale", ""),
                processing_time=processing_time,
                success=True
            )

            return response

        except Exception as e:
            print(f"❌ DecisionAgent error: {e}")
            import traceback
            traceback.print_exc()
            processing_time = time.perf_counter() - start_time

            # Return fallback response
            return DecisionResponse(
                request_id=request.request_id,
                plan_outline=["Step 0: Failed to create plan"],
                next_step=ActionStep(
                    step_index=0,
                    description=f"Planning error: {str(e)}",
                    type=StepType.NOP,
                    rationale="Planning failed"
                ),
                strategy_used=request.strategy,
                estimated_complexity="unknown",
                tools_required=[],
                planning_rationale=f"Error: {str(e)}",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    #plan methods
    async def _create_initial_plan(self, request: DecisionRequest) -> Dict[str, Any]:
        """
        Create initial execution plan from user query.
        
        Mode: INITIAL_PLAN (plan_mode = "initial")
        
        Args:
            request: DecisionRequest with query and perception
        
        Returns:
            Dict with plan_text and first step (step_index=0)
        """
        # Build decision input
        decision_input = {
            "plan_mode": "initial",
            "planning_strategy": request.strategy,
            "original_query": request.query,
            "perception": request.perception_analysis.model_dump(mode='json') if request.perception_analysis else {},
            "max_steps": request.max_steps
        }
        
        # Call LLM
        return await self._call_llm(decision_input)

    async def _generate_next_step(self, request: DecisionRequest) -> Dict[str, Any]:
        """
        Generate next step in current plan.
        
        Mode: NEXT_STEP (plan_mode = "mid_session")
        
        Args:
            request: DecisionRequest with current plan and completed steps
        
        Returns:
            Dict with plan_text and next step
        """
        # Build decision input
        # Serialize completed steps to avoid datetime serialization issues
        completed_steps_serialized = []
        if request.completed_steps:
            for step in request.completed_steps:
                if hasattr(step, 'model_dump'):
                    completed_steps_serialized.append(step.model_dump(mode='json'))
                else:
                    completed_steps_serialized.append(step)
        
        decision_input = {
            "plan_mode": "mid_session",
            "planning_strategy": request.strategy,
            "original_query": request.query,
            "current_plan_version": len(request.current_plan) if request.current_plan else 1,
            "current_plan": request.current_plan or [],
            "completed_steps": completed_steps_serialized,
            "max_steps": request.max_steps
        }
        
        # Call LLM
        return await self._call_llm(decision_input)

    async def _replan(self, request: DecisionRequest) -> Dict[str, Any]:
        """
        Revise plan after failure or new information.
        
        Mode: REPLAN (plan_mode = "mid_session" with failure context)
        
        Args:
            request: DecisionRequest with failed step info
        
        Returns:
            Dict with revised plan_text and next step
        """
        # Build decision input
        # Serialize completed steps to avoid datetime serialization issues
        completed_steps_serialized = []
        if request.completed_steps:
            for step in request.completed_steps:
                if hasattr(step, 'model_dump'):
                    completed_steps_serialized.append(step.model_dump(mode='json'))
                else:
                    completed_steps_serialized.append(step)
        
        # Serialize current_step/failed_steps if it has datetime fields
        current_step_serialized = {}
        if request.failed_steps:
            if hasattr(request.failed_steps, 'model_dump'):
                current_step_serialized = request.failed_steps.model_dump(mode='json')
            else:
                current_step_serialized = request.failed_steps
        
        decision_input = {
            "plan_mode": "mid_session",
            "planning_strategy": request.strategy,
            "original_query": request.query,
            "current_plan_version": len(request.current_plan) if request.current_plan else 1,
            "current_plan": request.current_plan or [],
            "completed_steps": completed_steps_serialized,
            "current_step": current_step_serialized,
            "max_steps": request.max_steps
        }
        
        # Call LLM
        return await self._call_llm(decision_input)

    #llm interaction
    async def _call_llm(self, decision_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Gemini LLM to generate plan.
        
        Args:
            decision_input: Decision input dict
        
        Returns:
            Parsed decision output dict
        """
        # Build tool descriptions
        tool_descriptions = ""
        if self.multi_mcp:
            function_list_text = self.multi_mcp.tool_description_wrapper()
            tool_descriptions = "\n".join(f"- `{desc.strip()}`" for desc in function_list_text)
            tool_descriptions = "\n\n### The ONLY Available Tools\n\n---\n\n" + tool_descriptions
        
        # Build full prompt
        # Use a custom JSON encoder to handle datetime objects
        import json
        from datetime import datetime
        
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        full_prompt = (
            f"{self.prompt_template.strip()}\n"
            f"{tool_descriptions}\n\n"
            f"```json\n{json.dumps(decision_input, indent=2, cls=DateTimeEncoder)}\n```"
        )
        
        try:
            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            
            raw_text = response.candidates[0].content.parts[0].text.strip()
            
            # Parse JSON from response
            parsed_result = self._parse_llm_output(raw_text)
            return parsed_result
            
        except ServerError as e:
            print(f"🚫 Decision LLM ServerError: {e}")
            return self._get_fallback_response(error=f"LLM server error: {str(e)}")
        
        except Exception as e:
            print(f"❌ Decision LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response(error=str(e))

    def _parse_llm_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse LLM output and extract decision JSON.
        
        Handles multiple formats and includes fallback parsing.
        
        Args:
            raw_text: Raw LLM response text
        
        Returns:
            Parsed decision dict
        """
        try:
            # Extract JSON block using regex
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON block found")
            
            json_block = match.group(1)
            
            try:
                output = json.loads(json_block)
            except json.JSONDecodeError as e:
                print("⚠️ JSON decode failed, attempting salvage via regex...")
                
                # Attempt to extract a 'code' block manually
                code_match = re.search(r'code\s*:\s*"(.*?)"', json_block, re.DOTALL)
                code_value = bytes(code_match.group(1), "utf-8").decode("unicode_escape") if code_match else ""
                
                output = {
                    "step_index": 0,
                    "description": "Recovered partial JSON from LLM.",
                    "type": "code" if code_value else "nop",  # Use lowercase
                    "code": code_value,
                    "conclusion": "",
                    "plan_text": ["Step 0: Partial plan recovered due to JSON decode error."],
                    "raw_text": raw_text[:1000]
                }
            
            # Handle flattened or nested format
            if "next_step" in output:
                output.update(output.pop("next_step"))
            
            # If json_block exists, flatten it to top level
            if "json_block" in output:
                json_block = output.pop("json_block")
                # Flatten json_block fields to top level (they take priority)
                for key in ["step_index", "description", "type", "code", "conclusion", "rationale"]:
                    if key in json_block:
                        output[key] = json_block[key]
                # Normalize type to lowercase
                if "type" in output:
                    output["type"] = output["type"].lower()
            
            # Patch missing fields
            defaults = {
                "step_index": 0,
                "description": "Missing from LLM response",
                "type": "nop",  # Use lowercase to match StepType enum
                "code": "",
                "conclusion": "",
                "plan_text": ["Step 0: No valid plan returned by LLM."],
                "rationale": "Generated by LLM"
            }
            for key, default in defaults.items():
                output.setdefault(key, default)
            
            return output
        
        except Exception as e:
            print(f"❌ Unrecoverable exception while parsing LLM response: {str(e)}")
            return self._get_fallback_response(error=f"Parse error: {str(e)}")


    def _get_fallback_response(self, error: str = "") -> Dict[str, Any]:
        """
        Get fallback response when LLM fails.
        
        Args:
            error: Error message
        
        Returns:
            Safe fallback decision dict
        """
        return {
            "step_index": 0,
            "description": f"Decision failed: {error}" if error else "Decision unavailable",
            "type": "nop",  # Use lowercase to match StepType enum
            "code": "",
            "conclusion": "",
            "plan_text": ["Step 0: Decision agent encountered an error."],
            "rationale": error or "Unknown error"
        }

    # ───────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ───────────────────────────────────────────────────────────────
    
    def _estimate_complexity(self, result: Dict[str, Any]) -> str:
        """
        Estimate plan complexity.
        
        Args:
            result: Decision output dict
        
        Returns:
            Complexity level: "simple", "moderate", "complex"
        """
        plan_steps = len(result.get("plan_text", []))
        code_length = len(result.get("code", ""))
        
        if plan_steps == 1 and code_length < 100:
            return "simple"
        elif plan_steps <= 2 and code_length < 300:
            return "moderate"
        else:
            return "complex"
    
    def _extract_tools_from_code(self, code: str) -> List[str]:
        """
        Extract tool/function names from code.
        
        Args:
            code: Python code string
        
        Returns:
            List of function names called in code
        """
        if not code:
            return []
        
        tools = []
        try:
            import ast
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Skip built-ins
                    if func_name not in ["print", "len", "str", "int", "float", "list", "dict"]:
                        tools.append(func_name)
        except:
            pass
        
        return list(set(tools))  # Remove duplicates
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"DecisionAgent("
            f"model={self.model_name}, "
            f"calls={self.metrics['total_calls']}, "
            f"avg_time={self.metrics.get('average_time', 0):.2f}s)"
        )





        
