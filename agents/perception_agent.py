"""
PerceptionAgent: Analyzes queries and execution results using LLM.

Uses Gemini LLM to interpret user queries and step results in ERORLL format.
"""
import os
import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from agents import BaseAgent
from core import (
    PerceptionRequest,
    PerceptionResponse,
    AnalysisMode,
    MemoryResult
)

class PerceptionAgent(BaseAgent):
    """
    PerceptionAgent: Analyzes queries and results using LLM.

    Features:
    - Two analysis modes: QUERY_ANALYSIS and RESULT_ANALYSIS
    - ERORLL format output (Entities, Requirements, Original/Local goals, Reasoning)
    - Memory context integration
    - Gemini 2.0 Flash LLM
    - Structured JSON parsing with fallbacks

    Modes:
        - QUERY_ANALYSIS: Interpret initial user query
        - RESULT_ANALYSIS: Evaluate step execution results
    
    Usage:
        perception = PerceptionAgent(
            config={"agent_name": "perception"},
            prompt_path="config/prompts/perception_prompt.txt",
            api_key="your-api-key"
        )
        await perception.initialize()
        
        request = PerceptionRequest(
            analysis_mode=AnalysisMode.QUERY_ANALYSIS,
            content="What is 2+2?",
            retrieved_context=[memory_result1, memory_result2],
            context_id=ctx.context_id
        )
        
        response = await perception.execute(request)
        print(response.entities, response.reasoning)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_path: str = "config/prompts/perception_prompt.txt",
        api_key: Optional[str] = None
    ): 

        """
        Initialize PerceptionAgent

        Args:
            config : from profiles.yaml
            prompt_path : path to perception prompt file
            api_key : GEMINI_API_KEY
        """

        super().__init__(config)

        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.prompt_path = Path(prompt_path)
        self.model_name = config.get("llm_model", "gemini-2.0-flash")
        
        self.client: Optional[genai.Client] = None
        self.prompt_template: Optional[str] = None

    
    async def initialize(self) -> None:
        """
        Initialize the perception agent
        loads prompt template and creates gemini client
        """

        try:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or config.")
            
            self.client = genai.Client(api_key=self.api_key)
        
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

            self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

            self.is_initialized = True
            print(f"✅ PerceptionAgent initialized with model: {self.model_name}")

        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise

    async def process(self, request: PerceptionRequest) -> PerceptionResponse:
        """
        Process perception request and return analysis.

        Args:
            request: PerceptionRequest with mode, content, context
        
        Returns
            PerceptionResponse with analysis
        """

        start_time = time.perf_counter()

        try:
            #Map to right analysis mode
            if request.analysis_mode == AnalysisMode.QUERY_ANALYSIS:
                result = await self._analyze_query(request)
            elif request.analysis_mode == AnalysisMode.RESULT_ANALYSIS:
                result = await self._analyze_result(request)
            else:
                raise ValueError(f"Invalid analysis mode: {request.analysis_mode}")
                
            processing_time = time.perf_counter() - start_time
            
            response = PerceptionResponse(
                request_id=request.request_id,
                entities=result.get("entities", []),
                requirements={"result_requirement": result.get("result_requirement", "")},
                reasoning=result.get("reasoning", ""),
                intent=result.get("result_requirement", ""),
                complexity=self._assess_complexity(result),
                confidence=float(result.get("confidence", 0.0)),
                extracted_data={
                    "original_goal_achieved": result.get("original_goal_achieved", False),
                    "local_goal_achieved": result.get("local_goal_achieved", False),
                    "local_reasoning": result.get("local_reasoning", ""),
                    "last_tooluse_summary": result.get("last_tooluse_summary", ""),
                    "solution_summary": result.get("solution_summary", "")
                },
                quality_assessment=result.get("local_reasoning", ""),
                processing_time=processing_time,
                success=True
            )

            return response
        
        except Exception as e:
            processing_time = time.perf_counter() - start_time

            return PerceptionResponse(
                request_id=request.request_id,
                entities=[],
                requirements={"result_requirement": "N/A"},
                reasoning=f"Perception error: {str(e)}",
                intent="unknown",
                complexity="unknown",
                confidence=0.0,
                extracted_data={
                    "original_goal_achieved": False,
                    "local_goal_achieved": False,
                    "solution_summary": "Not ready yet"
                },
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    async def _analyze_query(self, request: PerceptionRequest) -> Dict[str, Any]:
        """
        Analyze initial user query

        Mode: QUERY_ANALYSIS (snapshot_type = "user_query")

        Args
            request: PerceptionRequest with query
        
        Returns:
            Dict with ERORLL fields
        """

        perception_input = self._build_perception_input(
            raw_input=request.content,
            memory=request.retrieved_context or [],
            snapshot_type="user_query",
            current_plan=request.current_plan
        )

        return await self._call_llm(perception_input)
    
    async def _analyze_result(self, request: PerceptionRequest) -> Dict[str, Any]:
        """
        Analyze step execution result.
        
        Mode: RESULT_ANALYSIS (snapshot_type = "step_result")
        
        Args:
            request: PerceptionRequest with result content
        
        Returns:
            Dict with ERORLL fields
        """
        # Build perception input
        perception_input = self._build_perception_input(
            raw_input=request.content,
            memory=request.retrieved_context or [],
            snapshot_type="step_result",
            current_plan=request.current_plan
        )
        
        # Call LLM
        return await self._call_llm(perception_input)

    #LLM Interaction

    def _build_perception_input(
        self,
        raw_input: str,
        memory: List[MemoryResult],
        snapshot_type: str = "user_query",
        current_plan: Optional[str]=None
    ) -> Dict[str, Any]:
        """
        Build Perception Input Dict for LLM

        Args:
            raw_input: Query or result text
            memory: Retrieved memory results
            snapshot_type: "user_query" or "step_result"
            current_plan: Optional current plan text
        
        Returns:
            Dict ready for JSON serialization
        """

        # Convert MemoryResult objects to dict format
        if memory:
            memory_excerpt = {
                f"memory_{i+1}": {
                    "query": mem.query,
                    "result_requirement": mem.result_requirement,
                    "solution_summary": mem.solution_summary
                }
                for i, mem in enumerate(memory)
            }
        else:
            memory_excerpt = {}
        
        return {
            "run_id": str(uuid.uuid4()),
            "snapshot_type": snapshot_type,
            "raw_input": raw_input,
            "memory_excerpt": memory_excerpt,
            "prev_objective": "",
            "prev_confidence": None,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "schema_version": 1,
            "current_plan": current_plan or "Initial Query Mode, plan not created"
        }

    async def _call_llm(self, perception_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Gemini LLM with perception input.
        
        Args:
            perception_input: Perception input dict
        
        Returns:
            Parsed ERORLL dict
        """
        # Build full prompt
        full_prompt = (
            f"{self.prompt_template.strip()}\n\n"
            f"```json\n{json.dumps(perception_input, indent=2)}\n```"
        )
        
        try:
            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            
            raw_text = response.text.strip()
            
            # Parse JSON from response
            return self._parse_llm_output(raw_text)
            
        except ServerError as e:
            print(f"🚫 Perception LLM ServerError: {e}")
            return self._get_fallback_response(error=f"LLM server error: {str(e)}")
        
        except Exception as e:
            print(f"❌ Perception LLM error: {e}")
            return self._get_fallback_response(error=str(e))
    

    def _parse_llm_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse LLM output and extract ERORLL JSON.
        
        Args:
            raw_text: Raw LLM response text
        
        Returns:
            Parsed ERORLL dict
        """
        try:
            # Extract JSON block
            json_block = raw_text.split("```json")[1].split("```")[0].strip()
            output = json.loads(json_block)
            
            # Patch missing fields (from original)
            required_fields = {
                "entities": [],
                "result_requirement": "No requirement specified.",
                "original_goal_achieved": False,
                "reasoning": "No reasoning given.",
                "local_goal_achieved": False,
                "local_reasoning": "No local reasoning given.",
                "last_tooluse_summary": "None",
                "solution_summary": "No summary.",
                "confidence": "0.0"
            }
            
            for key, default in required_fields.items():
                output.setdefault(key, default)
            
            return output
            
        except Exception as e:
            print(f"❌ Failed to parse perception output: {e}")
            return self._get_fallback_response(error=f"Parse error: {str(e)}")

    def _get_fallback_response(self, error: str = "") -> Dict[str, Any]:
        """
        Get fallback response when LLM fails.
        
        Args:
            error: Error message
        
        Returns:
            Safe fallback ERORLL dict
        """
        return {
            "entities": [],
            "result_requirement": "N/A",
            "original_goal_achieved": False,
            "reasoning": f"Perception failed: {error}" if error else "Perception unavailable.",
            "local_goal_achieved": False,
            "local_reasoning": "Could not extract structured information.",
            "last_tooluse_summary": "None",
            "solution_summary": "Not ready yet",
            "confidence": "0.0"
        }

    def _assess_complexity(self, result: Dict[str, Any]) -> str:
        """
        Assess complexity based on perception result.
        
        Args:
            result: ERORLL dict
        
        Returns:
            Complexity level: "simple", "moderate", "complex"
        """
        entity_count = len(result.get("entities", []))
        confidence = float(result.get("confidence", 0))
        
        if entity_count == 0:
            return "simple"
        elif entity_count <= 3 and confidence > 0.7:
            return "simple"
        elif entity_count <= 6:
            return "moderate"
        else:
            return "complex"

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"PerceptionAgent("
            f"model={self.model_name}, "
            f"calls={self.metrics['total_calls']}, "
            f"avg_time={self.metrics.get('average_time', 0):.2f}s)"
        )

