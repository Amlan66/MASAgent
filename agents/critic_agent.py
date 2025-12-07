"""
CriticAgent: Validates queries, plans, and results.
Uses both rule-based checks and LLM-based evaluation for comprehensive validation.
"""

import os
import json
import time
import ast
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from agents import BaseAgent
from core import (
    CriticRequest,
    CriticResponse,
    CriticMode,
    ValidationIssue,
    GoalStatus
)

class CriticAgent(BaseAgent):
    """
    CriticAgent: Validates queries, plans, and execution results.
    Features:
    - Three validation modes: QUERY, PLAN, RESULT
    - Hybrid approach: Rule-based + LLM-based validation
    - Pre-execution validation (prevents bad plans)
    - Post-execution evaluation (assesses progress)
    - Tool availability checking
    - Past failure analysis
    - Code syntax validation

    Validation Modes:
        QUERY_VALIDATION: Is the query clear and answerable?
        PLAN_VALIDATION: Is the plan feasible before execution?
        RESULT_VALIDATION: Did the step succeed? Is goal achieved?

    Usage:
        critic = CriticAgent(
            config={"agent_name": "critic"},
            prompt_path="config/prompts/critic_prompt.txt",
            api_key="your-api-key"
        )
        await critic.initialize()
        
        # Validate a plan
        request = CriticRequest(
            critic_mode=CriticMode.PLAN_VALIDATION,
            subject={"plan": plan_text, "code": code_to_execute},
            available_tools=tool_list,
            past_failures=failure_history,
            context_id=ctx.context_id
        )
        
        response = await critic.execute(request)
        if response.approved:
            # Execute the plan
        else:
            # Show issues and replan    
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_path: str = "config/prompts/critic_prompt.txt",
        api_key: Optional[str] = None
    ):
        """
        Initialize CriticAgent

        Args:
            config: Configuration dictionary loaded from profiles.yaml
            prompt_path: Path to critic prompt file
            api_key: GEMINI_API_KEY
        """

        super().__init__(config)

        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.prompt_path = Path(prompt_path)
        self.model_name = config.get("llm_model", "gemini-2.0-flash")

        # Validation settings
        self.use_llm = config.get("use_llm_validation", True)
        self.strict_mode = config.get("strict_mode", False)
        self.min_confidence = config.get("min_confidence", 0.6)

        self.client: Optional[genai.Client] = None
        self.prompt_template: Optional[str] = None

    async def initialize(self) -> None:
        """
        Initialize the critic agent
        Loads prompt template and creates Gemini client if LLM validation is enabled.
        """
        try:
            # Initialize LLM client if needed
            if self.use_llm:
                if not self.api_key:
                    raise ValueError("GEMINI_API_KEY required for LLM validation")
                
                self.client = genai.Client(api_key=self.api_key)
                
                # Load prompt template if it exists
                if self.prompt_path.exists():
                    self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
                else:
                    print(f"⚠️ Critic prompt not found: {self.prompt_path}, using rule-based only")
                    self.use_llm = False
            
            self.is_initialized = True
            mode = "hybrid (rules + LLM)" if self.use_llm else "rule-based only"
            print(f"✅ CriticAgent initialized ({mode})")
            
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise
    
    async def process(self, request: CriticRequest) -> CriticResponse:
        """
        Process critic request and return validation results.
        
        Args:
            request: CriticRequest with mode, subject, context
        
        Returns:
            CriticResponse with validation results
        """
        start_time = time.perf_counter()

        try:
            if request.critic_mode == CriticMode.QUERY_VALIDATION:
                result = await self._validate_query(request)
            
            elif request.critic_mode == CriticMode.PLAN_VALIDATION:
                result = await self._validate_plan(request)
            
            elif request.critic_mode == CriticMode.RESULT_VALIDATION:
                result = await self._validate_result(request)
            else:
                raise ValueError(f"Invalid critic mode: {request.critic_mode}")
            
            processing_time = time.perf_counter() - start_time
        
            response = CriticResponse(
                request_id=request.request_id,
                approved=result["approved"],
                confidence=result["confidence"],
                evaluation=result["evaluation"],
                issues=result["issues"],
                recommendations=result["recommendations"],
                goal_status=result.get("goal_status"),
                next_action=result.get("next_action"),
                processing_time=processing_time,
                success=True
            )

            return response

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            return CriticResponse(
                request_id=request.request_id,
                approved=False,
                confidence=0.0,
                evaluation={"error": str(e)},
                issues=[
                    ValidationIssue(
                        severity="high",
                        type="validation_error",
                        description=f"Critic error: {str(e)}",
                        suggestion="Review the request and try again"
                    )
                ],
                recommendations=["Fix validation error before proceeding"],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    async def _validate_query(self, request: CriticRequest) -> Dict[str, Any]:
        """
        Validate if user query is clear and answerable.

        Checks:
        - Query is not empty
        - Query is understandable
        - Query has clear intent
        - Query is within system capabilities
        
        Args:
            request: CriticRequest with query in subject
        
        Returns:
            Dict with validation results
        """
        issues=[]
        recommendations=[]
        query = request.subject.get("query", "")

        if not query or len(query.strip()) < 3:
            issues.append(ValidationIssue(
                severity="high",
                type="empty_query",
                description="Query is empty or too short",
                suggestion="Please provide a clear and concise query"
            ))
        
        ambiguous_terms = ["it", "that", "this", "them"]
        if any(term in query.lower().strip() for term in ambiguous_terms):
            issues.append(ValidationIssue(
                severity="low",
                type="ambiguous_query",
                description="Query contains ambiguous references like 'it', 'that', 'this', 'them'",
                suggestion="Be more specific about what you're asking"
            ))
        
        llm_result = {}
        if self.use_llm and self.prompt_template:
            llm_result = await self._llm_validate_query(query)
            # Convert LLM issues to ValidationIssue objects
            issues.extend(self._convert_llm_issues_to_validation_issues(llm_result.get("issues", [])))
            recommendations.extend(self._convert_llm_recommendations_to_strings(llm_result.get("recommendations", [])))
        
        high_severity_issues = [i for i in issues if i.severity == "high"]
        approved = len(high_severity_issues) == 0
        confidence = 1.0 - (len(issues) * 0.2)  # Decrease by 0.2 per issue

        return {
            "approved": approved,
            "confidence": max(0.0, confidence),
            "evaluation": {
                "query_length": len(query),
                "has_ambiguity": len([i for i in issues if i.type == "ambiguous_query"]) > 0,
                "llm_evaluation": llm_result.get("evaluation", {})
            },
            "issues": issues,
            "recommendations": recommendations or ["Query looks good, proceed"],
            "next_action": "proceed" if approved else "clarify_query"
        }

    async def _validate_plan(self, request: CriticRequest) -> Dict[str, Any]:
        """
        Validate execution plan before running it.

        Checks:
        - All required tools are available
        - Code syntax is valid (if code present)
        - Plan doesn't repeat past failures
        - Plan is logically sound
        - No dangerous operations
        
        Args:
            request: CriticRequest with plan and code
        
        Returns:
            Dict with validation results
        """
        issues=[]
        recommendations=[]

        plan = request.subject.get("plan", "")
        code = request.subject.get("code", "")
        step_description = request.subject.get("description", "")

        if request.check_tool_availability and code:
            tool_issues = self._check_tool_availability(code, request.available_tools or [])
            issues.extend(tool_issues)
        
        if code:
            syntax_issues = self._check_code_syntax(code)
            issues.extend(syntax_issues)
            danger_issues = self._check_dangerous_operations(code)
            issues.extend(danger_issues)
        
        if request.past_failures:
            failure_issues = self._check_past_failures(
                code,
                step_description,
                request.past_failures
            )
            issues.extend(failure_issues)
        
        llm_result = {}
        if self.use_llm and self.prompt_template:
            llm_result = await self._llm_validate_plan(plan, code, step_description)
            # Convert LLM issues to ValidationIssue objects
            issues.extend(self._convert_llm_issues_to_validation_issues(llm_result.get("issues", [])))
            recommendations.extend(self._convert_llm_recommendations_to_strings(llm_result.get("recommendations", [])))

        high_severity_issues = [i for i in issues if i.severity == "high"]
        medium_severity_issues = [i for i in issues if i.severity == "medium"]

        if self.strict_mode:
            approved = len(high_severity_issues) == 0 and len(medium_severity_issues) == 0
        else:
            approved = len(high_severity_issues) == 0
        
        confidence = 1.0 - (len(high_severity_issues) * 0.3) - (len(medium_severity_issues) * 0.1)

        return {
            "approved": approved,
            "confidence": max(0.0, confidence),
            "evaluation":{
                "has_code": bool(code),
                "code_length": len(code) if code else 0,
                "tools_checked": len(request.available_tools or []),
                "past_failures_checked": len(request.past_failures or []),
                "llm_evaluation": llm_result.get("evaluation", {})
            },
            "issues": issues,
            "recommendations": recommendations or ["Plan looks good, proceed with execution"],
            "next_action": "execute" if approved else "replan"
        }

    async def _validate_result(self, request: CriticRequest) -> Dict[str, Any]:
        """
        Validate execution result and assessgoal achievement.
        Checks:
        - Step executed successfully
        - Result is useful/meaningful
        - Progress toward original goal
        - Goal achievement status

        Args:
            request: CriticRequest with result and goal info
        
        Returns:
            Dict with validation results and goal status
        """

        issues = []
        recommendations = []
        
        result = request.subject.get("result", "")
        error = request.subject.get("error", "")
        step_description = request.subject.get("description", "")
        original_goal = request.original_goal or ""

        if error:
            issues.append(ValidationIssue(
                severity="high",
                type="execution_error",
                description=f"Step failed: {error}",
                suggestion="Review error and replan with different approach"
            ))
        
        if not error and (not result or result.lower() in ["none", "null", ""]):
            issues.append(ValidationIssue(
                severity="medium",
                type="empty_result",
                description="Step produced no meaningful output",
                suggestion="Check if tool returned data or try alternative approach"
            ))
        
        goal_status = None
        llm_result = {}

        if request.check_goal_achievement and self.use_llm and self.prompt_template:
            llm_result = await self._llm_validate_result(
                result, 
                error,
                step_description, 
                original_goal
            )
            
            # Convert goal_status dict to GoalStatus object
            goal_status_dict = llm_result.get("goal_status")
            if goal_status_dict and isinstance(goal_status_dict, dict):
                goal_status = GoalStatus(**goal_status_dict)
            
            issues.extend(self._convert_llm_issues_to_validation_issues(llm_result.get("issues", [])))
            recommendations.extend(self._convert_llm_recommendations_to_strings(llm_result.get("recommendations", [])))
        
        # Determine approval (step was useful)
        step_succeeded = not bool(error)
        high_severity_issues = [i for i in issues if i.severity == "high"]
        approved = step_succeeded and len(high_severity_issues) == 0
        
        confidence = 0.9 if step_succeeded else 0.2
        confidence -= len(high_severity_issues) * 0.2

        if goal_status and goal_status.original_goal_achieved:
            next_action = "complete"
        elif approved:
            next_action = "continue"
        else:
            next_action = "replan"
        
        return {
            "approved": approved,
            "confidence": max(0.0, confidence),
            "evaluation": {
                "step_succeeded": step_succeeded,
                "has_error": bool(error),
                "result_length": len(str(result)),
                "llm_evaluation": llm_result.get("evaluation", {})
            },
            "issues": issues,
            "recommendations": recommendations or ["Step completed successfully"],
            "goal_status": goal_status,
            "next_action": next_action
        } 


    # Rule Based Validation
    def _check_tool_availability(self, code: str, available_tools: List[str]) -> List[ValidationIssue]:
        """
        Check if all tools used in code are available.
        
        Args:
            code: Python code to check
            available_tools: List of available tool names
        
        Returns:
            List of issues for missing tools
        """
        issues = []
        
        # Extract function calls from code
        try:
            tree = ast.parse(code)
            called_functions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
            
            # Check if each function is available
            for func_name in called_functions:
                # Skip built-ins and common functions
                if func_name in ["print", "len", "str", "int", "float", "list", "dict"]:
                    continue
                
                if func_name not in available_tools:
                    issues.append(ValidationIssue(
                        severity="high",
                        type="missing_tool",
                        description=f"Tool '{func_name}' is not available",
                        suggestion=f"Use one of: {', '.join(available_tools[:5])}"
                    ))
        
        except SyntaxError:
            # Syntax error will be caught by _check_code_syntax
            pass
        
        return issues

    def _check_code_syntax(self, code: str) -> List[ValidationIssue]:
        """
        Validate Python code syntax.
        
        Args:
            code: Python code to validate
        
        Returns:
            List of syntax issues
        """
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity="high",
                type="syntax_error",
                description=f"Code syntax error: {e.msg} at line {e.lineno}",
                suggestion="Fix syntax before execution"
            ))
        
        return issues

    def _check_past_failures(
        self, 
        code: str, 
        description: str, 
        past_failures: List[Dict]
    ) -> List[ValidationIssue]:
        """
        Check if plan repeats past failures.
        
        Args:
            code: Proposed code
            description: Step description
            past_failures: List of past failure records
        
        Returns:
            List of issues for repeated failures
        """
        issues = []
        
        for failure in past_failures:
            failure_code = failure.get("code", "")
            failure_desc = failure.get("description", "")
            
            # Simple similarity check (can be improved with fuzzy matching)
            if failure_code and code:
                # Check if code is very similar (>80% match)
                similarity = self._simple_similarity(code, failure_code)
                if similarity > 0.8:
                    issues.append(ValidationIssue(
                        severity="medium",
                        type="repeated_failure",
                        description=f"Plan similar to past failure: {failure_desc or failure.get('error', 'unknown error')}",
                        suggestion="Try a different approach or tool"
                    ))
        
        return issues
    
    def _check_dangerous_operations(self, code: str) -> List[ValidationIssue]:
        """
        Check for dangerous operations in code.
        
        Args:
            code: Python code to check
        
        Returns:
            List of issues for dangerous operations
        """
        issues = []
        
        dangerous_keywords = ["eval", "exec", "compile", "__import__", "open", "file"]
        
        for keyword in dangerous_keywords:
            if keyword in code:
                issues.append(ValidationIssue(
                    severity="high",
                    type="dangerous_operation",
                    description=f"Code contains potentially dangerous operation: {keyword}",
                    suggestion="Remove dangerous operations or use safer alternatives"
                ))
        
        return issues

    def _simple_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple string similarity (0.0 to 1.0).
        
        Args:
            str1: First string
            str2: Second string
        
        Returns:
            Similarity score
        """
        # Simple token-based similarity
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0

    # LLM Based Validation
    async def _llm_validate_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to validate query clarity and intent.
        
        Args:
            query: User query
        
        Returns:
            Dict with LLM validation results
        """
        prompt = f"""Validate this user query:

Query: {query}

Check:
1. Is it clear and understandable?
2. Does it have a clear intent?
3. Is it answerable?

Return JSON:
{{
  "is_clear": true/false,
  "intent": "description",
  "issues": [],
  "recommendations": []
}}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            result = self._parse_json_response(response.text)
            return result
            
        except Exception as e:
            print(f"⚠️ LLM query validation failed: {e}")
            return {"issues": [], "recommendations": []}
    
    async def _llm_validate_plan(
        self, 
        plan: str, 
        code: str, 
        description: str
    ) -> Dict[str, Any]:
        """
        Use LLM to validate plan logic and feasibility.
        
        Args:
            plan: Plan text
            code: Code to execute
            description: Step description
        
        Returns:
            Dict with LLM validation results
        """
        prompt = f"""Validate this execution plan:

Plan: {plan}
Step: {description}
Code: {code}

Check:
1. Is the logic sound?
2. Will the code achieve the step goal?
3. Are there any logical flaws?

Return JSON:
{{
  "is_feasible": true/false,
  "logic_sound": true/false,
  "issues": [],
  "recommendations": []
}}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            result = self._parse_json_response(response.text)
            return result
            
        except Exception as e:
            print(f"⚠️ LLM plan validation failed: {e}")
            return {"issues": [], "recommendations": []}
    

    async def _llm_validate_result(
        self,
        result: str,
        error: str,
        step_description: str,
        original_goal: str
    ) -> Dict[str, Any]:
        """
        Use LLM to assess goal achievement and progress.
        
        Args:
            result: Execution result
            error: Error message (if any)
            step_description: What the step was supposed to do
            original_goal: Original user query/goal
        
        Returns:
            Dict with goal status and recommendations
        """
        prompt = f"""Evaluate this step result:

Original Goal: {original_goal}
Step: {step_description}
Result: {result}
Error: {error or "None"}

Assess:
1. Did this step succeed?
2. Is the original goal now achieved?
3. Is progress being made?

Return JSON:
{{
  "step_succeeded": true/false,
  "original_goal_achieved": true/false,
  "step_goal_achieved": true/false,
  "progress_score": 0.0-1.0,
  "reasoning": "explanation",
  "recommendations": []
}}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            result = self._parse_json_response(response.text)
            
            # Build GoalStatus
            if result.get("original_goal_achieved") is not None:
                goal_status = {
                    "original_goal_achieved": result.get("original_goal_achieved", False),
                    "step_goal_achieved": result.get("step_goal_achieved", False),
                    "progress_score": float(result.get("progress_score", 0.0)),
                    "reasoning": result.get("reasoning", "")
                }
                result["goal_status"] = goal_status
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM result validation failed: {e}")
            return {"issues": [], "recommendations": []}
    
    def _convert_llm_issues_to_validation_issues(self, llm_issues: List[Any]) -> List[ValidationIssue]:
        """
        Converts raw LLM issues (strings or dicts) into ValidationIssue objects.
        
        Args:
            llm_issues: List of issues from LLM (can be strings or dicts)
        
        Returns:
            List of ValidationIssue objects
        """
        converted_issues = []
        for issue in llm_issues:
            if isinstance(issue, str):
                # Simple string issue
                converted_issues.append(ValidationIssue(
                    severity="medium",
                    type="llm_feedback",
                    description=issue
                ))
            elif isinstance(issue, dict):
                # Dict with structured fields
                converted_issues.append(ValidationIssue(
                    severity=issue.get("severity", "medium"),
                    type=issue.get("type", "llm_feedback"),
                    description=issue.get("description", "LLM reported an issue."),
                    suggestion=issue.get("suggestion")
                ))
            elif isinstance(issue, ValidationIssue):
                # Already a ValidationIssue object
                converted_issues.append(issue)
        return converted_issues
    
    def _convert_llm_recommendations_to_strings(self, llm_recommendations: List[Any]) -> List[str]:
        """
        Converts raw LLM recommendations (strings or dicts) into string list.
        
        Args:
            llm_recommendations: List of recommendations from LLM (can be strings or dicts)
        
        Returns:
            List of string recommendations
        """
        converted_recommendations = []
        for rec in llm_recommendations:
            if isinstance(rec, str):
                converted_recommendations.append(rec)
            elif isinstance(rec, dict):
                # Extract recommendation text from dict
                rec_text = rec.get("recommendation", str(rec))
                converted_recommendations.append(rec_text)
            else:
                converted_recommendations.append(str(rec))
        return converted_recommendations

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Args:
            text: LLM response text
        
        Returns:
            Parsed JSON dict
        """
        try:
            # Try to extract JSON block
            if "```json" in text:
                json_block = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_block = text.split("```")[1].split("```")[0].strip()
            else:
                json_block = text.strip()
            
            return json.loads(json_block)
            
        except Exception:
            return {}
    
    def __repr__(self) -> str:
        """Debug representation."""
        mode = "hybrid" if self.use_llm else "rule-based"
        return (
            f"CriticAgent("
            f"mode={mode}, "
            f"strict={self.strict_mode}, "
            f"calls={self.metrics['total_calls']})"
        )