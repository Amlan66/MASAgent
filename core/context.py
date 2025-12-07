"""
ExecutionContext: Shared state container for agent execution lifecycle

All agents and state handlers can read/write to this shared context. 
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from core.models import (
    MemoryResult,
    PerceptionResponse,
    PlanModel,
    CriticResponse,
    StepModel,
    SessionModel,
    StateTransition
)

class ExecutionContext:
    """
    Centralized state container for a single query execution.
    This is the sesion workspace that travels through the workflow.
    Each state handlers reads from and writes to this context.
    """

    def __init__(
        self,
        session_id: str,
        original_query: str,
        config: Dict[str, Any]
    ):
        """
        Initialize a new context

        Args:
            sessio_id: Unique Id for the execution session
            orioginal_query: The original user query text
            config: configurations loaded via profiles.yaml
        """

        self.session_id = session_id
        self.context_id = str(uuid.uuid4())
        
        self.original_query = original_query

        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

        self.config = config

        #initialize memory and analysis resutls
        self.retreieved_memories: List[MemoryResult] = []
        self.query_analysis: Optional[PerceptionResponse] = None
        self.query_validation: Optional[CriticResponse] = None

        #initialize planning and executioin
        self.current_plan: Optional[PlanModel] = None
        self.plan_history: List[PlanModel] = []
        self.executed_steps: List[StepModel] = []
        self.current_step: Optional[StepModel] = None
        self.current_step_index: int = 0
        self.step_failures: List[Dict[str, Any]] = []
        self._max_failures = config.get("max_failure_history", 3)

        #workflow state
        self.state_transitions: List[StateTransition] = []
        self.current_workflow_state: Optional[str] = None

        #outcomes
        self.status: str = "active" #active, completed, failed
        self.final_answer: Optional[str] = None
        self.solution_summary: Optional[str] = None
        self.confidence: Optional[float] = None
        self.error_message: Optional[str] = None

        #metrics
        self.total_steps_attempted: int = 0
        self.total_replans: int = 0
        self.total_time: Optional[float] = None

    #Step Management methods
    def add_step(self, step: StepModel) -> None:
        """
        Add a completed or failed step to the executed steps list
        Called after every step execution
        Args:
            step: Step with execution results
        """

        self.executed_steps.append(step)
        self.total_steps_attempted += 1

        if step.status == "failed":
            self.add_failure(step)
        
    def add_failure(self, step: StepModel) -> None:
        """
        Track a step failure to avoid repeating mistakes
        Keeps N failures (configurable)
        Args:
            step: Step that failed
        """
        failure_record = {
            "step_index": step.index,
            "description": step.description,
            "error": step.execution_response.error if step.execution_response else "Unknown error",
            "code": step.code,
            "timestamp":datetime.now()
        }
        self.step_failures.append(failure_record)

        #keep only recent N failures
        if len(self.step_failures) > self._max_failures:
            self.step_failures.pop(0)

    def get_recent_failures(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get recent failures to avoid suggesting failed approaches
        Used by Critic Agent
        Args:
            limit: Max number of failures to return (default:all)
        Returns:
            List of failure records
        """

        if limit is None:
            return self.step_failures.copy()
        return self.step_failures.copy()[-limit:]

    def transition_to_step(self, step_index: int) -> None:
        """
        Move to executing a specific step
        Updates current_step_index and current_step
        Args:
            step_index: Index of the step to execute
        """

        self.current_step_index = step_index
        if self.current_plan:
            for step in self.current_plan.steps:
                if step.index == step_index:
                    self.current_step = step
                    break
    
    #plan management methods
    def add_plan(self, plan: PlanModel, is_replan: bool = True) -> None:
        """
        Add a new plan to plan history
        Args:
            plan: New plan to add
            is_replan: Whether this is an actual replan (vs next step generation)
        """

        if self.current_plan:
            self.current_plan.is_active = False
            self.current_plan.superseded_by = plan.plan_id
            if is_replan:
                self.total_replans += 1
        
        self.plan_history.append(plan)
        self.current_plan = plan
    
    def get_current_plan(self) -> List[str]:
        """
        Get natural language description of the current plan
        Returns:
            List of step descriptions (["Step 0: Search Web", "Step 1: Extract price"])
        """

        if self.current_plan:
            return self.current_plan.plan_outline
        
        return []
    
    def get_completed_steps(self) -> List[StepModel]:
        """
        Get all completed steps 
        Used when replanning is done to check whats already done

        Returns:   
            List of completed steps 
        """

        return [step for step in self.executed_steps if step.status == "completed"]

    
    #worflow state
    def add_transition(self, from_state:str, to_state:str, reason: str = "") -> None:
        """
        Record a state transition in the workflow
        Used by WorkflowEngine to tract execution execution flow.

        Helpful for debugging : Shows what the agent has tried and why it changed states, like when did we go from 
        PLAN to EXECUTE_STEP

        Args:
            from_state: Previous workflow state
            to_state: New workflow state
            reason: Why the transition happened (optional)
        """

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=datetime.now()
        )
        self.state_transitions.append(transition)
        self.current_workflow_state = to_state

    # completion checks
    def is_complete(self) -> bool:
        """
        Check if execution is complete(success or failure)
        Returns:
            True if execution should stop
        """

        return self.status in ["completed", "failed"]
    
    def mark_complete(
        self,
        final_answer: str,
        solution_summary: str,
        confidence: float,
    ) -> None:
        """
        Mark execution as successfully completed
        Called when Critic Agent confirms the goal is achieved.
        Args:
            final_answer: The answer to the user query
            solution_summary: How we go to the answer
            confidence: 0.0-1.0
        """
        self.status="completed"
        self.final_answer = final_answer
        self.solution_summary = solution_summary
        self.confidence = confidence
        self.completed_at = datetime.now()

        if self.created_at and self.completed_at:
            self.total_time = (self.completed_at - self.created_at).total_seconds()
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark execution as failed
        Called when max retries exceeded or unrecoverable error

        Args:
            error_message: Why execution failed
        """
        self.status="failed"
        self.error_message = error_message
        self.completed_at = datetime.now()

        if self.created_at and self.completed_at:
            self.total_time = (self.completed_at - self.created_at).total_seconds()
    
    # serialization
    def to_session_model(self) -> SessionModel:
        """
        Convert context to SessionModel for storage
        THis is called at the end to save the session to JSON

        Returns:
            SessionModel ready for JSON serialization
        """

        return SessionModel(
            session_id=self.session_id,
            original_query=self.original_query,
            created_at=self.created_at,
            completed_at=self.completed_at,
            initial_perception=self.query_analysis,
            initial_critique=self.query_validation,
            retrieved_memories=self.retreieved_memories,
            plan_history=self.plan_history,
            current_plan=self.current_plan,
            executed_steps=self.executed_steps,
            step_failures=self.step_failures,
            status=self.status,
            final_answer=self.final_answer,
            solution_summary=self.solution_summary,
            confidence=self.confidence,
            total_steps_executed=self.total_steps_attempted,
            total_replans=self.total_replans,
            total_time=self.total_time
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dicitonary for debugging or logging
        Returns:
            Dictionary representation of the context
        """
        return{
            "session_id": self.session_id,
            "context_id": self.context_id,
            "query": self.original_query,
            "status": self.status,
            "current_state": self.current_workflow_state,
            "steps_attempted": self.total_steps_attempted,
            "replans": self.total_replans,
            "has_plan": self.current_plan is not None,
            "recent_failures":len(self.step_failures)
        }
    
    def __repr__(self) -> str:
        """
        String representation for debugging
        Returns:
            String with basic info
        """
        return(
            f"ExecutionContext(session_id={self.session_id[:8]}..., "
            f"query='{self.original_query[:50]}...', "
            f"status={self.status}, "
            f"steps={self.total_steps_attempted})"
        ) 
