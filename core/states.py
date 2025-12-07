"""
Workflow State Definitions

Defines all possible states in the agent execution workflow and transitions.
Used by the WorkflowEngine to manage execution flow as a state machine.
"""

from enum import Enum
from typing import Optional, Set
from datetime import datetime
from pydantic import BaseModel

class WorkflowState(Enum):
    """
    All possible states in the agent execution workflow.

    Each state represents a distinct phase of query processing. 
    The workflowEngine uses these to determine what to do next. 

    FLOW OVERVIEW:
       INIT -> RETRIEVE -> ANALYZE QUERY -> VALIDATE QuERY -> 
       PLAN -> VALIDATE PLAN -> EXECUTE STEP -> ANALYZE RESULT -> 
       VALIDATE RESULT -> DECIDE NEXT -> (loop or COMPLETE/FAILED)
    """

    #Initialize
    INIT = "init"
    """
    Initial State when quer is received.
    Sets up ExecutionCOntext, prepares for retrieval.

    Next: RETRIEVE
    """

    #Retrieve Phase
    RETRIEVE = "retrieve"
    """
    Search for relevant memories from past sessions.

    Agent: RetrieverAgent
    Next: ANALYZE_QUERY
    """

    #Query Phase
    ANALYZE_QUERY = "analyze_query"
    """
    Understand the users query using the PerceptionAgent.
    Extract entities, intent, requirements.

    Agent: PerceptionAgent (mode=QUeRY_ANALYSIS)
    Next: VALIDATE_QUERY
    """

    VALIDATE_QUERY = "validate_query"
    """
    Validate if query is clear and answerable using CriticAgent.

    Agent: CriticAgent(mode=query_validation)
    Next:
        -COMPLETE (if answer found in memory)
        -PLAN (if need to execute steps)
        -FAILED (if query is unclear or unanswerable)
    """

    #Planning phase
    PLAN = "plan"
    """
    Create a plan or revise execution plan using the DecisionAgent.

    Agent: DecisionAgent(mode=initial_plan or replan)
    Next: VALIDATE_PLAN
    """

    VALIDATE_PLAN = "validate_plan"
    """
    Validate plan before exeution

    Agent: CriticAgent(mode=plan_validation)
    Next:
        -EXECUTE_STEP (if plan approved)
        -PLAN (if plan rejected)
    """

    #Execution phase
    EXECUTE_STEP = "execute_step"
    """
    Execute current step's code using ExecutorAgent.
    Agent: ExecutorAgent
    Next: ANALYZE_RESULT
    """

    ANALYZE_RESULT = "analyze_result"
    """
    Analyze the execution result using PerceptionAgent.
    Extract what was learned, understand output in RESULT_ANALYSIS mode.

    Agent: PerceptionAgent(mode=RESULT_ANALYSIS)
    Next: VALIDATE_RESULT
    """

    VALIDATE_RESULT = "validate_result"
    """
    Validate if step succeeded and check goal achievement.

    Agent: CriticAgent(mode=result_validation)
    Next:
        - COMPLETE (If original_goal_achieved)
        - DECIDE_NEXT (If Step succeeded but original goal not achieved
        - PLAN (if step failed or need to replan)
    """

    #Decision phase
    DECIDE_NEXT = "decide_next"
    """
    Decide what to do after successful step.
    Check if more steps in plan, or if done.

    Next:
        - EXECUTE_STEP (if more steps in plan)
        - COMPLETE (if all steps done)
    """

    #Terminal states
    COMPLETE = "complete"
    """
    Terminal state when query is successfully answered.

    No next state. 
    """

    FAILED = "failed"
    """
    Terminal state when queery failed (max retries exceeded or unrecoverable error)

    No next state.
    """

    def is_terminal(self) -> bool:
        """
        Check if it is a terminal state.
        """
        return self in [WorkflowState.COMPLETE, WorkflowState.FAILED]
    
    def is_execution_step(self) -> bool:
        """
        Check if it is an execution step.
        """
        return self in [
            WorkflowState.EXECUTE_STEP,
            WorkflowState.ANALYZE_RESULT,
            WorkflowState.VALIDATE_RESULT
        ]
    
    def is_planning_step(self) -> bool:
        """
        Check if it is a planning step.
        """
        return self in [
            WorkflowState.PLAN,
            WorkflowState.VALIDATE_PLAN,
            WorkflowState.DECIDE_NEXT
        ]
    
    def get_allowed_transitions(self) -> Set["WorkflowState"]:
        """
        Get valid next states from current state.
        Enforces state machine rules.

        Returns:
            Set of allowed next states.
        """

        transitions = {
            WorkflowState.INIT: {WorkflowState.RETRIEVE},
            WorkflowState.RETRIEVE: {WorkflowState.ANALYZE_QUERY},
            WorkflowState.ANALYZE_QUERY: {WorkflowState.VALIDATE_QUERY},
            WorkflowState.VALIDATE_QUERY: {
                WorkflowState.COMPLETE,
                WorkflowState.PLAN,
                WorkflowState.FAILED
            },
            WorkflowState.PLAN: {WorkflowState.VALIDATE_PLAN},
            WorkflowState.VALIDATE_PLAN: {
                WorkflowState.EXECUTE_STEP,
                WorkflowState.PLAN,
                WorkflowState.FAILED
            },
            WorkflowState.EXECUTE_STEP : {WorkflowState.ANALYZE_RESULT},
            WorkflowState.ANALYZE_RESULT: {WorkflowState.VALIDATE_RESULT},
            WorkflowState.VALIDATE_RESULT: {
                WorkflowState.COMPLETE,
                WorkflowState.DECIDE_NEXT,
                WorkflowState.PLAN
            },
            WorkflowState.DECIDE_NEXT:{
                WorkflowState.EXECUTE_STEP,
                WorkflowState.COMPLETE,
                WorkflowState.PLAN  # Allow replanning when plan exhausted
            },
            WorkflowState.COMPLETE: set(),
            WorkflowState.FAILED: set()

        }

        return transitions.get(self, set())

    
    def can_transition_to(self, next_state: "WorkflowState") -> bool:
        """
        Check if transition to next step is permissible.

        Args:
            next_state: state to transition to

        Returns:
            True if transition is allowed.
        """
        return next_state in self.get_allowed_transitions()
    
    def __str__(self) -> str:
        """String representation for logging."""
        return self.value
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"WorkflowState.{self.name}"

class StateTransitionError(Exception):
    """
    Raised when an invalid state transition is attempted.
    """
    def __init__(self, from_state: WorkflowState, to_state: WorkflowState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition from {from_state.value} to {to_state.value}. "
            f"Allowed transitions: {[s.value for s in from_state.get_allowed_transitions()]}"
        )

def validate_transition(
    from_state: WorkflowState,
    to_state: WorkflowState,
    raise_error: bool = True
) -> bool:
    """
    Validate if a state transition is allowed.
    
    Args:
        from_state: Current state
        to_state: Desired next state
        raise_error: If True, raise StateTransitionError on invalid transition
        
    Returns:
        True if transition is valid
        
    Raises:
        StateTransitionError: If transition is invalid and raise_error=True
    """
    is_valid = from_state.can_transition_to(to_state)
    
    if not is_valid and raise_error:
        raise StateTransitionError(from_state, to_state)
    
    return is_valid