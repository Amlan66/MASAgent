# core/__init__.py

"""
Core module: Data models, context, and state definitions.
"""

from core.models import (
    AgentRequest,
    AgentResponse,
    RetrievalRequest,
    RetrievalResponse,
    MemoryResult,
    PerceptionRequest,
    PerceptionResponse,
    CriticRequest,
    CriticResponse,
    ValidationIssue,
    GoalStatus,
    DecisionRequest,
    DecisionResponse,
    ExecutionRequest,
    ExecutionResponse,
    SessionModel,
    StepModel,
    PlanModel,
    ActionStep,
    StateTransition,
    AgentConfig,
    CoordinatorConfig,
    AnalysisMode,
    CriticMode,
    DecisionMode,
    StepType
)

from core.context import ExecutionContext
from core.states import WorkflowState, StateTransitionError, validate_transition

__all__ = [
    # Base Models
    "AgentRequest",
    "AgentResponse",
    # Retriever Models
    "RetrievalRequest",
    "RetrievalResponse",
    "MemoryResult",
    # Perception Models
    "PerceptionRequest",
    "PerceptionResponse",
    "AnalysisMode",
    # Critic Models
    "CriticRequest",
    "CriticResponse",
    "ValidationIssue",
    "GoalStatus",
    "CriticMode",
    # Decision Models
    "DecisionRequest",
    "DecisionResponse",
    "ActionStep",
    "DecisionMode",
    # Executor Models
    "ExecutionRequest",
    "ExecutionResponse",
    # Session Models
    "SessionModel",
    "StepModel",
    "PlanModel",
    "StepType",
    # Workflow Models
    "StateTransition",
    # Config Models
    "AgentConfig",
    "CoordinatorConfig",
    # Context
    "ExecutionContext",
    # States
    "WorkflowState",
    "StateTransitionError",
    "validate_transition"
]

__version__ = "1.0.0"