from pydantic import BaseModel,Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

# base classes
class AgentRequest(BaseModel):
    """Base class for all agent requests"""

    #Identifiers
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    #Tracing
    context_id: str #for execution_context

    #configutation
    class Config:
        arbitrary_types_allowed = True
        json_encoders ={
            datetime:lambda v: v.isoformat()
        }

class AgentResponse(BaseModel):
    """Base class for all agent responses"""

    #Identifiers
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    #performance_tracking
    processing_time: float

    #status
    success: bool
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders ={
            datetime:lambda v: v.isoformat()
        }

# retriever agent models
class RetrievalRequest(AgentRequest):
    """Request to retrieve information from sources"""

    query:str

    sources: List[str] = Field(default=["memory"])
    strategy: str = Field(default="parallel")
    top_k: int = Field(default=3)

class MemoryResult(BaseModel):
    """Single Memory Result"""
    source: str
    file_path: str
    session_id: str
    query: str
    solution_summary: str
    result_requirement: str
    relevance_score: float
    retrieved_at: datetime = Field(default_factory=datetime.now)

class RetrievalResponse(AgentResponse):
    """Response from Retriever Agent"""
    
    results: List[MemoryResult]
    sources_tried: List[str]
    sources_succeeded: List[str]
    total_results: int

    retrieval_time_by_source: Dict[str, float]

# perception agent models
class AnalysisMode(str, Enum):
    """Mode of perception analysis"""
    QUERY_ANALYSIS = "query_analysis"
    RESULT_ANALYSIS = "result_analysis"

class PerceptionRequest(AgentRequest):
    analysis_mode: AnalysisMode
    content: str
    retrieved_context: Optional[List[MemoryResult]]
    current_plan: Optional[str]=None
    step_description: Optional[str]=None
    original_goal: Optional[str]=None

class PerceptionResponse(AgentResponse):
    """Response from Perception Agent"""
    entities: List[str]
    requirements: Dict[str, Any]
    reasoning: str
    intent: str
    complexity: str
    confidence: float
    extracted_data: Optional[Dict[str, Any]] = None
    quality_assessment: Optional[str] = None

# critic agent models
class CriticMode(str, Enum):
    """Mode of critic analysis"""
    QUERY_VALIDATION = "query_validation"
    PLAN_VALIDATION = "plan_validation"
    RESULT_VALIDATION = "result_validation"

class CriticRequest(AgentRequest):
    """Request to critique a query or plan and validate a result"""
    critic_mode: CriticMode
    subject: Dict[str, Any] #query, plan, result, etc.

    #context for validation
    available_tools: Optional[List[str]] = None
    past_failures: Optional[List[Dict]] = None
    original_goal: Optional[str]=None
    current_plan: Optional[str]=None

    #validation criteria
    check_feasibility: bool = True
    check_tool_availability: bool = True
    check_goal_achievement: bool = False

class ValidationIssue(BaseModel):
    """A single validation issue"""
    severity: str #high, medium, low
    type: str #missing tool, logic error, timeout risk etc
    description: str
    suggestion: Optional[str]=None

class GoalStatus(BaseModel):
    """Goal achievement status"""
    original_goal_achieved: bool
    step_goal_achieved: bool
    progress_score: float #0.0 - 1.0
    reasoning: str

class CriticResponse(AgentResponse):
    """Response from Critic Agent"""

    #validation results
    approved: bool
    confidence: float

    #evaluation details
    evaluation: Dict[str, Any]

    #issues found
    issues: List[ValidationIssue]

    #recommentations
    recommendations: List[str]

    #goal_status
    goal_status: Optional[GoalStatus] = None

    #next step
    next_action: Optional[str]=None

#decision agent models
class DecisionMode(str, Enum):
    """Mode of decision making"""
    INITIAL_PLAN="initial_plan"
    NEXT_STEP="next_step"
    REPLAN="replan"

class DecisionRequest(AgentRequest):
    """Request to make a plan or decision"""
    mode: DecisionMode
    query: str

    #Analysis results
    perception_analysis: Optional[PerceptionResponse]=None
    critic_feedback: Optional[CriticResponse]=None

    #Retrieved Data
    retrieved_data: Optional[List[MemoryResult]]=None

    #Current state for replan/next step
    current_plan: Optional[List[str]]=None
    completed_steps: Optional[List[Dict]] = None
    failed_steps: Optional[Dict] = None

    #config
    strategy: str = Field(default="exploratory")
    max_steps: int = Field(default=3)

class StepType(str, Enum):
    """Type of execution step"""
    CODE="code"
    CONCLUDE="conclude"
    NOP="nop"

class ActionStep(BaseModel):
    """A single execution step"""
    step_index: int
    description: str
    type: StepType
    code: Optional[str]=None
    conclusion: Optional[str]=None
    rationale: str

class DecisionResponse(AgentResponse):
    """Response from Decision Agent"""

    plan_outline: List[str] #natural language steps
    next_step: ActionStep

    #planning metadata
    strategy_used: str
    estimated_complexity: str
    tools_required: List[str]

    #reasoning
    planning_rationale: str

#executor agent models
class ExecutionRequest(AgentRequest):
    """Request to execute a plan"""

    #code to execute
    code:str

    timeout:Optional[int]=None

    #context
    step_description: str
    step_index: int
    previous_results: Optional[List[Dict[str, Any]]] = None

class ExecutionResponse(AgentResponse):
    """Response from Executor Agent"""

    #Result
    status: str #success, error
    result: Optional[str]=None
    error: Optional[str]=None

    #timing
    execution_time: str #ISO timestamp
    total_time: str #duration in seconds

    #metadata
    function_count: Optional[int]=None
    tools_called: Optional[List[str]]=None

#session models
class StepModel(BaseModel):
    """Represents a single execution step"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    index : int

    #context
    description: str
    type: StepType
    code: Optional[str]=None
    conclusion: Optional[str]=None

    #execution
    status: str = Field(default="pending") #pending, completed, failed, executing
    execution_request: Optional[ExecutionRequest]=None
    execution_response: Optional[ExecutionResponse]=None

    #analysis
    perception_analysis: Optional[PerceptionResponse]=None
    critic_validation: Optional[CriticResponse]=None

    #metadata
    attempts: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime]=None

    #lineage
    parent_step_id: Optional[str]=None
    was_replanned: bool = Field(default=False)

class PlanModel(BaseModel):
    """Represents an execution plan"""

    #identity
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int 

    #context
    plan_outline: List[str]
    steps: List[StepModel]

    #metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="decision_agent")
    strategy: str

    #context at creation
    perception_snapshot: Optional[PerceptionResponse]=None
    critic_feedback: Optional[CriticResponse]=None

    #outcome
    is_active: bool = Field(default=True)
    superseded_by: Optional[str]=None

class SessionModel(BaseModel):
    """Complete session state"""

    #identity
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    #context
    original_query: str

    #timeline
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime]=None

    #initial_analysis
    initial_perception: Optional[PerceptionResponse]=None
    initial_critic: Optional[CriticResponse]=None

    #Retrived_memories
    retrieved_memories: List[MemoryResult] = Field(default_factory=list)

    #planning history
    plan_history: List[PlanModel] = Field(default_factory=list)
    current_plan: Optional[PlanModel]=None

    #execution 
    executed_steps: List[StepModel] = Field(default_factory=list)
    failed_steps: List[StepModel] = Field(default_factory=list)

    #outcome
    status: str = Field(default="active") #active, completed, failed
    final_answer: Optional[str]=None
    solution_summary: Optional[str]=None
    confidence: Optional[float]=None

    #metadata
    total_steps_executed: int = Field(default=0)
    total_replans: int = Field(default=0)
    total_time: Optional[float] = None

    def to_storage_format(self) -> Dict:
        """Convert to format for JSON storage"""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_storage(cls, data: Dict) -> "SessionModel":
        """Load from storage JSON"""
        return cls(**data)

#workflow models
class StateTransition(BaseModel):
    """Represents a state transition in the workflow"""

    #transtion
    from_state: str
    to_state: str

    #context
    reason: str
    timestamp: datetime = Field(default_factory=datetime.now)

    #metadata
    context_snapshot: Optional[Dict]=None

#configuration models
class AgentConfig(BaseModel):
    """Configuration for the agent"""

    #identity
    agent_name: str
    agent_type: str

    #LLM config
    llm_model: Optional[str]=None
    llm_temperature: Optional[float]=None

    #prompts
    prompt_path: Optional[str]=None

    #limits
    timeout: Optional[int]=None
    max_retries: Optional[int]=None

    #Features
    features: Dict[str, Any] = Field(default_factory=dict)

class CoordinatorConfig(BaseModel):
    """Configuration of the coordinator"""

    #strategy
    planning_strategy: str = Field(default="exploratory")
    exploration_mode: str = Field(default="parallel")

    #limits
    max_steps: int = Field(default=3)
    max_retries_per_step: int = Field(default=3)
    session_timeout: int = Field(default=3600)

    #memory
    memory_enabled: bool = Field(default=True)
    memory_path: str = Field(default="memory/session_logs")

    #agents
    agent_configs: Dict[str, AgentConfig] = Field(default_factory=dict)

    #mcp
    mcp_servers: List[str] = Field(default_factory=list)