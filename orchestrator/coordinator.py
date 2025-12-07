"""
AgentCoordinator: Orchestrates all agents to process user queries.

Coordinates agents via WorkflowEngine.
"""
from multiprocessing import context
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from core import (
    ExecutionContext,
    RetrievalRequest,
    PerceptionRequest,
    PerceptionResponse,
    CriticRequest,
    DecisionRequest,
    DecisionMode,
    ExecutionRequest,
    AnalysisMode,
    CriticMode,
    PlanModel,
    StepModel,
    SessionModel,
    MemoryResult
)

from agents.memory_agent import MemoryRequest

from agents import (
    ExecutorAgent,
    RetrieverAgent,
    MemoryAgent,
    PerceptionAgent,
    CriticAgent,
    DecisionAgent
)

from orchestrator import WorkflowEngine

class AgentCoordinator:
    """
    AgentCoordinator: Manages all agents and orchestrates query processing.
    
    Responsibilities:
    1. Initialize all agents
    2. Create ExecutionContext for each query
    3. Delegate execution to WorkflowEngine
    4. Provide agent wrapper methods for state handlers
    5. Save session to memory
    
    Usage:
        coordinator = AgentCoordinator(
            retriever=retriever,
            perception=perception,
            critic=critic,
            decision=decision,
            executor=executor,
            memory=memory
        )
        
        session = await coordinator.handle_query("What is 2+2?")
        print(session.final_answer)
    """

    def __init__(
        self,
        retriever: RetrieverAgent,
        perception: PerceptionAgent,
        critic: CriticAgent,
        decision: DecisionAgent,
        executor: ExecutorAgent,
        memory: MemoryAgent,
        config: Dict[str, Any]
    ):
        """
        Initialize coordinator with all agents.
        
        Args:
            retriever: RetrieverAgent instance
            perception: PerceptionAgent instance
            critic: CriticAgent instance
            decision: DecisionAgent instance
            executor: ExecutorAgent instance
            memory: MemoryAgent instance
            config: Configuration dict (from profiles.yaml)
        """
        self.retriever = retriever
        self.perception = perception
        self.critic = critic
        self.decision = decision
        self.executor = executor
        self.memory = memory
        
        self.config = config
        self.workflow_engine = WorkflowEngine()

    async def handle_query(self, query: str) -> SessionModel:
        """
        Main entry point: process user query end to end.

        Args:
            query: User query string
        
        Returns:
            SessionModel with execution results
        """

        session_id = str(uuid.uuid4())
        context = ExecutionContext(
            session_id=session_id,
            original_query=query,
            config=self.config
        )

        final_state = await self.workflow_engine.execute(context, self)

        session = context.to_session_model()

        return session
    
    # Agent Wrapper Methods
    async def retrieve(
        self,
        query: str,
        context_id: str,
        top_k: int = 3
    ) -> List[MemoryResult]:
        """
        Retrieve relevant memories.
        
        Args:
            query: User query string
            context_id: Execution context ID
            top_k: Number of results
        """
        request = RetrievalRequest(
            query=query,
            sources=["memory"],
            top_k=top_k,
            context_id=context_id
        )

        response = await self.retriever.execute(request)
        return response.results
    
    async def analyze_query(
        self,
        query: str,
        memories: List[MemoryResult],
        context_id: str
    ) -> PerceptionResponse:
        """
        Analyze user query using PerceptionAgent.
        
        Args:
            query: User query
            memories: Retrieved memories
            context_id: Execution context ID
        
        Returns:
            PerceptionResponse with analysis
        """
        request = PerceptionRequest(
            analysis_mode=AnalysisMode.QUERY_ANALYSIS,
            content=query,
            retrieved_context=memories,
            context_id=context_id
        )
        
        response = await self.perception.execute(request)
        return response
    
    async def analyze_result(
        self,
        result: str,
        step_description: str,
        current_plan: List[str],
        context_id: str
    ) -> PerceptionResponse:
        """
        Analyze execution result using PerceptionAgent.
        
        Args:
            result: Execution result
            step_description: Description of the step
            current_plan: Current plan outline
            context_id: Execution context ID
        
        Returns:
            PerceptionResponse with analysis
        """
        request = PerceptionRequest(
            analysis_mode=AnalysisMode.RESULT_ANALYSIS,
            content=result,
            retrieved_context=[],
            step_description=step_description,
            current_plan="\n".join(current_plan) if current_plan else "",
            context_id=context_id
        )
        
        response = await self.perception.execute(request)
        return response

    async def validate_query(
        self,
        query: str,
        context_id: str
    ):
        """
        Validate user query using CriticAgent.
        
        Args:
            query: User query
            context_id: Execution context ID
        
        Returns:
            CriticResponse with validation results
        """
        request = CriticRequest(
            critic_mode=CriticMode.QUERY_VALIDATION,
            subject={"query": query},
            context_id=context_id
        )
        
        response = await self.critic.execute(request)
        return response
    
    async def validate_plan(
        self,
        plan: List[str],
        step: StepModel,
        past_failures: List[Dict],
        context_id: str
    ):
        """
        Validate execution plan using CriticAgent.
        
        Args:
            plan: Plan outline (list of strings)
            step: Next step to execute
            past_failures: List of past failure records
            context_id: Execution context ID
        
        Returns:
            CriticResponse with validation results
        """

        available_tools = self.executor.get_available_tools() if hasattr(self.executor, 'get_available_tools') else []

        request = CriticRequest(
            critic_mode=CriticMode.PLAN_VALIDATION,
            subject={
                "plan": "\n".join(plan),
                "code": step.code or "",
                "description": step.description
            },
            available_tools=available_tools,
            past_failures=past_failures,
            check_tool_availability=True,
            check_feasibility=True,
            context_id=context_id
        )

        response = await self.critic.execute(request)
        return response
    
    async def validate_result(
        self,
        result: str,
        error: str,
        step_description: str,
        original_goal: str,
        context_id: str
    ):
        """
        Validate execution result using CriticAgent.
        
        Args:
            result: Execution result
            error: Error message (if any)
            step_description: Description of executed step
            original_goal: Original user query
            context_id: Execution context ID
        
        Returns:
            CriticResponse with validation and goal status
        """
        request = CriticRequest(
            critic_mode=CriticMode.RESULT_VALIDATION,
            subject={
                "result": result,
                "error": error,
                "description": step_description
            },
            original_goal=original_goal,
            check_goal_achievement=True,
            context_id=context_id
        )
        
        response = await self.critic.execute(request)
        return response
    
    async def create_plan(
        self,
        query: str,
        perception: Optional[PerceptionResponse],
        current_plan: List[str],
        completed_steps: List[StepModel],
        mode: str,
        context_id: str
    ) -> PlanModel:
        """
        Create or revise execution plan using DecisionAgent.
        
        Args:
            query: User query
            perception: Perception analysis
            current_plan: Current plan outline
            completed_steps: Steps completed so far
            mode: "initial" or "replan"
            context_id: Execution context ID
        
        Returns:
            PlanModel with plan and next step
        """
        # Determine decision mode
        if mode == "initial":
            decision_mode = DecisionMode.INITIAL_PLAN
        elif mode == "next_step":
            decision_mode = DecisionMode.NEXT_STEP
        else:  # replan
            decision_mode = DecisionMode.REPLAN
        
        request = DecisionRequest(
            mode=decision_mode,
            query=query,
            perception_analysis=perception,
            current_plan=current_plan.plan_outline if current_plan else None,
            completed_steps=[s.model_dump() for s in completed_steps] if completed_steps else [],
            strategy=self.config.get("planning_strategy", "exploratory"),
            max_steps=self.config.get("max_steps", 10),
            context_id=context_id
        )
        
        response = await self.decision.execute(request)
        
        # Convert DecisionResponse to PlanModel
        # Calculate version: 
        # - initial plan = 1
        # - next_step = keep same version (just adding next step to existing plan)
        # - replan = increment version (actual replanning due to failure)
        if mode == "initial":
            version = 1
        elif mode == "next_step":
            # Keep the same version when just generating the next step
            version = current_plan.version if current_plan else 1
        else:  # replan
            # Increment version for actual replanning
            version = (current_plan.version + 1) if current_plan else 1
            
        # Build the steps list - preserve completed steps and add new step
        all_steps = []
        
        # Add all completed steps (preserve their state)
        for completed_step in completed_steps:
            all_steps.append(completed_step)
        
        # Add the new step
        new_step = StepModel(
            index=response.next_step.step_index,
            description=response.next_step.description,
            type=response.next_step.type,
            code=response.next_step.code,
            conclusion=response.next_step.conclusion,
            status="pending"
        )
        all_steps.append(new_step)
        
        # Sort steps by index to ensure proper order
        all_steps.sort(key=lambda x: x.index)
        
        plan = PlanModel(
            version=version,
            plan_outline=response.plan_outline,
            steps=all_steps,
            strategy=response.strategy_used,
            perception_snapshot=perception
        )
        
        return plan

    async def execute_step(
        self,
        code: str,
        step_description: str,
        step_index: int,
        context_id: str,
        previous_results: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Execute code using ExecutorAgent.
        
        Args:
            code: Python code to execute
            step_description: Description of the step
            step_index: Step index
            context_id: Execution context ID
        
        Returns:
            ExecutionResponse with result or error
        """
        request = ExecutionRequest(
            code=code,
            step_description=step_description,
            step_index=step_index,
            context_id=context_id,
            previous_results=previous_results
        )
        
        response = await self.executor.execute(request)
        return response

    async def save_session(self, context: ExecutionContext):
        """
        Save session to memory using MemoryAgent.
        
        Args:
            context: ExecutionContext to save
        """
        session = context.to_session_model()
        
        request = MemoryRequest(
            operation="save",
            session_data=session,
            context_id=context.context_id
        )
        
        response = await self.memory.execute(request)
        
        if response.success:
            print(f"💾 Session saved: {response.file_path}\n")
        else:
            print(f"⚠️  Failed to save session: {response.error_message}\n")
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"AgentCoordinator("
            f"retriever={self.retriever.agent_name}, "
            f"perception={self.perception.agent_name}, "
            f"critic={self.critic.agent_name}, "
            f"decision={self.decision.agent_name}, "
            f"executor={self.executor.agent_name}, "
            f"memory={self.memory.agent_name})"
        )