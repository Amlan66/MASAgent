"""
WorkflowEngine: State machine for agent execution flow.

Each state represents a distinct phase of query processing.
"""
from typing import Dict, Any, Optional
from datetime import datetime

from core import (
    WorkflowState,
    StateTransitionError,
    validate_transition,
    ExecutionContext,
    StepModel
)

class StateHandler:
    """
    Base class for state handlers.
    Each state handler processes one phase of the workflow.
    """

    def __init__(self, state: WorkflowState):
        self.state = state
    
    async def process(
        self, 
        context: ExecutionContext, 
        coordinator
    ) -> WorkflowState:
        """
        Process this state and return next state.
        
        Args:
            context: ExecutionContext with all runtime state
            coordinator: AgentCoordinator for calling agents
        
        Returns:
            Next WorkflowState to transition to
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement process()")
    
    def transition(
        self, 
        context: ExecutionContext, 
        to_state: WorkflowState, 
        reason: str = ""
    ) -> WorkflowState:
        """
        Validate and record state transition.
        
        Args:
            context: ExecutionContext
            to_state: Target state
            reason: Why transitioning
        
        Returns:
            Next state (to_state)
        
        Raises:
            StateTransitionError: If transition is invalid
        """
        validate_transition(self.state, to_state, raise_error=True)
        context.add_transition(self.state.value, to_state.value, reason)
        return to_state

#STATE HANDLERS
class InitStateHandler(StateHandler):
    """Initialize the execution context and prepare for processing."""

    def __init__(self):
        super().__init__(WorkflowState.INIT)

    async def process(
        self, 
        context: ExecutionContext, 
        coordinator
    ) -> WorkflowState:
        """
        Initialize context.

        Next: RETRIEVE
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Session: {context.session_id[:8]}")
        print(f"📝 Query: {context.original_query}")
        print(f"{'='*60}\n")
        
        return self.transition(context, WorkflowState.RETRIEVE, "Initialization complete")

class RetrieveStateHandler(StateHandler):
    """Retrieve relevant memories from past sessions."""
    
    def __init__(self):
        super().__init__(WorkflowState.RETRIEVE)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call RetrieverAgent to search memory.
        
        Next: ANALYZE_QUERY
        """
        print("🔍 Searching memory...")
        
        memories = await coordinator.retrieve(
            query=context.original_query,
            context_id=context.context_id
        )
        
        context.retrieved_memories = memories
        print(f"📦 Found {len(memories)} relevant memories\n")
        
        return self.transition(context, WorkflowState.ANALYZE_QUERY, "Memory search complete")

class AnalyzeQueryStateHandler(StateHandler):
    """Analyze user query using PerceptionAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.ANALYZE_QUERY)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call PerceptionAgent to understand query.
        
        Next: VALIDATE_QUERY
        """
        print("🧠 Analyzing query...")
        
        analysis = await coordinator.analyze_query(
            query=context.original_query,
            memories=context.retrieved_memories,
            context_id=context.context_id
        )
        
        context.query_analysis = analysis
        print(f"✓ Entities: {', '.join(analysis.entities)}")
        print(f"✓ Intent: {analysis.intent}\n")
        
        return self.transition(context, WorkflowState.VALIDATE_QUERY, "Query analyzed")

class ValidateQueryStateHandler(StateHandler):
    """Validate query using CriticAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.VALIDATE_QUERY)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call CriticAgent to validate query.
        
        Next: COMPLETE (if answer in memory) | PLAN | FAILED
        """
        print("✓ Validating query...")
        
        validation = await coordinator.validate_query(
            query=context.original_query,
            context_id=context.context_id
        )
        
        context.query_validation = validation
        
        if not validation.approved:
            print(f"❌ Query validation failed: {validation.issues[0].description}\n")
            context.mark_failed(f"Query validation failed: {validation.issues[0].description}")
            return self.transition(context, WorkflowState.FAILED, "Query not valid")
        
        # Check if answer already in memory (from query analysis)
        if context.query_analysis and context.query_analysis.extracted_data:
            if context.query_analysis.extracted_data.get("original_goal_achieved"):
                print("✅ Answer found in memory!\n")
                context.mark_complete(
                    final_answer=context.query_analysis.extracted_data.get("solution_summary", ""),
                    solution_summary=context.query_analysis.extracted_data.get("solution_summary", ""),
                    confidence=context.query_analysis.confidence
                )
                return self.transition(context, WorkflowState.COMPLETE, "Answer in memory")
        
        print("✓ Query is valid, proceeding to planning\n")
        return self.transition(context, WorkflowState.PLAN, "Query validated, need execution")

class PlanStateHandler(StateHandler):
    """Create or revise execution plan using DecisionAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.PLAN)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call DecisionAgent to create/revise plan.
        
        Next: VALIDATE_PLAN
        """
        if context.current_plan is None:
            print("📋 Creating initial plan...")
            mode = "initial"
        else:
            # Check if this is a replan (due to failure) or next step generation
            last_step = context.executed_steps[-1] if context.executed_steps else None
            if last_step and last_step.critic_validation and not last_step.critic_validation.approved:
                print("🔄 Replanning due to step failure...")
                mode = "replan"
            else:
                print("➡️  Generating next step...")
                mode = "next_step"
        
        plan = await coordinator.create_plan(
            query=context.original_query,
            perception=context.query_analysis,
            current_plan=context.current_plan,
            completed_steps=context.get_completed_steps(),
            mode=mode,
            context_id=context.context_id
        )
        
        context.add_plan(plan, is_replan=(mode == "replan"))
        
        # If this is next_step mode, transition to the new step
        if mode == "next_step" and plan.steps:
            # Find the pending step (the new step we just added)
            for step in plan.steps:
                if step.status == "pending":
                    print(f"🎯 Transitioning to step {step.index}")
                    context.transition_to_step(step.index)
                    break
        
        print(f"📝 Plan v{plan.version}:")
        for i, step_desc in enumerate(plan.plan_outline):
            print(f"  {i}. {step_desc}")
        
        print(f"🔄 Plan has {len(plan.steps)} executable steps:")
        for step in plan.steps:
            print(f"   Step {step.index}: {step.description[:50]}... (status: {step.status})")
        print(f"\n▶️  Next: {plan.steps[0].description}\n")
        
        return self.transition(context, WorkflowState.VALIDATE_PLAN, f"Plan v{plan.version} created")

class ValidatePlanStateHandler(StateHandler):
    """Validate plan before execution using CriticAgent (NEW!)."""
    
    def __init__(self):
        super().__init__(WorkflowState.VALIDATE_PLAN)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call CriticAgent to validate plan feasibility.
        
        Next: EXECUTE_STEP | PLAN (retry) | FAILED
        """
        print("🔍 Validating plan...")
        
        current_step = context.current_plan.steps[0]
        
        validation = await coordinator.validate_plan(
            plan=context.current_plan.plan_outline,
            step=current_step,
            past_failures=context.get_recent_failures(),
            context_id=context.context_id
        )
        
        if not validation.approved:
            print(f"⚠️  Plan validation failed:")
            for issue in validation.issues:
                print(f"   - {issue.description}")
            print()
            
            # Retry planning (with limit)
            if context.total_replans < context.config.get("max_replans", 3):
                return self.transition(context, WorkflowState.PLAN, "Plan rejected, retry")
            else:
                context.mark_failed("Max replan attempts exceeded")
                return self.transition(context, WorkflowState.FAILED, "Too many replan attempts")
        
        print("✓ Plan validated, ready to execute")
        print(f"🎯 About to execute step index: {context.current_step_index}\n")
        return self.transition(context, WorkflowState.EXECUTE_STEP, "Plan approved")

class ExecuteStepStateHandler(StateHandler):
    """Execute current step using ExecutorAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.EXECUTE_STEP)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call ExecutorAgent to run code.
        
        Next: ANALYZE_RESULT
        """
        current_step = context.current_plan.steps[context.current_step_index]
        
        print(f"⚙️  Executing Step {current_step.index}: {current_step.description}")
        print(f"🔍 Context: current_step_index={context.current_step_index}, plan has {len(context.current_plan.steps)} steps")
        
        if current_step.type.value == "code":
            # Get previous step results for execution context
            completed_steps = context.get_completed_steps()
            previous_results = []
            for step in completed_steps:
                if step.execution_response:
                    previous_results.append({
                        'step_index': step.index,
                        'description': step.description,
                        'execution_response': step.execution_response.model_dump() if hasattr(step.execution_response, 'model_dump') else step.execution_response.__dict__,
                        'result': step.execution_response.result if step.execution_response else None
                    })
            
            result = await coordinator.execute_step(
                code=current_step.code,
                step_description=current_step.description,
                step_index=current_step.index,
                context_id=context.context_id,
                previous_results=previous_results
            )
            
            current_step.execution_response = result
            
            if result.success:
                current_step.status = "completed"
                print(f"✓ Result: {result.result[:100]}...\n" if len(result.result) > 100 else f"✓ Result: {result.result}\n")
            else:
                current_step.status = "failed"
                print(f"❌ Error: {result.error}\n")
        
        elif current_step.type.value == "conclude":
            print(f"💡 Conclusion: {current_step.conclusion}\n")
            current_step.status = "completed"
        
        elif current_step.type.value == "nop":
            print(f"❓ Clarification needed: {current_step.description}\n")
            current_step.status = "clarification_needed"
            context.mark_failed("Clarification needed from user")
            return self.transition(context, WorkflowState.FAILED, "NOP step - clarification needed")
        
        context.add_step(current_step)
        
        return self.transition(context, WorkflowState.ANALYZE_RESULT, "Step executed")

class AnalyzeResultStateHandler(StateHandler):
    """Analyze execution result using PerceptionAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.ANALYZE_RESULT)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call PerceptionAgent to understand result.
        
        Next: VALIDATE_RESULT
        """
        print("🧠 Analyzing result...")
        
        current_step = context.executed_steps[-1]
        
        if current_step.execution_response:
            result_text = current_step.execution_response.result or current_step.execution_response.error
        else:
            result_text = current_step.conclusion or "No result"
        
        analysis = await coordinator.analyze_result(
            result=result_text,
            step_description=current_step.description,
            current_plan=context.get_current_plan(),
            context_id=context.context_id
        )
        
        current_step.perception_analysis = analysis
        print(f"✓ Analysis complete\n")
        
        return self.transition(context, WorkflowState.VALIDATE_RESULT, "Result analyzed")

class ValidateResultStateHandler(StateHandler):
    """Validate result and check goal achievement using CriticAgent."""
    
    def __init__(self):
        super().__init__(WorkflowState.VALIDATE_RESULT)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Call CriticAgent to validate result.
        
        Next: COMPLETE | DECIDE_NEXT | PLAN (replan)
        """
        print("✓ Validating result...")
        
        current_step = context.executed_steps[-1]
        
        validation = await coordinator.validate_result(
            result=current_step.execution_response.result if current_step.execution_response else "",
            error=current_step.execution_response.error if current_step.execution_response else "",
            step_description=current_step.description,
            original_goal=context.original_query,
            context_id=context.context_id
        )
        
        current_step.critic_validation = validation
        
        # Check goal achievement
        if validation.goal_status and validation.goal_status.original_goal_achieved:
            print("✅ Original goal achieved!\n")
            context.mark_complete(
                final_answer=validation.goal_status.reasoning,
                solution_summary=current_step.perception_analysis.extracted_data.get("solution_summary", ""),
                confidence=validation.confidence
            )
            return self.transition(context, WorkflowState.COMPLETE, "Goal achieved")
        
        # Check if step was useful
        if validation.approved:
            print("✓ Step succeeded, continuing\n")
            return self.transition(context, WorkflowState.DECIDE_NEXT, "Step succeeded")
        else:
            print("⚠️  Step failed or unhelpful, replanning\n")
            return self.transition(context, WorkflowState.PLAN, "Step failed, replan needed")

class DecideNextStateHandler(StateHandler):
    """Decide what to do after successful step."""
    
    def __init__(self):
        super().__init__(WorkflowState.DECIDE_NEXT)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Check if more steps in plan or if done.
        
        Next: EXECUTE_STEP | COMPLETE
        """
        # Check max steps limit
        if context.total_steps_attempted >= context.config.get("max_steps", 10):
            print("⚠️  Max steps reached\n")
            context.mark_failed("Max steps limit reached")
            return self.transition(context, WorkflowState.FAILED, "Max steps exceeded")
        
        # Check if more steps in current plan
        next_step_index = context.current_step_index + 1
        
        print(f"🔍 DEBUG: DecideNext state")
        print(f"  - Current step index: {context.current_step_index}")
        print(f"  - Next step index: {next_step_index}")
        print(f"  - Total steps in plan: {len(context.current_plan.steps) if context.current_plan else 0}")
        print(f"  - Plan outline length: {len(context.current_plan.plan_outline) if context.current_plan and context.current_plan.plan_outline else 0}")
        if context.current_plan and context.current_plan.steps:
            for i, step in enumerate(context.current_plan.steps):
                print(f"  - Step {i}: {step.description[:50]}...")
        
        # Check if there are more steps in the plan outline (not just executed steps)
        plan_outline_length = len(context.current_plan.plan_outline) if context.current_plan and context.current_plan.plan_outline else 0
        
        if next_step_index < plan_outline_length:
            # There are more steps in the plan outline, need to generate the next step
            print(f"📋 More steps in plan outline ({next_step_index} < {plan_outline_length}), generating next step...\n")
            
            # Go to PLAN to generate the executable code for the next step
            return self.transition(context, WorkflowState.PLAN, "Generate next step from plan outline")
        else:
            # Plan is complete
            print("✅ All steps in plan completed\n")
            context.mark_complete(
                final_answer="Plan completed successfully",
                solution_summary="All planned steps have been executed",
                confidence=0.9
            )
            return self.transition(context, WorkflowState.COMPLETE, "Plan completed")

class CompleteStateHandler(StateHandler):
    """Terminal state - execution completed successfully."""
    
    def __init__(self):
        super().__init__(WorkflowState.COMPLETE)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Mark completion and save session.
        
        Next: None (terminal)
        """
        print(f"\n{'='*60}")
        print(f"✅ SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Answer: {context.final_answer[:200]}...")
        print(f"📈 Confidence: {context.confidence:.2f}")
        print(f"⏱️  Time: {context.total_time:.2f}s")
        print(f"🔢 Steps: {context.total_steps_attempted}")
        print(f"🔄 Replans: {context.total_replans}")
        print(f"{'='*60}\n")
        
        # Save session
        await coordinator.save_session(context)
        
        return WorkflowState.COMPLETE  # Terminal

class FailedStateHandler(StateHandler):
    """Terminal state - execution failed."""
    
    def __init__(self):
        super().__init__(WorkflowState.FAILED)
    
    async def process(self, context: ExecutionContext, coordinator) -> WorkflowState:
        """
        Mark failure and save session.
        
        Next: None (terminal)
        """
        print(f"\n{'='*60}")
        print(f"❌ SESSION FAILED")
        print(f"{'='*60}")
        print(f"💥 Error: {context.error_message}")
        print(f"⏱️  Time: {context.total_time:.2f}s" if context.total_time else "⏱️  Time: N/A")
        print(f"🔢 Steps attempted: {context.total_steps_attempted}")
        print(f"{'='*60}\n")
        
        # Save session
        await coordinator.save_session(context)
        
        return WorkflowState.FAILED  # Terminal

#Workflow Engine
class WorkflowEngine:
    """
    State machine for agent execution flow.
    
    Replaces the old while loop with explicit state handlers.
    Each state handler processes one phase and returns next state.
    
    Usage:
        engine = WorkflowEngine()
        final_state = await engine.execute(context, coordinator)
    """
    
    def __init__(self):
        """Initialize workflow engine with state handlers."""
        self.handlers: Dict[WorkflowState, StateHandler] = {
            WorkflowState.INIT: InitStateHandler(),
            WorkflowState.RETRIEVE: RetrieveStateHandler(),
            WorkflowState.ANALYZE_QUERY: AnalyzeQueryStateHandler(),
            WorkflowState.VALIDATE_QUERY: ValidateQueryStateHandler(),
            WorkflowState.PLAN: PlanStateHandler(),
            WorkflowState.VALIDATE_PLAN: ValidatePlanStateHandler(),
            WorkflowState.EXECUTE_STEP: ExecuteStepStateHandler(),
            WorkflowState.ANALYZE_RESULT: AnalyzeResultStateHandler(),
            WorkflowState.VALIDATE_RESULT: ValidateResultStateHandler(),
            WorkflowState.DECIDE_NEXT: DecideNextStateHandler(),
            WorkflowState.COMPLETE: CompleteStateHandler(),
            WorkflowState.FAILED: FailedStateHandler()
        }
    
    async def execute(
        self, 
        context: ExecutionContext, 
        coordinator
    ) -> WorkflowState:
        """
        Execute workflow from INIT to terminal state.
        
        Args:
            context: ExecutionContext with query and config
            coordinator: AgentCoordinator for calling agents
        
        Returns:
            Final WorkflowState (COMPLETE or FAILED)
        """
        current_state = WorkflowState.INIT
        
        while not current_state.is_terminal():
            handler = self.handlers[current_state]
            
            try:
                next_state = await handler.process(context, coordinator)
                current_state = next_state
                
            except Exception as e:
                print(f"\n❌ Workflow error in {current_state.value}: {e}\n")
                context.mark_failed(f"Workflow error: {str(e)}")
                current_state = WorkflowState.FAILED
        
        return current_state
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"WorkflowEngine(states={len(self.handlers)})"