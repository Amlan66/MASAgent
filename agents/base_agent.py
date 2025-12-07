"""
BaseAgent : Abstract class for all the agents in the system.

All agents (retriever, perception, critic, decision, executor, memory) inherit from this class
to ensure consistent interface and behavior.

Benefits:
- Consistent API Across all agents
- Built-in metrics tracking
- Request validation
- Easy to test
- Type safety
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import time

from core.models import AgentRequest, AgentResponse

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    All concrete agents must implement the process() method.
    Provides common functionality:
    - Metrics tracking (calls, time, errors)
    - Request validation
    - Performance monitoring
    
    Example:
        class RetrieverAgent(BaseAgent):
            async def process(self, request: RetrievalRequest) -> RetrievalResponse:
                # Implementation here
                pass
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base agent
        Args:
            config: Configuration dictionary loaded from profiles.yaml

        """
        self.config = config
        self.agent_name = config.get("agent_name", self.__class__.__name__)
        self.agent_type = config.get("agent_type", "unknown")

        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "last_call_time": None,
            "errors": []
        }

        self.timeout = config.get("timeout",None)
        self.max_retries = config.get("max_retries",3)
        self.enabled = config.get("enabled",True)

        self.is_initialized = False
        self.initialization_error: Optional[str] = None

    #abstract methods

    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Main entry point for agent processing.
        
        All agents must implement this method.
        This is the core of the agent's functionality.
        
        Args:
            request: Request object (specific type per agent)
                    - RetrievalRequest for RetrieverAgent
                    - PerceptionRequest for PerceptionAgent
                    - etc.
        
        Returns:
            AgentResponse: Response object (specific type per agent)
        
        Raises:
            NotImplementedError: If subclass doesn't implement this
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process() method"
        )

    def validate_request(self, request: AgentRequest) -> bool:
        """
        Validate incoming request before processing.
        
        Base implementation checks basic requirements.
        Subclasses can override for custom validation.
        
        Args:
            request: Request to validate
        
        Returns:
            True if request is valid
        
        Raises:
            ValueError: If request is invalid
        """

        if not self.enabled:
            raise ValueError(f"{self.agent_name} is disabled")
        
        if hasattr(self, 'requires_initialization'):
            if self.requires_initialization and not self.is_initialized:
                raise ValueError(
                    f"{self.agent_name} not initialized. "
                    f"Error: {self.initialization_error}"
                )

        # Check request_id and context_id exist
        if not hasattr(request, 'request_id') or not request.request_id:
            raise ValueError("Request missing request_id")
        
        if not hasattr(request, 'context_id') or not request.context_id:
            raise ValueError("Request missing context_id")
        
        return True

    def validate_response(self, response: AgentResponse) -> bool:
        """
        Validate response before returning.
        
        Ensures response has required fields.
        Subclasses can override for custom validation.
        
        Args:
            response: Response to validate
        
        Returns:
            True if response is valid
        
        Raises:
            ValueError: If response is invalid
        """
                
        # Check required fields
        if not hasattr(response, 'response_id') or not response.response_id:
            raise ValueError("Response missing response_id")
        
        if not hasattr(response, 'success'):
            raise ValueError("Response missing success field")
        
        return True
    
    def update_metrics(
        self,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None
    ) -> None:
        """
        Update agent performance metrics.
        
        Called automatically after each process() call.
        
        Args:
            execution_time: Time taken in seconds
            success: Whether call succeeded
            error: Exception if failed (optional)
        """
        self.metrics["total_calls"] += 1
        self.metrics["total_time"] += execution_time
        self.metrics["last_call_time"] = datetime.now()
        
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
            
            # Track error (keep last 10)
            if error:
                error_record = {
                    "timestamp": datetime.now(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                }
                self.metrics["errors"].append(error_record)
                if len(self.metrics["errors"]) > 10:
                    self.metrics["errors"].pop(0)
        
        # Update time statistics
        self.metrics["average_time"] = (
            self.metrics["total_time"] / self.metrics["total_calls"]
        )
        self.metrics["min_time"] = min(
            self.metrics["min_time"],
            execution_time
        )
        self.metrics["max_time"] = max(
            self.metrics["max_time"],
            execution_time
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Useful for monitoring, debugging, and optimization.
        
        Returns:
            Dict with metrics:
                - total_calls: Total number of calls
                - successful_calls: Number of successes
                - failed_calls: Number of failures
                - success_rate: Percentage of successful calls
                - average_time: Average processing time
                - min_time: Fastest call
                - max_time: Slowest call
                - last_call_time: When last called
                - recent_errors: List of recent errors
        """
        metrics = self.metrics.copy()
        
        # Calculate success rate
        if metrics["total_calls"] > 0:
            metrics["success_rate"] = (
                metrics["successful_calls"] / metrics["total_calls"]
            ) * 100
        else:
            metrics["success_rate"] = 0.0
        
        # Format times for readability
        metrics["average_time"] = round(metrics["average_time"], 3)
        metrics["min_time"] = round(metrics["min_time"], 3) if metrics["min_time"] != float('inf') else 0.0
        metrics["max_time"] = round(metrics["max_time"], 3)
        
        # Add agent info
        metrics["agent_name"] = self.agent_name
        metrics["agent_type"] = self.agent_type
        
        return metrics
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics to zero.
        
        Useful for testing or fresh monitoring windows.
        """
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "last_call_time": None,
            "errors": []
        }
    
    # wrapper used by coordinator
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Wrapper around process() that adds validation and metrics

        This is what the coordinator will call and not process() directly
        Handles:
        - request validation
        - timing
        - metrics tracking
        - error handling
        - reponse validation

        Args:
            request: AgentRequest object
        Returns:
            AgentResponse object from process()
        """

        start_time = time.time()
        error = None
        
        try:
            self.validate_request(request)
            response = await self.process(request)
            self.validate_response(response)

            execution_time = time.time() - start_time
            self.update_metrics(execution_time, success=True)
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            error = e
            self.update_metrics(execution_time, success=False, error=error)
            raise 
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.agent_name}, "
            f"calls={self.metrics['total_calls']}, "
            f"success_rate={self.get_metrics()['success_rate']:.1f}%)"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.agent_name} ({self.agent_type})"

