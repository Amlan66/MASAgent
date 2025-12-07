"""
ExecutorAgent: Executes Python code in a sandboxed environment with MCP tool access.

"""

import ast
import asyncio
import time
import builtins
import textwrap
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from mcp_servers.multiMCP import MultiMCP

from agents import BaseAgent
from core import ExecutionRequest, ExecutionResponse

#configs
ALLOWED_MODULES = {
    "math", "cmath", "decimal", "fractions", "random", "statistics", 
    "itertools", "functools", "operator", "string", "re", "datetime", 
    "calendar", "time", "collections", "heapq", "bisect", "types", 
    "copy", "enum", "uuid", "dataclasses", "typing", "pprint", "json", 
    "base64", "hashlib", "hmac", "secrets", "struct", "zlib", "gzip", 
    "bz2", "lzma", "io", "pathlib", "tempfile", "textwrap", "difflib", 
    "unicodedata", "html", "html.parser", "xml", "xml.etree.ElementTree", 
    "csv", "sqlite3", "contextlib", "traceback", "ast", "tokenize", 
    "token", "builtins"
}

MAX_FUNCTIONS = 5
TIMEOUT_PER_FUNCTION = 500 #seconds

#AST Transformers
class KeywordStripper(ast.NodeTransformer):
    """
    Rewrite all function calls to remove keyword args and keep only values as positional.
    
    Example:
        add(x=1, y=2) -> add(1, 2)
    """
    def visit_Call(self, node):
        self.generic_visit(node)
        if node.keywords:
            # Convert all keyword arguments into positional args (discard names)
            for kw in node.keywords:
                node.args.append(kw.value)
            node.keywords = []
        return node

class AwaitTransformer(ast.NodeTransformer):
    """
    Auto-await known async MCP tool calls.
    
    Transforms:
        result = search_web("query") 
    Into:
        result = await search_web("query")
    """
    def __init__(self, async_funcs: Set[str]):
        self.async_funcs = async_funcs

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.async_funcs:
            return ast.Await(value=node)
        return node

#Executor AGent
class ExecutorAgent(BaseAgent):
    """
    Executes Python code with mcp tool access
    Features:
    - Sandboxed Execution
    - AST Transformations
    - Function call limits
    - Timeout enforcement
    - MCP Tool integration
    - Parallel tool execution support

    Usage:
        executor = ExecutorAgent(config={"agent_name": "executor"}, multi_mcp=multi_mcp)
        await executor.initialize()
        
        request = ExecutionRequest(
            code="result = add(2, 3)",
            step_description="Add two numbers",
            step_index=0,
            context_id=ctx.context_id
        )
        
        response = await executor.execute(request)
        print(response.result)  # "5"

    """

    def __init__(self, config: Dict[str, Any], multi_mcp: MultiMCP):
        """
        Initialize ExecutorAgent

        Args:
            config: Configuration dictionary loaded from profiles.yaml
            multi_mcp: MultiMCP instance for tool access
        """
        super().__init__(config)

        self.multi_mcp = multi_mcp
        self.max_functions = config.get("max_functions", MAX_FUNCTIONS)
        self.timeout_per_function = config.get("timeout_per_function", TIMEOUT_PER_FUNCTION)
        self.allowed_modules = config.get("allowed_modules", ALLOWED_MODULES)

        self.last_tools_called = []

    async def initialize(self) -> bool:
        """
        Initialize the executor agent
        Verifys MultiMCP is ready and tools are available
        """
        try:
            tools = self.multi_mcp.get_all_tools()
            if not tools:
                raise ValueError("MultiMCP has no tools registered")
            
            self.is_initialized = True
            print(f"✅ ExecutorAgent initialized with {len(tools)} tools")
        
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise
    
    async def process(self, request: ExecutionRequest) -> ExecutionResponse:
        """
        Execute Python code from the request.
        
        This is the main entry point called by the coordinator via execute().
        
        Args:
            request: ExecutionRequest with code, step info, context_id
        
        Returns:
            ExecutionResponse with result or error
        """

        start_time = time.perf_counter()
        start_timestamp = datetime.now().isoformat()
        
        try:
            result_dict = await self._execute_code(
                code=request.code,
                timeout_override=request.timeout,
                previous_results=request.previous_results
            )

            processing_time = time.perf_counter() - start_time

            response = ExecutionResponse(
                request_id=request.request_id,
                status=result_dict["status"],
                result=result_dict.get("result"),
                error=result_dict.get("error"),
                execution_time=result_dict["execution_time"],
                total_time=result_dict["total_time"],
                function_count=result_dict.get("function_count"),
                tools_called=self.last_tools_called,
                success=(result_dict["status"] == "success"),
                processing_time=processing_time
            )

            return response

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            return ExecutionResponse(
                request_id=request.request_id,
                status="error",
                error=f"ExecutorAgent error: {str(e)}",
                execution_time=start_timestamp,
                total_time=str(round(processing_time, 3)),
                success=False,
                processing_time=processing_time
            )

    #Core execution logic

    async def _execute_code(self, code:str, timeout_override:Optional[int]=None, previous_results:Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
        """
        Execute user code in sandboxed environment

        This is the core execution logic

        Args:
            code: Python code to execute
            timeout_override: Optional timeout in seconds 

        Returns:
            Dict with keys: status, result/error, execution_time, total_time, function_count
        """        

        start_time = time.perf_counter()
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            func_count = self._count_function_calls(code)
            if func_count > self.max_functions:
                return {
                    "status": "error",
                    "error": f"Too many functions ({func_count} > {self.max_functions})",
                    "execution_time": start_timestamp,
                    "total_time": str(round(time.perf_counter() - start_time, 3)),
                    "function_count": func_count
                }
            
            tool_funcs = {
                tool.name: self._make_tool_proxy(tool.name)
                for tool in self.multi_mcp.get_all_tools()
            }

            self.last_tools_called = []

            sandbox = self._build_safe_globals(tool_funcs)
            
            # Add previous step results to execution environment
            if previous_results:
                sandbox['previous_results'] = previous_results
                # Also provide completed_steps for backward compatibility
                sandbox['completed_steps'] = previous_results
            
            local_vars = {}

            cleaned_code = textwrap.dedent(code.strip())
            tree = ast.parse(cleaned_code)

            has_return = any(isinstance(node, ast.Return) for node in tree.body)

            has_result = any(
                isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "result" for t in node.targets
                )
                for node in tree.body
            )
            if not has_return and has_result:
                tree.body.append(ast.Return(value=ast.Name(id="result", ctx=ast.Load())))
            
            # Apply AST transformations
            tree = KeywordStripper().visit(tree)
            tree = AwaitTransformer(set(tool_funcs)).visit(tree)
            ast.fix_missing_locations(tree)
            
            # Wrap in async function
            func_def = ast.AsyncFunctionDef(
                name="__main",
                args=ast.arguments(
                    posonlyargs=[], 
                    args=[], 
                    kwonlyargs=[], 
                    kw_defaults=[], 
                    defaults=[]
                ),
                body=tree.body,
                decorator_list=[]
            )
            wrapper = ast.Module(body=[func_def], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            
            # Compile and execute
            compiled = compile(wrapper, filename="<user_code>", mode="exec")
            exec(compiled, sandbox, local_vars)

            # Run with timeout
            try:
                if timeout_override:
                    timeout = timeout_override
                else:
                    timeout = max(3, func_count * self.timeout_per_function)
                
                returned = await asyncio.wait_for(
                    local_vars["__main"](), 
                    timeout=timeout
                )
                
                result_value = returned if returned is not None else sandbox.get("result_holder", "None")
                
                # Handle MCP tool errors (CallToolResult with isError=True)
                if hasattr(result_value, "isError") and getattr(result_value, "isError", False):
                    error_msg = None
                    try:
                        error_msg = result_value.content[0].text.strip()
                    except Exception:
                        error_msg = str(result_value)
                    
                    return {
                        "status": "error",
                        "error": error_msg,
                        "execution_time": start_timestamp,
                        "total_time": str(round(time.perf_counter() - start_time, 3)),
                        "function_count": func_count
                    }
                
                # Success
                return {
                    "status": "success",
                    "result": str(result_value),
                    "execution_time": start_timestamp,
                    "total_time": str(round(time.perf_counter() - start_time, 3)),
                    "function_count": func_count
                }
                
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "error": f"Execution timed out after {timeout} seconds",
                    "execution_time": start_timestamp,
                    "total_time": str(round(time.perf_counter() - start_time, 3)),
                    "function_count": func_count
                }
            
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)}",
                    "execution_time": start_timestamp,
                    "total_time": str(round(time.perf_counter() - start_time, 3)),
                    "function_count": func_count
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": start_timestamp,
                "total_time": str(round(time.perf_counter() - start_time, 3))
            }

    # ───────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ───────────────────────────────────────────────────────────────
    
    def _count_function_calls(self, code: str) -> int:
        """
        Count number of function calls in code.
        Used to enforce MAX_FUNCTIONS limit.
        
        Args:
            code: Python code string
        
        Returns:
            Number of function calls
        """
        tree = ast.parse(code)
        return sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    
    def _build_safe_globals(self, mcp_funcs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build sandboxed globals dictionary.
        
        Includes:
        - Restricted builtins (no file I/O, no exec, etc.)
        - Allowed modules (math, json, etc.)
        - MCP tool functions
        - final_answer() helper
        - parallel() for concurrent tool calls
        
        Args:
            mcp_funcs: Dict of tool_name -> tool_proxy_function
        
        Returns:
            Globals dict for code execution
        """
        safe_globals = {
            "__builtins__": {
                k: getattr(builtins, k)
                for k in ("range", "len", "int", "float", "str", "list", 
                         "dict", "print", "sum", "__import__")
            },
            **mcp_funcs,
        }
        
        # Import allowed modules
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass  # Skip if module not available
        
        # final_answer() helper for LLM-style results
        safe_globals["final_answer"] = lambda x: safe_globals.setdefault("result_holder", x)
        
        # parallel() for concurrent tool execution
        async def parallel(*tool_calls):
            """
            Execute multiple MCP tools in parallel.
            
            Usage:
                results = parallel(
                    ("add", 1, 2),
                    ("multiply", 3, 4)
                )
            """
            coros = [
                self.multi_mcp.function_wrapper(tool_name, *args)
                for tool_name, *args in tool_calls
            ]
            return await asyncio.gather(*coros)
        
        safe_globals["parallel"] = parallel
        
        return safe_globals
    
    def _make_tool_proxy(self, tool_name: str):
        """
        Create a proxy function for an MCP tool.
        
        The proxy:
        1. Tracks that this tool was called (for metrics)
        2. Forwards call to MultiMCP.function_wrapper()
        
        Args:
            tool_name: Name of the MCP tool
        
        Returns:
            Async function that calls the tool
        """
        async def _tool_fn(*args):
            # Track tool usage
            if tool_name not in self.last_tools_called:
                self.last_tools_called.append(tool_name)
            
            # Call tool via MultiMCP
            result = await self.multi_mcp.function_wrapper(tool_name, *args)
            
            return result
        
        _tool_fn.__name__ = tool_name
        return _tool_fn
    
    # ───────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ───────────────────────────────────────────────────────────────
    
    def get_available_tools(self) -> list:
        """
        Get list of available MCP tools.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.multi_mcp.get_all_tools()]
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"ExecutorAgent("
            f"tools={len(self.get_available_tools())}, "
            f"max_functions={self.max_functions}, "
            f"calls={self.metrics['total_calls']})"
        )