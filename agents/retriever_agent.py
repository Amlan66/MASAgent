"""
RetrieverAgent: Retrieves relevant information from memory and other sources.

This agent wraps the existing MemorySearch logic into the new agent architecture.
Uses RapidFuzz for fuzzy matching against past successful sessions.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rapidfuzz import fuzz

from agents import BaseAgent
from core import RetrievalRequest, RetrievalResponse, MemoryResult

class RetrieverAgent(BaseAgent):
    """
    RetrieverAgent: Searches for relevant past sessions from memory.
    
    Features:
    - RapidFuzz-based fuzzy matching
    - Configurable scoring (query, summary, length penalty)
    - Multiple JSON format support
    - Filters for original_goal_achieved=True only
    - Extensible for future RAG/vector search
    
    Scoring Formula (from original):
        score = 0.5 * query_score + 0.4 * summary_score - 0.05 * length_penalty
    
    Usage:
        retriever = RetrieverAgent(
            config={"agent_name": "retriever"},
            memory_path="storage/session_logs"
        )
        await retriever.initialize()
        
        request = RetrievalRequest(
            query="What is 2+2?",
            sources=["memory"],
            top_k=3,
            context_id=ctx.context_id
        )
        
        response = await retriever.execute(request)
        for result in response.results:
            print(f"{result.query} -> {result.solution_summary}")
    """

    def __init__(self, config: Dict[str, Any], memory_path: str = "storage/session_logs"):
        """
        Initialize RetrieverAgent

        Args:
            config: Configuration dictionary loaded from profiles.yaml
            memory_path: Path to the memory directory
        """
        super().__init__(config)

        self.memory_path = Path(memory_path)
        self.default_top_k = config.get("default_top_k", 3)

        #scoring weights
        self.query_weight = config.get("query_weight", 0.5)
        self.summary_weight = config.get("summary_weight", 0.4)
        self.length_penalty_weight = config.get("length_penalty_weight", 0.05)

        self._memory_cache: Optional[List[Dict]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = config.get("cache_ttl", 300)

    async def initialize(self) -> None:
        """
        Initialize the retriever agent
        Verifies memory path is exists and is readable
        """

        try:
            if not self.memory_path.exists():
                self.memory_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Memory path created: {self.memory_path}")

            if not os.access(self.memory_path, os.R_OK):
                raise PermissionError(f"Memory path is not readable: {self.memory_path}")
            
            session_count = len(list(self.memory_path.rglob("*.json")))

            self.is_initialized = True
            print(f"✅ RetrieverAgent initialized with {session_count} session(s) in memory")
        
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise
    
    async def process(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Process Retrieval Request and return relevant memories
        Args:
            request: RetrievalRequest with query, sources, top_k

        Returns:
            RetrievalResponse with matching memories
        """

        start_time = time.perf_counter()
        
        all_results = []
        sources_tried = []
        sources_succeeded = []
        retrieval_time_by_source = {}

        try:
            for source in request.sources:
                sources_tried.append(source)
                source_start = time.perf_counter()

                if source == "memory":
                    source_results = await self._search_memory(
                        query=request.query,
                        top_k=request.top_k
                    )
                    all_results.extend(source_results)
                    sources_succeeded.append(source)
                
                elif source == "rag":
                    #TODO: Implement RAG retrieval
                    pass
                
                elif source == "graph":
                    #Future knowledge graph search
                    pass
            
                retrieval_time_by_source[source] = time.perf_counter() - source_start
            
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)

            final_results = all_results[:request.top_k]
            
            processing_time = time.perf_counter() - start_time

            return RetrievalResponse(
                request_id=request.request_id,
                results=final_results,
                sources_tried=sources_tried,
                sources_succeeded=sources_succeeded,
                total_results=len(final_results),
                retrieval_time_by_source=retrieval_time_by_source,
                processing_time=processing_time,
                success=True
            )
        except Exception as e:
            processing_time = time.perf_counter() - start_time

            return RetrievalResponse(
                request_id=request.request_id,
                results=[],
                sources_tried=sources_tried,
                sources_succeeded=sources_succeeded,
                total_results=0,
                retrieval_time_by_source=retrieval_time_by_source,
                processing_time=processing_time,
                success=False,
                error_message=f"RetrieverAgent error: {str(e)}"
            )
    
    #core search logic
    async def _search_memory(self, query:str, top_k:int) -> List[MemoryResult]:
        """
        Search memory for relevant past sessions.
        Uses RapidFuzz for fuzzy matching with custom scoring
        Only returns sessions where original_goal_achieved=True

        Args:
            User query to match against
            top_k: Number of top results to return

        Returns:
            List of MemoryResult objects sorted by relevance score
        """

        memory_results = await self._load_memory_entries()
        
        if not memory_results:
            return []
        
        scored_results = []

        for entry in memory_results:
            query_score = fuzz.partial_ratio(
                query.lower(),
                entry["query"].lower()
            )
            summary_score = fuzz.partial_ratio(
                query.lower(),
                entry["solution_summary"].lower()
            )
            length_penalty = len(entry["solution_summary"]) / 100

            score = (
                self.query_weight * query_score +
                self.summary_weight * summary_score -
                self.length_penalty_weight * length_penalty
            )

            scored_results.append((score, entry))

        top_matches = sorted(scored_results, key=lambda x: x[0], reverse=True)[:top_k]

        results = []
        for score, entry in top_matches:
            result = MemoryResult(
                source="memory",
                file_path=entry["file_path"],
                session_id=entry.get("session_id", "unknown"),
                query=entry["query"],
                solution_summary=entry["solution_summary"],
                result_requirement=entry["result_requirement"],
                reevance_score=score,
                retrieved_at=datetime.now()
            )
            results.append(result)
        
        return results
    
    async def _load_memory_entries(self) -> List[Dict]:
        """
        Load all memory entries from session logs.
        
        Supports multiple JSON formats:
        - FORMAT 1: List of sessions
        - FORMAT 2: Single session dict
        - FORMAT 3: Dict with "turns" key
        
        Returns:
            List of memory entry dicts
        """
        # Check cache
        if self._is_cache_valid():
            return self._memory_cache
        
        memory_entries = []
        all_json_files = list(self.memory_path.rglob("*.json"))
        
        print(f"🔍 Found {len(all_json_files)} JSON file(s) in '{self.memory_path}'")
        
        for file in all_json_files:
            count_before = len(memory_entries)
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # FORMAT 1: List of sessions
                if isinstance(content, list):
                    for session in content:
                        self._extract_entry(session, file.name, memory_entries)
                
                # FORMAT 2: Single session dict
                elif isinstance(content, dict) and "session_id" in content:
                    self._extract_entry(content, file.name, memory_entries)
                
                # FORMAT 3: Dict with turns
                elif isinstance(content, dict) and "turns" in content:
                    for turn in content["turns"]:
                        self._extract_entry(turn, file.name, memory_entries)
                
            except Exception as e:
                print(f"⚠️ Skipping '{file}': {e}")
                continue
            
            count_after = len(memory_entries)
            if count_after > count_before:
                print(f"✅ {file.name}: {count_after - count_before} matching entries")
        
        print(f"📦 Total usable memory entries collected: {len(memory_entries)}\n")
        
        # Update cache
        self._memory_cache = memory_entries
        self._cache_timestamp = time.time()
        
        return memory_entries

    
    def _extract_entry(self, obj: dict, file_name:str, memory_entries:List[Dict]) -> None:
        """
        Extract memory entry from session object

        Only extracts if original_goal_achieved=True
        Uses recursive search to handle nested structure

        Args:
            obj: Session object (dict)
            file_name: Name of source file
            memory_entries: list to append results to
        """

        original_obj = obj #keep top-level reference

        def recursive_find(obj):
            """Recursive search for original_goal_achieved=True"""
            
            if isinstance(obj, dict):
                if obj.get("original_goal_achieved") is True:
                    query = extract_query(original_obj)
                    return {
                        "query":query,
                        "summary": obj.get("solution_summary", ""),
                        "requirement": obj.get("result_requirement", ""),
                        "session_id": original_obj.get("session_id", "")
                    }
                for v in obj.values():
                    result = recursive_find(v)
                    if result:
                        return result
            
            elif isinstance(obj, list):
                for item in obj:
                    result = recursive_find(item)
                    if result:
                        return result
                
            return None
        
        def extract_query(obj) -> str:
            """Recursively extract query field"""
            if isinstance(obj, dict):
                if "query" in obj and isinstance(obj["query"], str):
                    return obj["query"]
                if "original_query" in obj and isinstance(obj["original_query"], str):
                    return obj["original_query"]
                for v in obj.values():
                    q = extract_query(v)
                    if q:
                        return q
            elif isinstance(obj, list):
                for item in obj:
                    q = extract_query(item)
                    if q:
                        return q
            return ""
        
        try:
            match = recursive_find(obj)
            if match and match["query"]:
                memory_entries.append({
                    "file_path": file_name,
                    "session_id": match["session_id"],
                    "query": match["query"],
                    "result_requirement": match["requirement"],
                    "solution_summary": match["summary"]
                })
        except Exception as e:
            print(f"❌ Error parsing {file_name}: {e}")


    def _is_cache_valid(self) -> bool:
        """
        Check if memory cache is still valid.
        
        Returns:
            True if cache exists and hasn't expired
        """
        if self._memory_cache is None or self._cache_timestamp is None:
            return False
        
        age = time.time() - self._cache_timestamp
        return age < self._cache_ttl
    
    def clear_cache(self) -> None:
        """
        Clear the memory cache.
        Useful when new sessions are added.
        """
        self._memory_cache = None
        self._cache_timestamp = None
    
    # ───────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ───────────────────────────────────────────────────────────────
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available memory.
        
        Returns:
            Dict with file count, entry count, cache status
        """
        file_count = len(list(self.memory_path.rglob("*.json")))
        cache_valid = self._is_cache_valid()
        entry_count = len(self._memory_cache) if cache_valid else 0
        
        return {
            "memory_path": str(self.memory_path),
            "total_files": file_count,
            "cached_entries": entry_count,
            "cache_valid": cache_valid,
            "cache_age": time.time() - self._cache_timestamp if self._cache_timestamp else None
        }
    
    def __repr__(self) -> str:
        """Debug representation."""
        stats = self.get_memory_stats()
        return (
            f"RetrieverAgent("
            f"files={stats['total_files']}, "
            f"cached={stats['cached_entries']}, "
            f"calls={self.metrics['total_calls']})"
        )
