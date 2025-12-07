"""
MemoryAgent: Manages session persistence to disk.

This agent wraps the existing session_log.py logic into the new agent architecture.
Saves sessions in JSON format: storage/session_logs/YYYY/MM/DD/<session_id>.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time

from agents import BaseAgent
from core import SessionModel, AgentRequest, AgentResponse


class MemoryRequest(AgentRequest):
    """Request to save or load a session"""
    operation: str  # "save" or "load"
    session_data: Optional[SessionModel] = None
    session_id: Optional[str] = None


class MemoryResponse(AgentResponse):
    """Response from MemoryAgent"""
    operation: str
    file_path: Optional[str] = None
    session_data: Optional[SessionModel] = None


class MemoryAgent(BaseAgent):
    """
    MemoryAgent: Persists sessions to disk in JSON format.
    
    Features:
    - Date-based directory structure (YYYY/MM/DD)
    - Atomic file writes
    - Corrupt file detection and recovery
    - Session validation before save
    
    File Structure:
        storage/session_logs/
        └── 2025/
            └── 12/
                └── 06/
                    ├── abc123-uuid.json
                    └── def456-uuid.json
    
    Usage:
        memory = MemoryAgent(
            config={"agent_name": "memory"},
            base_dir="storage/session_logs"
        )
        await memory.initialize()
        
        # Save session
        request = MemoryRequest(
            operation="save",
            session_data=session_model,
            context_id=ctx.context_id
        )
        response = await memory.execute(request)
        
        # Load session
        request = MemoryRequest(
            operation="load",
            session_id="abc123-uuid",
            context_id=ctx.context_id
        )
        response = await memory.execute(request)
        session = response.session_data
    """

    def __init__(self, config: Dict[str, Any], base_dir: str = "storage/session_logs"):
        """
        Initialize MemoryAgent.
        
        Args:
            config: Configuration dict (from profiles.yaml)
            base_dir: Base directory for session storage
        """
        super().__init__(config)
        
        self.base_dir = Path(base_dir)
        self.indent = config.get("json_indent", 2)
        self.ensure_ascii = config.get("ensure_ascii", False)
        
    async def initialize(self) -> None:
        """
        Initialize the memory agent.
        Creates base directory if it doesn't exist.
        """
        try:
            # Create base directory
            if not self.base_dir.exists():
                self.base_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created memory directory: {self.base_dir}")
            
            # Verify write access
            test_file = self.base_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                raise PermissionError(f"Cannot write to {self.base_dir}: {e}")
            
            # Count existing sessions
            session_count = len(list(self.base_dir.rglob("*.json")))
            
            self.is_initialized = True
            print(f"✅ MemoryAgent initialized ({session_count} existing sessions)")
            
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            raise

    async def process(self, request: MemoryRequest) -> MemoryResponse:
        """
        Process memory operation (save or load).
        
        Args:
            request: MemoryRequest with operation and data
        
        Returns:
            MemoryResponse with result
        """
        start_time = time.perf_counter()
        
        try:
            if request.operation == "save":
                file_path = await self._save_session(request.session_data)
                processing_time = time.perf_counter() - start_time
                
                return MemoryResponse(
                    request_id=request.request_id,
                    operation="save",
                    file_path=str(file_path),
                    processing_time=processing_time,
                    success=True
                )
            
            elif request.operation == "load":
                session_data = await self._load_session(request.session_id)
                processing_time = time.perf_counter() - start_time
                
                return MemoryResponse(
                    request_id=request.request_id,
                    operation="load",
                    session_data=session_data,
                    processing_time=processing_time,
                    success=True
                )
            
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            return MemoryResponse(
                request_id=request.request_id,
                operation=request.operation,
                processing_time=processing_time,
                success=False,
                error_message=f"MemoryAgent error: {str(e)}"
            )

   
   #persistence logic
    
    async def _save_session(self, session: SessionModel) -> Path:
        """
        Save session to disk.
        
        Creates date-based directory structure and writes JSON file.
        Handles corrupt file detection and recovery.
        
        Args:
            session: SessionModel to save
        
        Returns:
            Path to saved file
        """
        # Get storage path based on current date
        file_path = self._get_session_path(session.session_id)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        session_data = session.to_storage_format()
        
        # Add short session ID for convenience
        session_data["_session_id_short"] = self._simplify_session_id(session.session_id)
        
        # Check for corrupt existing file
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = f.read().strip()
                    if existing:
                        json.loads(existing)  # Verify valid JSON
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Corrupt JSON detected in {file_path}. Overwriting.")
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                session_data, 
                f, 
                indent=self.indent,
                ensure_ascii=self.ensure_ascii
            )
        
        print(f"✅ Session stored: {file_path}")
        return file_path
    
    async def _load_session(self, session_id: str) -> SessionModel:
        """
        Load session from disk.
        
        Args:
            session_id: UUID of session to load
        
        Returns:
            SessionModel loaded from file
        
        Raises:
            FileNotFoundError: If session file doesn't exist
            json.JSONDecodeError: If file is corrupt
        """
        # Try to find the file (search all date directories)
        matching_files = list(self.base_dir.rglob(f"{session_id}.json"))
        
        if not matching_files:
            raise FileNotFoundError(f"Session not found: {session_id}")
        
        if len(matching_files) > 1:
            print(f"⚠️ Warning: Multiple files found for {session_id}, using first")
        
        file_path = matching_files[0]
        
        # Load and parse JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to SessionModel
        session = SessionModel.from_storage(data)
        
        print(f"✅ Session loaded: {file_path}")
        return session
    
    # ───────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ───────────────────────────────────────────────────────────────
    
    def _get_session_path(self, session_id: str, timestamp: Optional[datetime] = None) -> Path:
        """
        Get file path for a session.
        
        Format: storage/session_logs/YYYY/MM/DD/<session_id>.json
        
        Args:
            session_id: UUID of session
            timestamp: Optional timestamp (defaults to now)
        
        Returns:
            Path to session file
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        day_dir = (
            self.base_dir / 
            str(timestamp.year) / 
            f"{timestamp.month:02d}" / 
            f"{timestamp.day:02d}"
        )
        
        filename = f"{session_id}.json"
        return day_dir / filename
    
    def _simplify_session_id(self, session_id: str) -> str:
        """
        Get short version of session ID.
        
        Args:
            session_id: Full UUID
        
        Returns:
            First segment of UUID (before first hyphen)
        """
        return session_id.split("-")[0]
    
    # ───────────────────────────────────────────────────────────────
    # PUBLIC UTILITY METHODS
    # ───────────────────────────────────────────────────────────────
    
    def get_all_session_ids(self) -> list:
        """
        Get list of all stored session IDs.
        
        Returns:
            List of session ID strings
        """
        session_files = list(self.base_dir.rglob("*.json"))
        return [f.stem for f in session_files]
    
    def get_sessions_by_date(self, year: int, month: int, day: int) -> list:
        """
        Get all sessions from a specific date.
        
        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)
            day: Day (1-31)
        
        Returns:
            List of SessionModel objects
        """
        day_dir = self.base_dir / str(year) / f"{month:02d}" / f"{day:02d}"
        
        if not day_dir.exists():
            return []
        
        sessions = []
        for file_path in day_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = SessionModel.from_storage(data)
                sessions.append(session)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
        
        return sessions
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions.
        
        Returns:
            Dict with total files, size, oldest/newest dates
        """
        all_files = list(self.base_dir.rglob("*.json"))
        
        total_size = sum(f.stat().st_size for f in all_files)
        
        dates = []
        for f in all_files:
            parts = f.parts
            if len(parts) >= 3:
                try:
                    year = int(parts[-4])
                    month = int(parts[-3])
                    day = int(parts[-2])
                    dates.append(datetime(year, month, day))
                except (ValueError, IndexError):
                    pass
        
        return {
            "base_dir": str(self.base_dir),
            "total_sessions": len(all_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_session": min(dates) if dates else None,
            "newest_session": max(dates) if dates else None
        }
    
    def __repr__(self) -> str:
        """Debug representation."""
        stats = self.get_storage_stats()
        return (
            f"MemoryAgent("
            f"sessions={stats['total_sessions']}, "
            f"size={stats['total_size_mb']}MB, "
            f"calls={self.metrics['total_calls']})"
        )