import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    event_id: str
    trace_id: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    step_name: str
    status: str = "INFO"  # INFO, SUCCESS, ERROR, WARN
    payload: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None


class TraceStep(BaseModel):
    step_name: str
    status: str
    timestamp: str
    tool_name: Optional[str] = None
    model_name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None


class TraceView(BaseModel):
    trace_id: str
    session_id: str
    query: str
    task: Optional[str] = None
    selected_tool: Optional[str] = None
    selected_model: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    total_duration_ms: float = 0.0
    status: str = "RUNNING"  # RUNNING, COMPLETED, FAILED
    steps: List[TraceStep] = Field(default_factory=list)
    confidence: Optional[float] = None
    error_message: Optional[str] = None


class EventStream:
    """Manages structured JSONL event logging and trace generation for an execution."""

    def __init__(self, trace_id: str, session_id: str, log_dir: Optional[Path] = None):
        self.trace_id = trace_id
        self.session_id = session_id
        self.log_dir = log_dir
        self.events: List[TraceEvent] = []
        self._start_time = time.time()
        self._step_start_times: Dict[str, float] = {}

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"{self.trace_id}.jsonl"
        else:
            self.log_file = None

    def start_step(self, step_name: str):
        self._step_start_times[step_name] = time.time()

    def emit(self, event_type: str, step_name: str, payload: Dict[str, Any] = None, status: str = "INFO"):
        now = time.time()
        start = self._step_start_times.get(step_name, now)
        duration_ms = (now - start) * 1000.0

        event = TraceEvent(
            event_id=f"evt_{len(self.events) + 1:04d}",
            trace_id=self.trace_id,
            session_id=self.session_id,
            event_type=event_type,
            step_name=step_name,
            status=status,
            payload=payload or {},
            duration_ms=round(duration_ms, 2)
        )
        self.events.append(event)

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

        return event

    def project_trace_view(self, query: str = "") -> TraceView:
        steps: List[TraceStep] = []
        task = None
        selected_tool = None
        selected_model = None
        overall_status = "RUNNING"
        error_msg = None
        confidence = None

        for evt in self.events:
            p = evt.payload
            if evt.event_type == "TASK_CLASSIFIED":
                task = p.get("task")
            elif evt.event_type == "TOOL_SELECTED":
                selected_tool = p.get("tool_name")
                selected_model = p.get("model_name")
            elif evt.event_type == "QUERY_COMPLETED":
                overall_status = "COMPLETED"
                confidence = p.get("confidence")
            elif evt.event_type == "QUERY_FAILED":
                overall_status = "FAILED"
                error_msg = p.get("error")

            step = TraceStep(
                step_name=evt.step_name,
                status=evt.status,
                timestamp=evt.timestamp,
                tool_name=p.get("tool_name"),
                model_name=p.get("model_name"),
                parameters=p.get("parameters", {}),
                details=p,
                duration_ms=evt.duration_ms
            )
            steps.append(step)

        total_ms = (time.time() - self._start_time) * 1000.0

        created_at = self.events[0].timestamp if self.events else datetime.now(timezone.utc).isoformat()
        completed_at = self.events[-1].timestamp if overall_status in ("COMPLETED", "FAILED") else None

        return TraceView(
            trace_id=self.trace_id,
            session_id=self.session_id,
            query=query,
            task=task,
            selected_tool=selected_tool,
            selected_model=selected_model,
            created_at=created_at,
            completed_at=completed_at,
            total_duration_ms=round(total_ms, 2),
            status=overall_status,
            steps=steps,
            confidence=confidence,
            error_message=error_msg
        )
