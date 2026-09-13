import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import numpy as np


def _json_default(obj: Any):
    """JSON fallback for numpy scalars/arrays and other non-JSON payload values."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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
    execution_trace: List[str] = Field(default_factory=list)
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
        self._lock = threading.Lock()
        self._event_seq = 0

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

        # Serialize the whole read-modify-write under a lock so concurrent emits
        # cannot reuse an event id, interleave, or corrupt JSONL lines.
        with self._lock:
            self._event_seq += 1
            event = TraceEvent(
                event_id=f"evt_{self._event_seq:04d}",
                trace_id=self.trace_id,
                session_id=self.session_id,
                event_type=event_type,
                step_name=step_name,
                status=status,
                payload=payload or {},
                duration_ms=round(duration_ms, 2)
            )

            # Persist first: a failed write must not leave the event half-recorded
            # in memory or a truncated line on disk.
            if self.log_file:
                line = json.dumps(event.model_dump(mode="python"), default=_json_default)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

            self.events.append(event)

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

        human_trace = self.to_human_trace()

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
            execution_trace=human_trace,
            confidence=confidence,
            error_message=error_msg
        )

    def to_human_trace(self) -> List[str]:
        """Converts internal structured trace events into human-readable execution steps."""
        trace_lines = []
        for evt in self.events:
            p = evt.payload
            et = evt.event_type
            if et == "TRACE_STARTED":
                count = p.get("image_count", 1)
                trace_lines.append(f"Session initialized with {count} input image{'s' if count != 1 else ''}")
            elif et == "TASK_CLASSIFIED":
                task = p.get("task", "")
                if task == "opt_sar_fusion":
                    trace_lines.append("Detected cross-modal optical/multi-spectral + SAR pair")
                elif task == "change_vqa":
                    trace_lines.append("Detected bi-temporal pair")
                elif task == "grounding":
                    trace_lines.append("Detected text-guided region grounding task")
                elif task == "caption":
                    trace_lines.append("Detected remote sensing scene captioning task")
                else:
                    trace_lines.append(f"Detected visual question answering ({task})")
            elif et == "INPUTS_VALIDATED":
                trace_lines.append("Input validated")
            elif et == "TOOL_SELECTED":
                model_name = p.get("model_name", p.get("tool_name", "specialist"))
                trace_lines.append(f"Selected {model_name}")
            elif et == "TOOL_INVOKED":
                model_name = p.get("model_name", p.get("tool_name", "tool"))
                trace_lines.append(f"Executed specialist model: {model_name}")
            elif et == "MODALITY_FEATURE_EXTRACTED":
                modality = p.get("modality", "sensor")
                trace_lines.append(f"Extracted {modality} features")
            elif et == "CROSS_MODAL_FUSION":
                trace_lines.append("Performed cross-modal fusion")
            elif et == "TEMPORAL_COMPARISON":
                trace_lines.append("Compared T1 and T2")
            elif et == "CHANGE_EVIDENCE_EXTRACTED":
                trace_lines.append("Extracted change evidence")
            elif et == "OUTPUT_AGGREGATED":
                trace_lines.append("Aggregated specialist outputs and structured evidence")
            elif et == "EVIDENCE_RENDERED":
                trace_lines.append("Rendered visual evidence overlay")
            elif et == "QUERY_COMPLETED":
                trace_lines.append("Generated final result")
            elif et == "QUERY_FAILED":
                trace_lines.append(f"Execution failed: {p.get('error', 'Unknown error')}")

        # Deduplicate consecutive identical messages
        deduped = []
        for line in trace_lines:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return deduped
