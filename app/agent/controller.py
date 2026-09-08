import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.data.ingestion import ImageMetadataEnvelope
from app.registry.store import ToolRegistryStore
from app.agent.classifier import TaskClassifier
from app.agent.validator import AgentInputValidator
from app.agent.planner import ExecutionPlanner
from app.agent.executor import PlanExecutor
from app.agent.aggregator import OutputAggregator, AggregatedResponse
from app.events.stream import EventStream, TraceView
from app.evidence.renderer import EvidenceRenderer
from app.reports.generator import ReportGenerator
from app.storage.sandbox import StorageSandbox


class QueryExecutionResult(BaseModel):
    trace_id: str
    session_id: str
    task: str
    selected_tool: str
    selected_model: str
    answer: str
    confidence: float
    boxes: Optional[List[List[int]]] = None
    evidence_url: Optional[str] = None
    report_url: Optional[str] = None
    trace: TraceView
    duration_ms: float


class AgentController:
    """
    Main Agentic Controller orchestrating query interpretation, input validation,
    specialist tool selection, execution, output aggregation, visual evidence, and reporting.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistryStore] = None,
        sandbox: Optional[StorageSandbox] = None
    ):
        self.sandbox = sandbox or StorageSandbox()
        self.registry = registry or ToolRegistryStore()
        self.planner = ExecutionPlanner(self.registry)
        self.executor = PlanExecutor(self.registry)
        self.evidence_renderer = EvidenceRenderer(self.sandbox)
        self.report_generator = ReportGenerator(self.sandbox)

    def execute_query(
        self,
        session_id: str,
        query: str,
        images: List[ImageMetadataEnvelope],
        parameters: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> QueryExecutionResult:
        tid = trace_id or f"tr_{uuid.uuid4().hex[:12]}"
        trace_dir = self.sandbox.get_trace_dir(session_id)
        stream = EventStream(tid, session_id, log_dir=trace_dir)
        start_time = time.time()

        stream.emit("TRACE_STARTED", "Init", {"query": query, "image_count": len(images)})

        try:
            # Step 1: Task Classification
            stream.start_step("TaskClassification")
            task, task_conf = TaskClassifier.classify(query, images)
            stream.emit(
                "TASK_CLASSIFIED",
                "TaskClassification",
                {"task": task, "confidence": task_conf},
                status="SUCCESS"
            )

            # Step 2: Input Compatibility Validation
            stream.start_step("InputValidation")
            AgentInputValidator.validate(task, images)
            stream.emit(
                "INPUTS_VALIDATED",
                "InputValidation",
                {"validated_task": task, "images_verified": len(images)},
                status="SUCCESS"
            )

            # Step 3: Tool Selection & Planning
            stream.start_step("ToolSelection")
            plan = self.planner.create_plan(task, query, images, parameters)
            selected_step = plan.steps[0]
            stream.emit(
                "TOOL_SELECTED",
                "ToolSelection",
                {
                    "tool_name": selected_step.tool_name,
                    "model_name": selected_step.model_name,
                    "parameters": selected_step.parameters
                },
                status="SUCCESS"
            )

            # Step 4: Tool Execution
            tool_outputs = self.executor.execute(plan, stream)

            # Step 5: Output Aggregation
            stream.start_step("OutputAggregation")
            aggregated: AggregatedResponse = OutputAggregator.aggregate(tool_outputs)
            stream.emit(
                "OUTPUT_AGGREGATED",
                "OutputAggregation",
                {"confidence": aggregated.confidence, "has_boxes": bool(aggregated.boxes)},
                status="SUCCESS"
            )

            # Step 6: Visual Evidence Rendering
            stream.start_step("EvidenceRendering")
            evidence_path = None
            evidence_url = None

            if task == "grounding" and aggregated.boxes:
                evidence_path = self.evidence_renderer.render_bounding_boxes(
                    session_id=session_id,
                    envelope=images[0],
                    boxes=aggregated.boxes,
                    label=query[:25]
                )
            elif task == "change_vqa" and len(images) >= 2:
                evidence_path = self.evidence_renderer.render_bi_temporal_change(
                    session_id=session_id,
                    env_t1=images[0],
                    env_t2=images[1],
                    change_boxes=aggregated.boxes
                )
            elif task == "opt_sar_fusion" and len(images) >= 2:
                # Resolve optical vs SAR order
                env1, env2 = images[0], images[1]
                if env1.modality == "sar":
                    opt_e, sar_e = env2, env1
                else:
                    opt_e, sar_e = env1, env2
                evidence_path = self.evidence_renderer.render_optical_sar_split(
                    session_id=session_id,
                    opt_env=opt_e,
                    sar_env=sar_e
                )

            if evidence_path:
                evidence_url = f"/api/evidence/{evidence_path.name}?session_id={session_id}"
                stream.emit(
                    "EVIDENCE_RENDERED",
                    "EvidenceRendering",
                    {"evidence_file": evidence_path.name, "evidence_url": evidence_url},
                    status="SUCCESS"
                )

            # Step 7: Final Trace Projection
            stream.emit(
                "QUERY_COMPLETED",
                "Finalization",
                {"confidence": aggregated.confidence, "answer_len": len(aggregated.answer)},
                status="SUCCESS"
            )
            trace_view = stream.project_trace_view(query=query)

            # Step 8: Downloadable Audit Report Generation
            stream.start_step("ReportGeneration")
            report_path = self.report_generator.generate_html_report(
                session_id=session_id,
                query=query,
                answer=aggregated.answer,
                images=images,
                trace_view=trace_view,
                evidence_path=evidence_path
            )
            report_url = f"/api/report/{report_path.name}?session_id={session_id}"

            total_ms = (time.time() - start_time) * 1000.0

            return QueryExecutionResult(
                trace_id=tid,
                session_id=session_id,
                task=task,
                selected_tool=aggregated.primary_tool,
                selected_model=aggregated.primary_model,
                answer=aggregated.answer,
                confidence=aggregated.confidence,
                boxes=aggregated.boxes,
                evidence_url=evidence_url,
                report_url=report_url,
                trace=trace_view,
                duration_ms=round(total_ms, 2)
            )

        except Exception as e:
            stream.emit("QUERY_FAILED", "ErrorHandling", {"error": str(e)}, status="ERROR")
            trace_view = stream.project_trace_view(query=query)
            raise e
