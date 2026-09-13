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
from app.agent.aggregator import OutputAggregator, AggregatedResponse, CALIBRATION_TEMPERATURE
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
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    calibration_temperature: float = CALIBRATION_TEMPERATURE
    calibration_method: str = "temperature_scaling"
    uncertainty_flag: bool = False
    conflict_detected: bool = False
    uncertainty_explanation: Optional[str] = None
    is_decomposed: bool = False
    decomposition_reasoning: Optional[str] = None
    subtasks: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_events: Optional[List[Dict[str, Any]]] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    trace_view: Optional[TraceView] = None
    boxes: Optional[List[List[int]]] = None
    evidence_url: Optional[str] = None
    diff_mask_url: Optional[str] = None
    diff_overlay_url: Optional[str] = None
    geojson_url: Optional[str] = None
    geo_boxes: Optional[List[Dict[str, Any]]] = None
    report_url: Optional[str] = None
    duration_ms: float



class AgentController:
    """
    Main Agentic Controller orchestrating explainable 6-phase execution:
    Input Validation → Task Identification → Model Selection → Execution → Evidence Collection → Final Result.
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
            # PHASE 1: Input Validation
            stream.start_step("InputValidation")
            task, task_conf = TaskClassifier.classify(query, images)
            AgentInputValidator.validate(task, images)
            stream.emit(
                "INPUTS_VALIDATED",
                "InputValidation",
                {"image_count": len(images), "status": "COMPATIBLE", "validated_for_task": task},
                status="SUCCESS"
            )

            # PHASE 2: Task Identification & Query Decomposition
            stream.start_step("TaskIdentification")
            plan = self.planner.create_plan(task, query, images, parameters)
            stream.emit(
                "TASK_CLASSIFIED",
                "TaskIdentification",
                {"task": task, "confidence": task_conf},
                status="SUCCESS"
            )
            stream.emit(
                "TASK_IDENTIFIED",
                "TaskIdentification",
                {
                    "task": task,
                    "confidence": task_conf,
                    "is_decomposed": plan.is_decomposed,
                    "decomposition_reasoning": plan.decomposition_reasoning,
                    "subtask_count": len(plan.steps)
                },
                status="SUCCESS"
            )

            # PHASE 3: Model Selection & Workflow Planning
            stream.start_step("ModelSelection")
            models_selected = [s.model_name for s in plan.steps]
            tools_selected = [s.tool_name for s in plan.steps]
            stream.emit(
                "TOOL_SELECTED",
                "ModelSelection",
                {"tool_name": plan.steps[0].tool_name, "model_name": plan.steps[0].model_name},
                status="SUCCESS"
            )
            stream.emit(
                "MODELS_SELECTED",
                "ModelSelection",
                {
                    "models": models_selected,
                    "tools": tools_selected,
                    "steps": [
                        {"id": s.step_id, "title": s.subtask_title, "model": s.model_name, "tool": s.tool_name}
                        for s in plan.steps
                    ]
                },
                status="SUCCESS"
            )

            # PHASE 4: Execution (Specialist Model Inference)
            stream.start_step("Execution")
            tool_outputs = self.executor.execute(plan, stream)
            if task == "opt_sar_fusion":
                stream.emit("MODALITY_FEATURE_EXTRACTED", "Execution", {"modality": "optical"}, status="SUCCESS")
                stream.emit("MODALITY_FEATURE_EXTRACTED", "Execution", {"modality": "sar"}, status="SUCCESS")
                stream.emit("CROSS_MODAL_FUSION", "Execution", {"strategy": "cross_attention"}, status="SUCCESS")
            elif task in ("change_vqa", "temporal_sequence"):
                stream.emit("TEMPORAL_COMPARISON", "Execution", {"time_from": "T1", "time_to": f"T{len(images)}"}, status="SUCCESS")
                stream.emit("CHANGE_EVIDENCE_EXTRACTED", "Execution", {"steps": len(images)}, status="SUCCESS")

            stream.emit(
                "SPECIALISTS_EXECUTED",
                "Execution",
                {"executed_steps": len(tool_outputs), "models": models_selected},
                status="SUCCESS"
            )

            # PHASE 5: Evidence Collection & Multi-Model Fusion
            stream.start_step("EvidenceCollection")
            aggregated: AggregatedResponse = OutputAggregator.aggregate(tool_outputs)
            stream.emit(
                "EVIDENCE_COLLECTED",
                "EvidenceCollection",
                {
                    "total_evidence_items": len(aggregated.evidence),
                    "boxes_detected": len(aggregated.boxes) if aggregated.boxes else 0,
                    "calibrated_confidence": aggregated.confidence,
                    "raw_confidence": aggregated.raw_confidence,
                    "calibration_temperature": aggregated.calibration_temperature,
                    "calibration_method": "temperature_scaling",
                    "uncertainty_flag": aggregated.uncertainty_flag,
                    "conflict_detected": aggregated.conflict_detected,
                    "explanation": aggregated.uncertainty_explanation
                },
                status="SUCCESS"
            )

            # PHASE 6: Final Result Synthesis (Visual Evidence Rendering & Audit Reporting)
            stream.start_step("FinalResult")
            evidence_path = None
            evidence_url = None
            diff_mask_url = None
            diff_overlay_url = None
            geojson_url = None

            # Consolidate spatial boxes from structured evidence if top-level boxes is missing
            if not aggregated.boxes and aggregated.evidence:
                extracted_b = [
                    ev["region"] for ev in aggregated.evidence
                    if ev.get("region") and len(ev["region"]) == 4
                ]
                if extracted_b:
                    aggregated.boxes = extracted_b

            # Render Task-Specific Evidence
            if task == "temporal_sequence" and len(images) >= 3:
                evidence_path = self.evidence_renderer.render_temporal_sequence_filmstrip(
                    session_id=session_id,
                    envelopes=images,
                    boxes=aggregated.boxes
                )
            elif task == "change_vqa" and len(images) >= 2:
                evidence_path = self.evidence_renderer.render_bi_temporal_change(
                    session_id=session_id,
                    env_t1=images[0],
                    env_t2=images[1],
                    change_boxes=aggregated.boxes
                )
                # Compute pixel-level difference heatmap and transparent mask
                diff_res = self.evidence_renderer.render_difference_heatmap(
                    session_id=session_id,
                    env_t1=images[0],
                    env_t2=images[1],
                    change_boxes=aggregated.boxes
                )
                diff_overlay_url = f"/api/evidence/{diff_res['overlay_path'].name}?session_id={session_id}"
                diff_mask_url = f"/api/evidence/{diff_res['mask_path'].name}?session_id={session_id}"
            elif task == "opt_sar_fusion" and len(images) >= 2:
                env1, env2 = images[0], images[1]
                if env1.modality == "sar":
                    opt_e, sar_e = env2, env1
                else:
                    opt_e, sar_e = env1, env2
                evidence_path = self.evidence_renderer.render_optical_sar_split(
                    session_id=session_id,
                    opt_env=opt_e,
                    sar_env=sar_e,
                    boxes=aggregated.boxes
                )
            elif task == "caption":
                # Uniform visual evidence for captioning, but NO fabricated boxes/regions:
                # render the analysed scene as-is; boxes stay None and evidence stays empty.
                evidence_path = self.evidence_renderer.render_analysed_image(
                    session_id=session_id,
                    envelope=images[0],
                    label="Scene under analysis"
                )
            else:
                # Single-image VQA or Grounding: render the analysed scene. Draw boxes
                # only when a specialist genuinely returned a localization; never
                # synthesize a fallback region when localization produced nothing.
                if aggregated.boxes:
                    evidence_path = self.evidence_renderer.render_bounding_boxes(
                        session_id=session_id,
                        envelope=images[0],
                        boxes=aggregated.boxes,
                        label=query[:24] if query else "Analysed Target"
                    )
                else:
                    evidence_path = self.evidence_renderer.render_analysed_image(
                        session_id=session_id,
                        envelope=images[0],
                        label="Scene under analysis"
                    )

            # Generate GeoJSON only when the target image carries real geographic
            # bounds. Pixel boxes are still returned, but no WGS84 geometry is invented.
            if aggregated.boxes:
                target_env = images[-1] if len(images) >= 2 else images[0]
                if target_env is not None and target_env.bounds:
                    geojson_path = self.evidence_renderer.save_geojson_evidence(
                        session_id=session_id,
                        envelope=target_env,
                        boxes=aggregated.boxes,
                        label=f"Spatial Target ({task})",
                        evidence_items=aggregated.evidence
                    )
                    geojson_url = f"/api/evidence/{geojson_path.name}?session_id={session_id}"

            if evidence_path:
                evidence_url = f"/api/evidence/{evidence_path.name}?session_id={session_id}"

            stream.emit(
                "EVIDENCE_RENDERED",
                "FinalResult",
                {
                    "evidence_url": evidence_url,
                    "diff_mask_url": diff_mask_url,
                    "geojson_url": geojson_url,
                    "overlay_file": evidence_path.name if evidence_path else None
                },
                status="SUCCESS"
            )
            # Finalize stream event log with QUERY_COMPLETED
            stream.emit(
                "QUERY_COMPLETED",
                "FinalResult",
                {
                    "status": "COMPLETED",
                    "confidence": aggregated.confidence,
                    "models": models_selected,
                    "answer": aggregated.answer
                },
                status="SUCCESS"
            )

            # Generate Downloadable HTML Audit Report
            trace_view = stream.project_trace_view(query=query)
            report_path = self.report_generator.generate_html_report(
                session_id=session_id,
                query=query,
                answer=aggregated.answer,
                images=images,
                trace_view=trace_view,
                evidence_path=evidence_path,
                evidence_items=aggregated.evidence
            )
            report_url = f"/api/report/{report_path.name}?session_id={session_id}"

            total_ms = (time.time() - start_time) * 1000.0

            # Extract temporal events and geo-boxes if present
            temporal_events = None
            geo_boxes = None
            for out in tool_outputs:
                if "temporal_events" in out.metadata:
                    temporal_events = out.metadata["temporal_events"]
                if "geo_boxes" in out.metadata:
                    geo_boxes = out.metadata["geo_boxes"]

            return QueryExecutionResult(
                trace_id=tid,
                session_id=session_id,
                task=task,
                selected_tool=aggregated.primary_tool,
                selected_model=aggregated.primary_model,
                answer=aggregated.answer,
                confidence=aggregated.confidence,
                raw_confidence=aggregated.raw_confidence,
                calibration_temperature=aggregated.calibration_temperature,
                calibration_method="temperature_scaling",
                uncertainty_flag=aggregated.uncertainty_flag,
                conflict_detected=aggregated.conflict_detected,
                uncertainty_explanation=aggregated.uncertainty_explanation,
                is_decomposed=plan.is_decomposed,
                decomposition_reasoning=plan.decomposition_reasoning,
                subtasks=aggregated.subtasks_summary,
                temporal_events=temporal_events,
                evidence=aggregated.evidence,
                trace=stream.to_human_trace(),
                trace_view=trace_view,
                boxes=aggregated.boxes,
                evidence_url=evidence_url,
                diff_mask_url=diff_mask_url,
                diff_overlay_url=diff_overlay_url,
                geojson_url=geojson_url,
                geo_boxes=geo_boxes,
                report_url=report_url,
                duration_ms=round(total_ms, 2)
            )


        except Exception as e:
            stream.emit("QUERY_FAILED", "ErrorHandling", {"error": str(e)}, status="ERROR")
            raise e
