from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.data.ingestion import ImageMetadataEnvelope
from app.registry.store import ToolRegistryStore


class ExecutionStep(BaseModel):
    step_id: str
    tool_name: str
    model_name: str
    task: str
    subtask_title: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    plan_id: str
    task: str
    is_decomposed: bool = False
    decomposition_reasoning: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)


class ExecutionPlanner:
    """
    Constructs execution plans and decomposes composite questions into
    ordered subtasks assigned to appropriate specialist models.
    """

    TASK_TOOL_MAP = {
        "vqa": "rs-vqa",
        "caption": "rs-caption",
        "grounding": "rs-ground",
        "change_vqa": "change-vqa",
        "opt_sar_fusion": "opt-sar-fusion",
        "temporal_sequence": "temporal-sequence"
    }

    def __init__(self, registry: ToolRegistryStore):
        self.registry = registry

    def create_plan(
        self,
        task: str,
        query: str,
        images: List[ImageMetadataEnvelope],
        user_parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        q_lower = query.lower().strip()
        steps: List[ExecutionStep] = []
        is_decomposed = False
        decomposition_reasoning = None

        primary_tool_name = self.TASK_TOOL_MAP.get(task, "rs-vqa")
        primary_desc = self.registry.get_descriptor(primary_tool_name)
        if not primary_desc:
            raise ValueError(f"No registered tool found for task '{task}' ({primary_tool_name})")

        primary_tool = self.registry.get_tool(primary_tool_name)
        clean_params = primary_tool.validate_and_filter_params(user_parameters)

        step_inputs = {
            "query": query,
            "images": images,
            "modality": images[0].modality if images else "optical",
            "envelope": images[0] if images else None
        }

        # --- QUERY DECOMPOSITION LOGIC ---
        # 1. Composite: Grounding + Semantic Description / Counting
        has_grounding_intent = any(k in q_lower for k in ["highlight", "locate", "box", "draw", "bounding", "where is", "where are", "pinpoint", "mark"])
        has_vqa_intent = any(k in q_lower for k in ["how many", "count", "describe", "what type", "classify", "what is", "assess", "condition"])

        # 2. Composite: Cross-Modal SAR Backscatter + Temporal Change
        has_sar_intent = any(k in q_lower for k in ["sar", "radar", "backscatter", "penetrate", "cross-modal"])
        has_change_intent = any(k in q_lower for k in ["changed", "between", "before and after", "t1", "t2", "expansion", "growth", "increased"])

        if task == "grounding" and has_vqa_intent:
            is_decomposed = True
            decomposition_reasoning = (
                "Query requires spatial localization and attribute reasoning. Decomposed into "
                "Subtask 1: Region Grounding via EarthDial-4B Grounding Head, and "
                "Subtask 2: Visual Feature Attribute Answering via EarthDial-4B VQA."
            )
            # Step 1: Grounding
            steps.append(ExecutionStep(
                step_id="step_subtask_01_ground",
                tool_name="rs-ground",
                model_name=self.registry.get_descriptor("rs-ground")["model_name"],
                task="grounding",
                subtask_title="Spatial Region Localization",
                inputs=step_inputs,
                parameters=self.registry.get_tool("rs-ground").validate_and_filter_params(user_parameters)
            ))
            # Step 2: VQA
            steps.append(ExecutionStep(
                step_id="step_subtask_02_vqa",
                tool_name="rs-vqa",
                model_name=self.registry.get_descriptor("rs-vqa")["model_name"],
                task="vqa",
                subtask_title="Visual Feature Attribute Reasoning",
                inputs=step_inputs,
                parameters=self.registry.get_tool("rs-vqa").validate_and_filter_params(user_parameters)
            ))

        elif (task == "change_vqa" or task == "opt_sar_fusion") and has_sar_intent and has_change_intent and len(images) >= 2:
            is_decomposed = True
            decomposition_reasoning = (
                "Query fuses all-weather radar backscatter verification with temporal change analysis. "
                "Decomposed into Subtask 1: Cross-Modal Feature Fusion (DOFA-Large), followed by "
                "Subtask 2: Multi-Modal Bi-Temporal Change Verification (EarthDial-4B)."
            )
            dofa_desc = self.registry.get_descriptor("opt-sar-fusion")
            change_desc = self.registry.get_descriptor("change-vqa")
            if dofa_desc and change_desc:
                steps.append(ExecutionStep(
                    step_id="step_subtask_01_fusion",
                    tool_name="opt-sar-fusion",
                    model_name=dofa_desc["model_name"],
                    task="opt_sar_fusion",
                    subtask_title="Cross-Modal Optical/Multi-Spectral-SAR Backscatter Extraction",
                    inputs=step_inputs,
                    parameters=self.registry.get_tool("opt-sar-fusion").validate_and_filter_params(user_parameters)
                ))
                steps.append(ExecutionStep(
                    step_id="step_subtask_02_change",
                    tool_name="change-vqa",
                    model_name=change_desc["model_name"],
                    task="change_vqa",
                    subtask_title="Bi-Temporal Change Quantification",
                    inputs=step_inputs,
                    parameters=self.registry.get_tool("change-vqa").validate_and_filter_params(user_parameters)
                ))

        elif task == "caption" and has_grounding_intent:
            is_decomposed = True
            decomposition_reasoning = (
                "Query requests comprehensive land cover scene description alongside specific target localization. "
                "Decomposed into Subtask 1: Scene Captioning (EarthDial-4B) and Subtask 2: Target Grounding (EarthDial-4B)."
            )
            steps.append(ExecutionStep(
                step_id="step_subtask_01_caption",
                tool_name="rs-caption",
                model_name=self.registry.get_descriptor("rs-caption")["model_name"],
                task="caption",
                subtask_title="Global Land-Cover Captioning",
                inputs=step_inputs,
                parameters=self.registry.get_tool("rs-caption").validate_and_filter_params(user_parameters)
            ))
            steps.append(ExecutionStep(
                step_id="step_subtask_02_ground",
                tool_name="rs-ground",
                model_name=self.registry.get_descriptor("rs-ground")["model_name"],
                task="grounding",
                subtask_title="Key Feature Bounding Box Grounding",
                inputs=step_inputs,
                parameters=self.registry.get_tool("rs-ground").validate_and_filter_params(user_parameters)
            ))

        elif task == "temporal_sequence" and has_grounding_intent and len(images) >= 3:
            is_decomposed = True
            decomposition_reasoning = (
                "Query asks for cumulative temporal sequence analysis as well as localization of highest change zones. "
                "Decomposed into Subtask 1: Multi-Epoch Timeline Progression (EarthDial-4B Sequence), and "
                "Subtask 2: Latest-Epoch Target Grounding (EarthDial-4B)."
            )
            steps.append(ExecutionStep(
                step_id="step_subtask_01_seq",
                tool_name="temporal-sequence",
                model_name=self.registry.get_descriptor("temporal-sequence")["model_name"],
                task="temporal_sequence",
                subtask_title="Multi-Epoch Sequence & Event Trend Extraction",
                inputs=step_inputs,
                parameters=self.registry.get_tool("temporal-sequence").validate_and_filter_params(user_parameters)
            ))
            steps.append(ExecutionStep(
                step_id="step_subtask_02_ground",
                tool_name="rs-ground",
                model_name=self.registry.get_descriptor("rs-ground")["model_name"],
                task="grounding",
                subtask_title="Epoch TN Anomaly Grounding",
                inputs={**step_inputs, "envelope": images[-1], "images": [images[-1]]},
                parameters=self.registry.get_tool("rs-ground").validate_and_filter_params(user_parameters)
            ))

        # Default: Single atomic specialist task
        if not steps:
            steps.append(ExecutionStep(
                step_id=f"step_{task}_01",
                tool_name=primary_tool_name,
                model_name=primary_desc["model_name"],
                task=task,
                subtask_title="Atomic Specialist Inference",
                inputs=step_inputs,
                parameters=clean_params
            ))

        return ExecutionPlan(
            plan_id=f"plan_{task}",
            task=task,
            is_decomposed=is_decomposed,
            decomposition_reasoning=decomposition_reasoning,
            steps=steps
        )
