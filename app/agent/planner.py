from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.data.ingestion import ImageMetadataEnvelope
from app.registry.store import ToolRegistryStore


class ExecutionStep(BaseModel):
    step_id: str
    tool_name: str
    model_name: str
    task: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    plan_id: str
    task: str
    steps: List[ExecutionStep] = Field(default_factory=list)


class ExecutionPlanner:
    """Constructs execution plans based on task label and registry descriptors."""

    TASK_TOOL_MAP = {
        "vqa": "rs-vqa",
        "caption": "rs-caption",
        "grounding": "rs-ground",
        "change_vqa": "change-vqa",
        "opt_sar_fusion": "opt-sar-fusion"
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
        tool_name = self.TASK_TOOL_MAP.get(task, "rs-vqa")
        desc = self.registry.get_descriptor(tool_name)
        if not desc:
            raise ValueError(f"No registered tool found for task '{task}' ({tool_name})")

        tool = self.registry.get_tool(tool_name)
        clean_params = tool.validate_and_filter_params(user_parameters)

        step_inputs = {
            "query": query,
            "images": images,
            "modality": images[0].modality if images else "optical",
            "envelope": images[0] if images else None
        }

        step = ExecutionStep(
            step_id=f"step_{task}_01",
            tool_name=tool_name,
            model_name=desc["model_name"],
            task=task,
            inputs=step_inputs,
            parameters=clean_params
        )

        return ExecutionPlan(
            plan_id=f"plan_{task}",
            task=task,
            steps=[step]
        )
