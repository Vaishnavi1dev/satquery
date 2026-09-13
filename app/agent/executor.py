import time
from typing import List
from app.agent.planner import ExecutionPlan
from app.registry.store import ToolRegistryStore
from app.tools.base import ToolOutput
from app.events.stream import EventStream
from app.runtime.manager import ModelExecutionError


class PlanExecutor:
    """Executes execution plan steps via the tool registry and records structured trace events."""

    def __init__(self, registry: ToolRegistryStore):
        self.registry = registry

    def execute(self, plan: ExecutionPlan, event_stream: EventStream) -> List[ToolOutput]:
        outputs: List[ToolOutput] = []

        for step in plan.steps:
            event_stream.start_step(step.step_id)
            tool = self.registry.get_tool(step.tool_name)
            if not tool:
                err = f"Specialist tool '{step.tool_name}' not available in registry."
                event_stream.emit("TOOL_FAILED", step.step_id, {"error": err}, status="ERROR")
                raise ModelExecutionError("MDL_TOOL_NOT_FOUND", err)

            event_stream.emit(
                "TOOL_INVOKED",
                step.step_id,
                {
                    "tool_name": step.tool_name,
                    "model_name": step.model_name,
                    "subtask_title": step.subtask_title,
                    "stage": "Model Ingestion & Pre-flight",
                    "action": f"Ingesting imagery & initializing {step.model_name} tensor pipeline",
                    "parameters": step.parameters
                },
                status="INFO"
            )

            try:
                out = tool.invoke(step.inputs, parameters=step.parameters)
                outputs.append(out)

                event_stream.emit(
                    "TOOL_COMPLETED",
                    step.step_id,
                    {
                        "tool_name": step.tool_name,
                        "model_name": step.model_name,
                        "subtask_title": step.subtask_title,
                        "stage": "Model Inference Execution",
                        "action": f"{step.subtask_title or 'Specialist Inference'}: Neural reasoning & spatial feature extraction",
                        "confidence": out.confidence,
                        "boxes_count": len(out.boxes) if out.boxes else 0,
                        "evidence_type": out.evidence_type
                    },
                    status="SUCCESS"
                )
            except Exception as e:
                event_stream.emit("TOOL_FAILED", step.step_id, {"error": str(e)}, status="ERROR")
                raise

        return outputs
