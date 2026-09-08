from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.tools.base import ToolOutput


class AggregatedResponse(BaseModel):
    answer: str
    boxes: Optional[List[List[int]]] = None
    confidence: float
    evidence_type: str
    primary_tool: str
    primary_model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OutputAggregator:
    """Combines textual, spatial, and confidence outputs from specialist tool executions."""

    @staticmethod
    def aggregate(outputs: List[ToolOutput]) -> AggregatedResponse:
        if not outputs:
            return AggregatedResponse(
                answer="No output produced by specialist models.",
                boxes=None,
                confidence=0.0,
                evidence_type="none",
                primary_tool="none",
                primary_model="none"
            )

        if len(outputs) == 1:
            out = outputs[0]
            return AggregatedResponse(
                answer=out.text,
                boxes=out.boxes,
                confidence=out.confidence,
                evidence_type=out.evidence_type,
                primary_tool=out.tool_name,
                primary_model=out.model_name,
                metadata=out.metadata
            )

        # Merge multiple tool outputs (e.g. captioning + grounding chain)
        combined_text = "\n\n".join([o.text for o in outputs])
        all_boxes = []
        for o in outputs:
            if o.boxes:
                all_boxes.extend(o.boxes)

        avg_conf = sum(o.confidence for o in outputs) / len(outputs)

        return AggregatedResponse(
            answer=combined_text,
            boxes=all_boxes if all_boxes else None,
            confidence=round(avg_conf, 3),
            evidence_type=outputs[0].evidence_type,
            primary_tool=outputs[0].tool_name,
            primary_model=outputs[0].model_name,
            metadata={"multi_tool_execution": True, "tool_count": len(outputs)}
        )
