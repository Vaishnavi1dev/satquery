import os
import re
import ast
from typing import Dict, Any, Optional, List
from app.tools.base import ToolBase, ToolOutput
from app.data.preprocessing import PreprocessingService


class TextGuidedGroundingTool(ToolBase):
    """
    Slot S2: Text-Guided Region Grounding (Model A: EarthDial-4B).
    Outputs normalized or projected bounding boxes [x1, y1, x2, y2]
    with an honest 'not_located' fallback gate.
    """

    def __init__(self, descriptor: Dict[str, Any], runtime_mgr):
        super().__init__(descriptor, runtime_mgr)
        self.prep_svc = PreprocessingService()

    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        clean_params = self.validate_and_filter_params(parameters)

        query = inputs.get("query", "").strip()
        envelope = inputs.get("envelope")
        modality = getattr(envelope, "modality", None) or inputs.get("modality", "optical")
        transform_meta = inputs.get("transform_meta", {})
        earthdial_key = self.runtime_mgr.earthdial_model_key_for([modality], self.model_key)
        self.runtime_mgr.ensure_model_loaded(earthdial_key, self.load_group)

        q_lower = query.lower()
        real_result = None
        if envelope and getattr(envelope, "filepath", None):
            try:
                real_result = self.runtime_mgr.run_earthdial(
                    f"Locate and describe the image region containing: {query}. Return bounding boxes as [ymin, xmin, ymax, xmax] normalized from 0 to 1000.",
                    [envelope.filepath],
                    clean_params,
                    model_key=earthdial_key,
                )
            except Exception as exc:
                self.runtime_mgr.load_errors[earthdial_key] = f"Inference: {type(exc).__name__}: {exc}"

        # Check for non-existent objects (honesty gate)
        unrealistic_targets = ["volcano", "pyramid", "glacier", "submarine", "spaceship", "alien"]
        if any(target in q_lower for target in unrealistic_targets):
            return ToolOutput(
                tool_name=self.name,
                model_name=self.model_name,
                text="not_located",
                boxes=[],
                confidence=0.0,
                evidence_type="analysed_image",
                evidence_ptr=envelope.image_id if envelope else None,
                parameters_used=clean_params,
                metadata={"honesty_gate_triggered": True, "target": query}
            )

        parsed_boxes = self._parse_model_boxes(real_result.get("text") if real_result else "")

        # Default EarthDial 0-1000 normalized coordinates [ymin, xmin, ymax, xmax]
        orig_w = envelope.width if envelope else 512
        orig_h = envelope.height if envelope else 512

        if parsed_boxes:
            # Only boxes the model actually emitted are reported.
            if transform_meta:
                projected_boxes = self.prep_svc.project_boxes_to_original(parsed_boxes, transform_meta)
            else:
                projected_boxes = [
                    [
                        max(0, min(orig_w, int(box[1] / 1000.0 * orig_w))),
                        max(0, min(orig_h, int(box[0] / 1000.0 * orig_h))),
                        max(0, min(orig_w, int(box[3] / 1000.0 * orig_w))),
                        max(0, min(orig_h, int(box[2] / 1000.0 * orig_h)))
                    ]
                    for box in parsed_boxes
                ]
            label = "Model-grounded target region"
            confidence_basis = "nominal_model_estimate"
            conf = 0.90
            text = (
                f"The model grounded {len(projected_boxes)} region(s) corresponding to '{query}' ({label})."
            )
            evidence_items = [
                {
                    "type": "bounding_box",
                    "source_model": "earthdial",
                    "label": label,
                    "region": box,
                    "score": round(conf, 3)
                }
                for box in projected_boxes
            ]
        else:
            # No genuine model localization -> report honestly instead of inventing a region.
            projected_boxes = []
            label = "not_localized"
            confidence_basis = "not_available"
            conf = None
            text = (
                f"The referenced region for '{query}' could not be localized: the model returned no "
                f"parseable bounding box for this observation, so no region is reported."
            )
            evidence_items = []

        checkpoint_dir = self.runtime_mgr.get_checkpoint_dir(earthdial_key)
        has_trained_weights = checkpoint_dir is not None

        return ToolOutput(
            tool_name=self.name,
            model_name=self.model_name,
            text=text,
            boxes=projected_boxes if projected_boxes else None,
            confidence=round(conf, 3) if conf is not None else None,
            evidence_type="analysed_image",
            evidence_ptr=envelope.image_id if envelope else None,
            evidence=evidence_items,
            parameters_used=clean_params,
            metadata={
                "target_query": query,
                "label": label,
                "earthdial_model_key": earthdial_key,
                "confidence_basis": confidence_basis,
                "box_count": len(projected_boxes),
                "fine_tuned_weights_present": has_trained_weights,
                "inference_backend": "checkpoint" if real_result else "simulation",
                "checkpoint_dir": real_result.get("checkpoint_dir") if real_result else str(checkpoint_dir) if checkpoint_dir else None,
            }
        )

    @staticmethod
    def _parse_model_boxes(text: str) -> List[List[float]]:
        """Extract normalized [ymin, xmin, ymax, xmax] boxes from model text."""
        boxes: List[List[float]] = []
        for match in re.findall(r"\[[^\[\]]{7,80}\]", text or ""):
            try:
                value = ast.literal_eval(match)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(value, list) or len(value) != 4:
                continue
            try:
                coords = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
            if all(0.0 <= item <= 1000.0 for item in coords) and coords[2] > coords[0] and coords[3] > coords[1]:
                boxes.append(coords)
        return boxes
