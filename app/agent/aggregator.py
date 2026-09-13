import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.tools.base import ToolOutput


CALIBRATION_TEMPERATURE = 1.15


class AggregatedResponse(BaseModel):
    answer: str
    boxes: Optional[List[List[int]]] = None
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    calibration_temperature: float = CALIBRATION_TEMPERATURE
    uncertainty_flag: bool = False
    conflict_detected: bool = False
    uncertainty_explanation: Optional[str] = None
    evidence_type: str
    primary_tool: str
    primary_model: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    conflicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    subtasks_summary: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OutputAggregator:
    """
    Combines textual, spatial, and confidence outputs from specialist tool executions,
    performing multi-model evidence fusion, calibrated uncertainty calculation, and conflict flagging.
    """

    @staticmethod
    def calibrate_probability(p: float, temperature: float = CALIBRATION_TEMPERATURE) -> float:
        """Applies real temperature scaling to a raw probability via its logit."""
        eps = 1e-6
        p_clamped = min(max(p, eps), 1.0 - eps)
        logit = math.log(p_clamped / (1.0 - p_clamped))
        scaled = logit / temperature
        return 1.0 / (1.0 + math.exp(-scaled))

    @staticmethod
    def aggregate(outputs: List[ToolOutput]) -> AggregatedResponse:
        if not outputs:
            return AggregatedResponse(
                answer="No output produced by specialist models.",
                boxes=None,
                confidence=None,
                uncertainty_flag=True,
                uncertainty_explanation="Zero specialist outputs returned.",
                evidence_type="none",
                primary_tool="none",
                primary_model="none",
                evidence=[]
            )

        if len(outputs) == 1:
            out = outputs[0]
            raw = out.confidence
            calibrated = OutputAggregator.calibrate_probability(raw) if raw is not None else None
            conf = calibrated
            uncertainty_flag = False
            uncertainty_exp = None
            answer_text = out.text

            # Single-model uncertainty check on the CALIBRATED probability (< 0.65 or honesty fallback)
            if conf is not None and conf < 0.65:
                uncertainty_flag = True
                uncertainty_exp = f"Specialist confidence ({round(conf * 100)}%) is below certainty threshold (65%). Ambient noise or feature ambiguity present."
                answer_text = f"⚠️ [Uncertainty Flag - Low Model Confidence ({round(conf * 100)}%)]\n{uncertainty_exp}\n\n{out.text}"
            elif out.metadata.get("honesty_gate_triggered"):
                uncertainty_flag = True
                uncertainty_exp = "Honest fallback triggered: target features not reliably detected in imagery."

            return AggregatedResponse(
                answer=answer_text,
                boxes=out.boxes,
                confidence=calibrated,
                raw_confidence=raw,
                calibration_temperature=CALIBRATION_TEMPERATURE,
                uncertainty_flag=uncertainty_flag,
                conflict_detected=False,
                uncertainty_explanation=uncertainty_exp,
                evidence_type=out.evidence_type,
                primary_tool=out.tool_name,
                primary_model=out.model_name,
                evidence=out.evidence,
                subtasks_summary=[{
                    "tool": out.tool_name,
                    "model": out.model_name,
                    "confidence": out.confidence,
                    "evidence_count": len(out.evidence)
                }],
                metadata=out.metadata
            )

        # Multi-model evidence fusion across multiple specialist steps
        primary = outputs[0]
        supporting: List[Dict[str, Any]] = []
        conflicting: List[Dict[str, Any]] = []
        all_evidence: List[Dict[str, Any]] = list(primary.evidence)
        all_boxes: List[List[int]] = list(primary.boxes) if primary.boxes else []
        subtasks_summary: List[Dict[str, Any]] = []

        conflict_detected = False
        conflict_reasons: List[str] = []

        # Record primary subtask
        subtasks_summary.append({
            "tool": primary.tool_name,
            "model": primary.model_name,
            "confidence": primary.confidence,
            "evidence_count": len(primary.evidence)
        })

        for i, secondary in enumerate(outputs[1:], start=2):
            if secondary.boxes:
                all_boxes.extend(secondary.boxes)

            subtasks_summary.append({
                "tool": secondary.tool_name,
                "model": secondary.model_name,
                "confidence": secondary.confidence,
                "evidence_count": len(secondary.evidence)
            })

            # Check for conflict on CALIBRATED confidences: significant variance (>0.30)
            step_conflict = False
            primary_cal = OutputAggregator.calibrate_probability(primary.confidence) if primary.confidence is not None else None
            secondary_cal = OutputAggregator.calibrate_probability(secondary.confidence) if secondary.confidence is not None else None
            if primary_cal is not None and secondary_cal is not None:
                if abs(primary_cal - secondary_cal) > 0.30:
                    step_conflict = True
                    conflict_reasons.append(
                        f"Confidence divergence between {primary.model_name} ({round(primary_cal * 100)}%) "
                        f"and {secondary.model_name} ({round(secondary_cal * 100)}%)."
                    )

            # Check honesty gate trigger
            sec_meta = secondary.metadata or {}
            if sec_meta.get("honesty_gate_triggered"):
                step_conflict = True
                conflict_reasons.append(f"{secondary.model_name} triggered an honesty rejection gate.")

            # Modality discrepancy check (e.g. SAR detects targets where optical was obscured)
            if "not detected" in primary.text.lower() and "detected" in secondary.text.lower():
                step_conflict = True
                conflict_reasons.append(
                    f"{secondary.model_name} identified features obscured or omitted in {primary.model_name}."
                )

            if step_conflict:
                conflict_detected = True

            for ev_item in secondary.evidence:
                all_evidence.append(ev_item)
                if step_conflict:
                    conflicting.append(ev_item)
                else:
                    supporting.append(ev_item)

        # Average raw and calibrated confidence across valid outputs
        valid_confs = [o.confidence for o in outputs if o.confidence is not None]
        raw_confidence = round(sum(valid_confs) / len(valid_confs), 3) if valid_confs else None
        calibrated_confs = [OutputAggregator.calibrate_probability(c) for c in valid_confs]
        avg_conf = round(sum(calibrated_confs) / len(calibrated_confs), 3) if calibrated_confs else None

        uncertainty_flag = conflict_detected or (avg_conf is not None and avg_conf < 0.65)
        uncertainty_exp = "; ".join(conflict_reasons) if conflict_reasons else (
            "Aggregated confidence is below certainty threshold." if uncertainty_flag else None
        )

        # Synthesize multi-model answer text
        sections: List[str] = []

        if uncertainty_flag:
            warning_header = (
                "⚠️ [Uncertainty & Multi-Model Discrepancy Flag]\n"
                f"{uncertainty_exp or 'Discrepancy observed between specialist observations. Manual analyst audit recommended.'}"
            )
            sections.append(warning_header)

        # Section 1: Subtask & Specialist breakdown
        for idx, out in enumerate(outputs, start=1):
            sections.append(f"Specialist {idx} ({out.model_name} via `{out.tool_name}`):\n{out.text}")

        if supporting:
            sec_models = ", ".join(sorted(list({ev.get("source_model", "Specialist") for ev in supporting})))
            sections.append(f"Supporting Evidence ({sec_models}):\n" + "\n".join(f"• {ev.get('description', '')}" for ev in supporting if ev.get("description")))

        if conflicting:
            conf_models = ", ".join(sorted(list({ev.get("source_model", "Specialist") for ev in conflicting})))
            sections.append(f"Conflicting Evidence / Discrepancies ({conf_models}):\n" + "\n".join(f"• {ev.get('description', '')}" for ev in conflicting if ev.get("description")))

        # Section 2: Unified Multi-Model Evidence Synthesis
        models_involved = " & ".join(dict.fromkeys(o.model_name for o in outputs))
        total_ev = len(all_evidence)
        box_count = len(all_boxes)
        consensus_text = (
            f"Multi-Model Consensus ({models_involved}):\n"
            f"Integrated reasoning across {len(outputs)} specialist models successfully synthesized "
            f"{total_ev} evidence features and {box_count} localized regions."
        )
        if conflict_detected:
            consensus_text += " Note that cross-modal discrepancies exist as detailed above."
        sections.append(consensus_text)

        composed_text = "\n\n".join(sections)

        return AggregatedResponse(
            answer=composed_text,
            boxes=all_boxes if all_boxes else None,
            confidence=avg_conf,
            raw_confidence=raw_confidence,
            calibration_temperature=CALIBRATION_TEMPERATURE,
            uncertainty_flag=uncertainty_flag,
            conflict_detected=conflict_detected,
            uncertainty_explanation=uncertainty_exp,
            evidence_type=primary.evidence_type,
            primary_tool=primary.tool_name,
            primary_model=primary.model_name,
            evidence=all_evidence,
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            subtasks_summary=subtasks_summary,
            metadata={
                "multi_model_fusion": True,
                "tool_count": len(outputs),
                "conflict_detected": conflict_detected,
                "models_involved": list(dict.fromkeys(o.model_name for o in outputs))
            }
        )
