from pathlib import Path
from typing import Dict, Any, List, Optional
from app.tools.base import ToolBase, ToolOutput
from app.runtime.manager import ModelExecutionError
from app.data.ingestion import pixel_box_to_geo


def _measure_pair_change(filepath_a, filepath_b, width, height):
    """Measure genuine pixel-level change between two consecutive epoch images.

    Mirrors ``BiTemporalChangeVQATool._measure_change``: both scenes are reduced to
    a canonical 256x256 RGB grid, the mean absolute per-pixel difference across
    channels is computed, and the highest-change region is localized. Returns
    ``None`` when either pair cannot be read or no meaningful change is present.
    """
    try:
        from PIL import Image
        import numpy as np
        from app.data.ingestion import ImageIngestionService

        service = ImageIngestionService()
        data1, _ = service.read_image_data(Path(filepath_a))
        data2, _ = service.read_image_data(Path(filepath_b))

        def as_rgb(data):
            if data.ndim == 2:
                return np.repeat(data[:, :, None], 3, axis=2)
            if data.shape[2] == 1:
                return np.repeat(data, 3, axis=2)
            return data[:, :, :3]

        img1 = Image.fromarray(as_rgb(data1).astype(np.uint8)).convert("RGB").resize((256, 256))
        img2 = Image.fromarray(as_rgb(data2).astype(np.uint8)).convert("RGB").resize((256, 256))
        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)
        diff = np.mean(np.abs(arr2 - arr1), axis=2)
        threshold = max(18.0, float(np.percentile(diff, 85)))
        active = diff >= threshold
        if not np.any(active):
            return None
        ys, xs = np.where(active)
        x_lo = int(np.percentile(xs, 5))
        x_hi = int(np.percentile(xs, 95))
        y_lo = int(np.percentile(ys, 5))
        y_hi = int(np.percentile(ys, 95))
        x1 = int(x_lo / 256 * width)
        y1 = int(y_lo / 256 * height)
        x2 = max(x1 + 1, int(x_hi / 256 * width))
        y2 = max(y1 + 1, int(y_hi / 256 * height))

        min_size = 8
        if x2 - x1 < min_size:
            cx = (x1 + x2) // 2
            x1 = cx - min_size // 2
            x2 = x1 + min_size
        if y2 - y1 < min_size:
            cy = (y1 + y2) // 2
            y1 = cy - min_size // 2
            y2 = y1 + min_size

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 - x1 < min_size:
            x1 = max(0, x2 - min_size)
            x2 = min(width, x1 + min_size)
        if y2 - y1 < min_size:
            y1 = max(0, y2 - min_size)
            y2 = min(height, y1 + min_size)
        return {"changed_pct": float(np.mean(active) * 100), "box": [x1, y1, x2, y2]}
    except Exception:
        return None


class MultiTemporalSequenceTool(ToolBase):
    """
    Multi-Temporal Sequence Analysis Tool (EarthDial-4B Multi-Temporal Sequence Engine).
    Analyzes sequences of N >= 3 satellite observations (T1 -> T2 -> ... -> TN) across Optical, MS, and SAR
    to identify a progressive timeline of events, cumulative land transformation,
    and cyclical trends with GIS coordinates.
    """


    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        clean_params = self.validate_and_filter_params(parameters)

        images = inputs.get("images", [])
        if len(images) < 3:
            raise ModelExecutionError(
                "MDL_INSUFFICIENT_TEMPORAL_SEQUENCE",
                f"Multi-temporal sequence analysis requires at least 3 co-registered images (T1...TN), got {len(images)}."
            )

        earthdial_key = self.runtime_mgr.earthdial_model_key_for(
            [img.modality for img in images], self.model_key
        )
        self.runtime_mgr.ensure_model_loaded(earthdial_key, self.load_group)

        query = inputs.get("query", "").strip()
        q_lower = query.lower() if query else "summarize temporal sequence events"

        num_steps = len(images)
        w = max(img.width for img in images)
        h = max(img.height for img in images)

        # Detect sequence domain focus
        is_urban = any(k in q_lower for k in ["built-up", "urban", "construction", "building", "infrastructure", "expansion"])
        is_water = any(k in q_lower for k in ["water", "flood", "river", "reservoir", "lake", "inundation"])
        is_veg = any(k in q_lower for k in ["vegetation", "forest", "crop", "deforestation", "agriculture", "canopy"])

        if is_urban:
            focus_label = "urban"
        elif is_water:
            focus_label = "hydrological"
        elif is_veg:
            focus_label = "canopy"
        else:
            focus_label = "general"

        temporal_events: List[Dict[str, Any]] = []
        all_boxes: List[List[int]] = []
        step_narratives: List[str] = []
        model_step_texts: List[Optional[str]] = []

        cumulative_delta = 0.0
        total_steps = len(images) - 1
        measured_steps = 0
        any_model_text = False

        for i in range(len(images) - 1):
            env_a = images[i]
            env_b = images[i + 1]
            step_label = f"T{i+1} → T{i+2}"

            # Measure the genuine pixel difference between consecutive epochs.
            measured = None
            fp_a = getattr(env_a, "filepath", None)
            fp_b = getattr(env_b, "filepath", None)
            if fp_a and fp_b:
                measured = _measure_pair_change(fp_a, fp_b, w, h)

            if measured is not None:
                changed_pct = measured["changed_pct"]
                step_delta = round(changed_pct, 1)
                box = list(measured["box"])
                category = f"Measured Surface Change (focus: {focus_label})"
                desc = (
                    f"Phase {i+1} ({step_label}): Measured surface difference of {changed_pct:.1f}% "
                    f"between observations, localized at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]."
                )
                measured_steps += 1
            else:
                # No genuine measurement: never invent a delta, region, or category.
                step_delta = None
                box = None
                category = "Measurement Unavailable"
                desc = f"Phase {i+1} ({step_label}): Measurement unavailable for this transition."

            # Per-transition model call: compare ONLY this consecutive pair so the
            # model describes each transition rather than collapsing to one generic T1-vs-TN sentence.
            model_text: Optional[str] = None
            if fp_a and fp_b and Path(fp_a).exists() and Path(fp_b).exists():
                prompt = (
                    f"Compare these two co-registered satellite observations of the same area: "
                    f"the earlier observation (T{i+1}) and the later observation (T{i+2}). "
                    f"Concretely describe what changed between them, focusing on new construction, "
                    f"vegetation change, water change, or land transformation. Be concise."
                )
                try:
                    real = self.runtime_mgr.run_earthdial(
                        prompt, [fp_a, fp_b], clean_params, model_key=earthdial_key
                    )
                    if real and real.get("text"):
                        model_text = real["text"].strip() or None
                except Exception as exc:
                    self.runtime_mgr.load_errors[earthdial_key] = f"Inference: {type(exc).__name__}: {exc}"
            if model_text:
                any_model_text = True

            if step_delta is not None:
                cumulative_delta += step_delta
            if box is not None:
                all_boxes.append(box)
            step_narratives.append(desc)
            model_step_texts.append(model_text)

            if measured is not None:
                step_conf = 0.90
                step_basis = "measured_pixel_analysis"
            elif model_text:
                step_conf = 0.85
                step_basis = "nominal_model_estimate"
            else:
                step_conf = None
                step_basis = "not_available"

            temporal_events.append({
                "transition": step_label,
                "from_index": i + 1,
                "to_index": i + 2,
                "from_filename": env_a.filename,
                "to_filename": env_b.filename,
                "category": category,
                "delta_pct": step_delta,
                "cumulative_delta_pct": round(cumulative_delta, 1) if measured_steps > 0 else None,
                "region": box,
                "description": desc,
                "model_description": model_text,
                "confidence": step_conf,
                "confidence_basis": step_basis,
            })

        all_measured = total_steps > 0 and measured_steps == total_steps
        any_measured = measured_steps > 0

        # Compose a genuine multi-temporal narrative: one bullet per consecutive transition,
        # each carrying its own model description (or the measured description).
        header = (
            f"Multi-Temporal Sequence Analysis across {num_steps} sequential satellite acquisitions "
            f"(T1 through T{num_steps}):"
        )
        phase_lines: List[str] = []
        for i, ev in enumerate(temporal_events):
            step_label = ev["transition"]
            box = ev["region"]
            delta = ev["delta_pct"]
            model_text = model_step_texts[i] if i < len(model_step_texts) else None
            if model_text:
                body = model_text
            else:
                desc = ev["description"]
                prefix = f"Phase {i+1} ({step_label}): "
                body = desc[len(prefix):] if desc.startswith(prefix) else desc
            if delta is not None and box is not None:
                suffix = (
                    f"(measured surface difference {delta:.1f}%, "
                    f"region [{box[0]}, {box[1]}, {box[2]}, {box[3]}])."
                )
            else:
                suffix = "(measurement unavailable; no delta or region reported)."
            phase_lines.append(f"• Phase {i+1} ({step_label}): {body} {suffix}")
        phase_block = "\n".join(phase_lines)

        # Closing synthesis: report the real cumulative difference when it can be
        # computed, otherwise say so honestly. Always mentions "cumulative".
        if all_measured:
            mean_delta = cumulative_delta / total_steps if total_steps else 0.0
            synthesis_line = (
                f"Synthesis Trend: across the entire sequence from T1 to T{num_steps}, the mean measured "
                f"surface difference per transition was {mean_delta:.1f}% (cumulative {cumulative_delta:+.1f}%)."
            )
        elif measured_steps == 0:
            synthesis_line = (
                f"Synthesis Trend: across the entire sequence from T1 to T{num_steps}, the cumulative "
                f"surface difference could not be computed because no transition produced a valid "
                f"pixel-difference measurement."
            )
        else:
            synthesis_line = (
                f"Synthesis Trend: across the entire sequence from T1 to T{num_steps}, the cumulative "
                f"surface difference could only be partially computed from {measured_steps} of "
                f"{total_steps} measured transitions ({cumulative_delta:+.1f}%)."
            )

        full_text = header + "\n" + phase_block + "\n\n" + synthesis_line

        if any_measured:
            overall_conf = 0.90
            confidence_basis = "measured_pixel_analysis"
        elif any_model_text:
            overall_conf = 0.85
            confidence_basis = "nominal_model_estimate"
        else:
            overall_conf = None
            confidence_basis = "not_available"

        evidence_items = [
            {
                "type": "temporal_sequence",
                "source_model": self.model_name,
                "score": overall_conf,
                "confidence": overall_conf,
                "confidence_basis": confidence_basis,
                "description": f"Continuous temporal trajectory analyzed across {num_steps} observation epochs.",
                "total_transitions": len(temporal_events),
                "measured_transitions": measured_steps,
                "cumulative_delta_pct": round(cumulative_delta, 1) if any_measured else None,
            }
        ]

        for ev in temporal_events:
            evidence_items.append({
                "type": "temporal_event",
                "source_model": self.model_name,
                "transition": ev["transition"],
                "category": ev["category"],
                "delta_pct": ev["delta_pct"],
                "region": ev["region"],
                "score": ev["confidence"],
                "confidence": ev["confidence"],
                "confidence_basis": ev["confidence_basis"],
                "description": ev["description"],
            })

        return ToolOutput(
            tool_name=self.name,
            model_name=self.model_name,
            text=full_text,
            boxes=all_boxes or None,
            confidence=overall_conf,
            evidence_type="temporal_sequence",
            evidence=evidence_items,
            parameters_used=clean_params,
            metadata={
                "sequence_length": num_steps,
                "temporal_events": temporal_events,
                "cumulative_delta_pct": round(cumulative_delta, 1) if any_measured else None,
                "measured_transitions": measured_steps,
                "confidence_basis": confidence_basis,
                "inference_backend": (
                    "checkpoint" if any_model_text
                    else ("pixel_analysis" if any_measured else "unavailable")
                ),
                "model_step_texts": model_step_texts,
                "model_narrative": " ".join(t for t in model_step_texts if t) or None,
                "earthdial_model_key": earthdial_key,
                "checkpoint_dir": (
                    str(self.runtime_mgr.get_checkpoint_dir(earthdial_key))
                    if self.runtime_mgr.get_checkpoint_dir(earthdial_key)
                    else None
                ),
                "model_key": self.model_key,
            }
        )