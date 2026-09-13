from typing import Dict, Any, Optional
from app.tools.base import ToolBase, ToolOutput
from app.runtime.manager import ModelExecutionError
from app.data.ingestion import pixel_box_to_geo, ImageIngestionService
from pathlib import Path


class BiTemporalChangeVQATool(ToolBase):
    """
    Slot S3: Bi-Temporal Change Description & Change-VQA (EarthDial-4B Multi-Image Engine).
    Analyzes paired images from T1 and T2 across Optical, Multispectral, and SAR
    to detect, quantify, and localize temporal land and structural changes with GIS coordinates.
    """

    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        clean_params = self.validate_and_filter_params(parameters)

        images = inputs.get("images", [])
        self.enforce_joint_use(images, required_count=2)

        self.runtime_mgr.ensure_model_loaded(self.model_key, self.load_group)

        query = inputs.get("query", "").strip()
        if not query:
            raise ModelExecutionError("MDL_EMPTY_QUERY", "Change-VQA query cannot be empty.")

        q_lower = query.lower()
        t1_env = images[0]
        t2_env = images[1]
        modality = t2_env.modality

        real_result = None
        if getattr(t1_env, "filepath", None) and getattr(t2_env, "filepath", None):
            try:
                real_result = self.runtime_mgr.run_earthdial(
                    query,
                    [t1_env.filepath, t2_env.filepath],
                    clean_params,
                )
            except Exception as exc:
                self.runtime_mgr.load_errors[self.model_key] = f"Inference: {type(exc).__name__}: {exc}"

        orig_w = max(t1_env.width, t2_env.width)
        orig_h = max(t1_env.height, t2_env.height)

        # Only a genuine pixel-level measurement may produce a change ratio, region,
        # or change type. When measurement is unavailable, report that honestly
        # instead of substituting fabricated values.
        measured_change = self._measure_change(t1_env, t2_env, orig_w, orig_h)

        model_text = None
        if real_result and real_result.get("text"):
            model_text = real_result["text"].strip() or None

        boxes = None
        change_ratio = None
        change_type = None
        conf = None

        if measured_change:
            changed_pct = measured_change["changed_pct"]
            boxes = [measured_change["box"]]
            change_ratio = round(changed_pct / 100.0, 3)
            focus = ""
            if any(k in q_lower for k in ["fire", "burn", "scar", "wildfire"]):
                focus = " Query focus: the measured difference is reported as a candidate burn-scar region."
            elif any(k in q_lower for k in ["flood", "water", "inundation"]):
                focus = " Query focus: the measured difference is reported as a candidate water/inundation region."
            elif any(k in q_lower for k in ["built", "urban", "construction", "building"]):
                focus = " Query focus: the measured difference is reported as a candidate built-up change region."
            text = (
                f"Pixel-aligned comparison between Observation T1 and T2 identifies "
                f"{changed_pct:.1f}% of the aligned scene as materially different. "
                f"The highest-change region is localized in the displayed evidence overlay.{focus}"
            )
            change_type = "measured_surface_difference"
            conf = 0.88
        else:
            text = (
                "Bi-temporal comparison between Observation T1 and Observation T2 could not compute "
                "a pixel-difference measurement for these inputs; no change ratio, change region, or "
                f"change type can be reported. Query: '{query}'."
            )

        if model_text:
            # Keep the genuine EarthDial per-pair observation when the model ran.
            text = (
                text
                + "\n\n\U0001F9E0 Specialist VLM Observation (EarthDial-4B Multi-Modal, checkpoint):\n"
                + model_text
            )

        if measured_change:
            confidence_basis = "measured_pixel_analysis"
            if model_text:
                conf = max(conf, 0.90)
        elif model_text:
            confidence_basis = "nominal_model_estimate"
            conf = 0.90
        else:
            confidence_basis = "not_available"
            conf = None

        # Compute real-world geographic coordinates only when the image is genuinely
        # georeferenced; otherwise keep pixel localization and report no WGS84 geometry.
        geo_boxes = [pixel_box_to_geo(b, t2_env) for b in (boxes or [])]
        real_geo_boxes = [g for g in geo_boxes if g is not None]
        if real_geo_boxes:
            primary_geo = real_geo_boxes[0]["formatted_coords"]
            text += f"\n\n📍 **Geographic Coordinates (WGS84):** `{primary_geo}`"

        change_detected = measured_change is not None
        score = round(conf, 3) if conf is not None else None

        if boxes:
            evidence_items = [
                {
                    "type": "change_region",
                    "source_model": "earthdial-4b",
                    "region": box,
                    "geo_coordinates": gbox["formatted_coords"] if gbox else None,
                    "geo_box": gbox["geo_box"] if gbox else None,
                    "time_from": "T1",
                    "time_to": "T2",
                    "change_detected": change_detected,
                    "change_type": change_type,
                    "change_ratio": change_ratio,
                    "score": score,
                }
                for box, gbox in zip(boxes, geo_boxes)
            ]
        else:
            evidence_items = [
                {
                    "type": "change_region",
                    "source_model": "earthdial-4b",
                    "region": None,
                    "geo_coordinates": None,
                    "geo_box": None,
                    "time_from": "T1",
                    "time_to": "T2",
                    "change_detected": change_detected,
                    "change_type": change_type,
                    "change_ratio": None,
                    "score": score,
                }
            ]

        return ToolOutput(
            tool_name=self.name,
            model_name=self.model_name,
            text=text,
            boxes=boxes,
            confidence=score,
            evidence_type="bi_temporal_pair",
            evidence_ptr=f"{t1_env.image_id}__{t2_env.image_id}",
            evidence=evidence_items,
            parameters_used=clean_params,
            metadata={
                "t1_image_id": t1_env.image_id,
                "t2_image_id": t2_env.image_id,
                "slot": "S3",
                "modality": modality,
                "change_detected": change_detected,
                "change_type": change_type,
                "change_ratio": change_ratio,
                "geo_boxes": real_geo_boxes or None,
                "confidence_basis": confidence_basis,
                "inference_backend": (
                    "checkpoint" if real_result
                    else ("pixel_analysis" if measured_change else "unavailable")
                ),
                "checkpoint_dir": real_result.get("checkpoint_dir") if real_result else (
                    str(self.runtime_mgr.get_checkpoint_dir(self.model_key))
                    if self.runtime_mgr.get_checkpoint_dir(self.model_key)
                    else None
                ),
            }
        )

    @staticmethod
    def _measure_change(env_t1, env_t2, width: int, height: int) -> Optional[Dict[str, Any]]:
        try:
            service = ImageIngestionService()
            data1, _ = service.read_image_data(Path(env_t1.filepath))
            data2, _ = service.read_image_data(Path(env_t2.filepath))
            from PIL import Image
            import numpy as np

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