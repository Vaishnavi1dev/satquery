from typing import List, Tuple
from app.data.ingestion import ImageMetadataEnvelope


class TaskClassifier:
    """
    Deterministic task classifier for remote-sensing queries.
    Maps (query, images) -> TaskLabel in {vqa, caption, grounding, change_vqa, opt_sar_fusion}.
    """

    TASKS = ["vqa", "caption", "grounding", "change_vqa", "opt_sar_fusion", "temporal_sequence"]

    @classmethod
    def classify(cls, query: str, images: List[ImageMetadataEnvelope]) -> Tuple[str, float]:
        q_lower = query.lower().strip()
        num_images = len(images)

        # 0. Multi-temporal sequence routing (N >= 3 images)
        if num_images >= 3:
            return "temporal_sequence", 0.98

        is_seq_query = any(k in q_lower for k in [
            "timeline", "sequence", "progression", "trend over time", "evolution across",
            "t1 to t", "epochs", "consecutive", "time series"
        ])
        if is_seq_query and num_images >= 3:
            return "temporal_sequence", 0.99

        # 1. Multi-image routing (2 images)
        if num_images == 2:
            m1 = images[0].modality
            m2 = images[1].modality

            is_opt1 = m1 in ("optical", "multispectral")
            is_sar1 = m1 == "sar"
            is_opt2 = m2 in ("optical", "multispectral")
            is_sar2 = m2 == "sar"

            is_change_query = any(k in q_lower for k in [
                "between these two", "what changed", "change between", "has construction increased",
                "has the built-up area increased", "increased, decreased, or remained", "increased between",
                "dates", "t1", "t2", "over time", "temporal", "difference between"
            ])
            is_cross_modal_pair = (is_opt1 and is_sar2) or (is_sar1 and is_opt2)
            sar_fusion_triggers = [
                "sar", "radar", "backscatter", "microwave", "cloud penetration", "through cloud",
                "penetrate", "dielectric", "polarization", "polarisation", "vv", "vh", "all-weather",
                "cross-modal", "fusion", "both optical and sar", "radar reflectance", "moisture",
                "surface roughness", "double-bounce", "correlate"
            ]
            # A temporal question that asks for SAR confirmation is a composite
            # change workflow. The planner then schedules both specialist tools.
            if is_change_query and is_cross_modal_pair and any(k in q_lower for k in sar_fusion_triggers):
                return "change_vqa", 0.97
            if is_change_query and not is_cross_modal_pair:
                return "change_vqa", 0.96

            # When an optical + SAR pair is present:
            if is_cross_modal_pair:
                # Check if the query specifically requests SAR / radar / cross-modal fusion
                if any(k in q_lower for k in sar_fusion_triggers):
                    # Query explicitly requires cross-modal radar fusion (Mode B - DOFA-ViT-B)
                    return "opt_sar_fusion", 0.98
                else:
                    # Query is about visual features, land-cover, buildings, objects, grounding, etc.
                    # The Agent autonomously routes to Single Optical Scene Mode (Mode A - EarthDial-4B)!
                    grounding_triggers = [
                        "highlight", "locate", "box", "draw", "bounding", "where is", "where are",
                        "detect the region", "segment", "pinpoint", "mark"
                    ]
                    if any(trigger in q_lower for trigger in grounding_triggers):
                        return "grounding", 0.94

                    caption_triggers = [
                        "describe the land-cover", "describe", "caption", "scene description",
                        "summarize the scene", "give an overview", "detailed summary", "what does this scene show"
                    ]
                    if any(trigger in q_lower for trigger in caption_triggers):
                        return "caption", 0.93

                    return "vqa", 0.95

            # Otherwise, 2 images of same modality are treated as bi-temporal change analysis
            return "change_vqa", 0.96

        # Check explicit change queries even if single image uploaded (will fail validation with clear error)
        if any(k in q_lower for k in ["between these two", "what changed", "change between", "has the built-up area increased", "increased, decreased, or remained"]):
            return "change_vqa", 0.92

        if any(k in q_lower for k in ["optical and sar", "use the optical and sar", "together to identify"]):
            return "opt_sar_fusion", 0.92

        # 2. Single-image routing
        # Check text-guided region grounding
        grounding_triggers = [
            "highlight", "locate", "box", "draw", "bounding", "where is", "where are",
            "detect the region", "segment", "pinpoint", "mark"
        ]
        if any(trigger in q_lower for trigger in grounding_triggers):
            return "grounding", 0.94

        # Check scene captioning / land-cover description
        caption_triggers = [
            "describe the land-cover", "describe", "caption", "scene description",
            "summarize the scene", "give an overview", "detailed summary", "what does this scene show"
        ]
        if any(trigger in q_lower for trigger in caption_triggers):
            return "caption", 0.93

        # Default single-image task: Visual Question Answering (VQA)
        return "vqa", 0.95
