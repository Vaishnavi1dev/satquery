from typing import List, Tuple
from app.data.ingestion import ImageMetadataEnvelope


class TaskClassifier:
    """
    Deterministic task classifier for remote-sensing queries.
    Maps (query, images) -> TaskLabel in {vqa, caption, grounding, change_vqa, opt_sar_fusion}.
    """

    TASKS = ["vqa", "caption", "grounding", "change_vqa", "opt_sar_fusion"]

    @classmethod
    def classify(cls, query: str, images: List[ImageMetadataEnvelope]) -> Tuple[str, float]:
        q_lower = query.lower().strip()
        num_images = len(images)

        # 1. Multi-image routing
        if num_images == 2:
            m1 = images[0].modality
            m2 = images[1].modality

            is_opt1 = m1 in ("optical", "multispectral")
            is_sar1 = m1 == "sar"
            is_opt2 = m2 in ("optical", "multispectral")
            is_sar2 = m2 == "sar"

            # Check for cross-modal optical-SAR pair
            if (is_opt1 and is_sar2) or (is_sar1 and is_opt2):
                return "opt_sar_fusion", 0.98

            # Check explicit optical-SAR keywords in query
            if any(k in q_lower for k in ["optical and sar", "sar and optical", "radar and optical", "together"]):
                return "opt_sar_fusion", 0.95

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
