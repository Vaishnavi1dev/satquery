from typing import List
from app.data.ingestion import ImageMetadataEnvelope
from app.data.pair_validator import PairValidator, ValidationError


class AgentInputValidator:
    """Validates input compatibility against classified task requirements."""

    @staticmethod
    def validate(task: str, images: List[ImageMetadataEnvelope]):
        if not images:
            raise ValidationError("VAL_NO_INPUT_IMAGES", "At least one input image is required.")

        if task in ("vqa", "caption", "grounding"):
            if len(images) > 1:
                # We analyze the primary (first) image for single-image tasks
                pass
            PairValidator.validate_single(images[0])

        elif task == "change_vqa":
            if len(images) < 2:
                raise ValidationError(
                    "VAL_INVALID_INPUT_COUNT",
                    f"Task '{task}' (bi-temporal change analysis) requires exactly 2 observations (T1 and T2). "
                    f"Provided: {len(images)} image."
                )
            PairValidator.validate_bi_temporal_pair(images[0], images[1])

        elif task == "opt_sar_fusion":
            if len(images) < 2:
                raise ValidationError(
                    "VAL_INVALID_INPUT_COUNT",
                    f"Task '{task}' (cross-modal optical-SAR joint analysis) requires 2 co-registered images "
                    f"(one Optical/Multispectral and one SAR). Provided: {len(images)} image."
                )
            PairValidator.validate_cross_modal_pair(images[0], images[1])
