from typing import List, Tuple, Optional
from pydantic import BaseModel
from app.data.ingestion import ImageMetadataEnvelope


class ValidationError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class PairValidationResult(BaseModel):
    is_valid: bool
    pair_type: str  # "single", "bi_temporal", "cross_modal_opt_sar"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = []


def _require_valid_dimensions(width: int, height: int) -> None:
    """Reject zero/negative image dimensions before any ratio arithmetic.

    ``abs(a - b) / max(a, b)`` raises ``ZeroDivisionError`` when either image reports
    a zero dimension, so every pairing path funnels through this shared guard first.
    """
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        raise ValidationError("VAL_CORRUPT_IMAGE", "Image dimensions are invalid or corrupted.")
    if width <= 0 or height <= 0:
        raise ValidationError("VAL_CORRUPT_IMAGE", "Image dimensions are invalid or corrupted.")


def _dimension_diff_pct(a: int, b: int) -> float:
    """Percent dimension difference, safe when a dimension is zero."""
    denom = max(int(a), int(b))
    if denom <= 0:
        return 0.0
    return abs(int(a) - int(b)) / float(denom) * 100.0


class PairValidator:
    """Validates compatibility of image inputs for remote-sensing workflows."""

    @staticmethod
    def validate_single(envelope: ImageMetadataEnvelope) -> PairValidationResult:
        _require_valid_dimensions(envelope.width, envelope.height)
        return PairValidationResult(is_valid=True, pair_type="single")

    @staticmethod
    def validate_bi_temporal_pair(
        img_t1: ImageMetadataEnvelope,
        img_t2: ImageMetadataEnvelope,
        tolerance_pct: float = 15.0
    ) -> PairValidationResult:
        warnings: List[str] = []

        _require_valid_dimensions(img_t1.width, img_t1.height)
        _require_valid_dimensions(img_t2.width, img_t2.height)

        # Check dimension compatibility
        w_diff = _dimension_diff_pct(img_t1.width, img_t2.width)
        h_diff = _dimension_diff_pct(img_t1.height, img_t2.height)

        # Dimension/GSD mismatch is non-fatal: the rendering and preprocessing stages
        # harmonize image sizes (resize/resample), so surface it as a warning instead of failing.
        if w_diff > tolerance_pct or h_diff > tolerance_pct:
            warnings.append(
                f"Bi-temporal images have mismatched dimensions: T1 is {img_t1.width}x{img_t1.height}, "
                f"T2 is {img_t2.width}x{img_t2.height} (difference exceeds {tolerance_pct}% tolerance). "
                "Rendering/preprocessing will harmonize image sizes."
            )

        # Check CRS compatibility if both have CRS
        if img_t1.crs and img_t2.crs and img_t1.crs != img_t2.crs:
            warnings.append(f"CRS mismatch: T1 has {img_t1.crs}, T2 has {img_t2.crs}.")

        return PairValidationResult(is_valid=True, pair_type="bi_temporal", warnings=warnings)

    @staticmethod
    def validate_cross_modal_pair(
        img1: ImageMetadataEnvelope,
        img2: ImageMetadataEnvelope,
        tolerance_pct: float = 20.0
    ) -> Tuple[ImageMetadataEnvelope, ImageMetadataEnvelope]:
        """
        Validates and orders (optical_envelope, sar_envelope).
        Raises ValidationError if pair does not satisfy Optical + SAR pairing.
        """
        is_opt1 = img1.modality in ("optical", "multispectral")
        is_sar1 = img1.modality == "sar"

        is_opt2 = img2.modality in ("optical", "multispectral")
        is_sar2 = img2.modality == "sar"

        if is_opt1 and is_sar2:
            opt_img, sar_img = img1, img2
        elif is_sar1 and is_opt2:
            opt_img, sar_img = img2, img1
        else:
            raise ValidationError(
                "VAL_MODALITY_MISMATCH",
                f"Cross-modal analysis requires one Optical/Multispectral image and one SAR image. "
                f"Received modalities: '{img1.modality}' and '{img2.modality}'."
            )

        # Dimension/GSD mismatch is non-fatal: rendering/preprocessing harmonizes sizes.
        # This tuple-returning validator has no warnings channel; the modality (optical+SAR)
        # pairing check above remains the hard requirement.

        return opt_img, sar_img

    @staticmethod
    def validate_temporal_sequence(
        images: List[ImageMetadataEnvelope],
        tolerance_pct: float = 25.0
    ) -> PairValidationResult:
        """Validates a multi-temporal sequence of N >= 3 satellite observations."""
        if len(images) < 3:
            raise ValidationError(
                "VAL_INSUFFICIENT_SEQUENCE",
                f"Multi-temporal sequence analysis requires at least 3 co-registered scenes, received {len(images)}."
            )

        ref = images[0]
        _require_valid_dimensions(ref.width, ref.height)
        warnings: List[str] = []
        for i, img in enumerate(images[1:], start=2):
            _require_valid_dimensions(img.width, img.height)
            w_diff = _dimension_diff_pct(ref.width, img.width)
            h_diff = _dimension_diff_pct(ref.height, img.height)
            # Mismatched dimensions/GSD are surfaced as warnings because downstream
            # rendering and preprocessing harmonize sizes rather than resampling inputs here.
            if w_diff > tolerance_pct or h_diff > tolerance_pct:
                warnings.append(
                    f"Observation T{i} ({img.width}x{img.height}) dimensions mismatch baseline T1 "
                    f"({ref.width}x{ref.height}) by >{tolerance_pct}%. Rendering/preprocessing will harmonize image sizes."
                )
            if ref.crs and img.crs and ref.crs != img.crs:
                warnings.append(f"CRS difference at epoch T{i}: {img.crs} vs baseline {ref.crs}.")

        return PairValidationResult(is_valid=True, pair_type="temporal_sequence", warnings=warnings)
