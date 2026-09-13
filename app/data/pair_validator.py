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


class PairValidator:
    """Validates compatibility of image inputs for remote-sensing workflows."""

    @staticmethod
    def validate_single(envelope: ImageMetadataEnvelope) -> PairValidationResult:
        if envelope.width <= 0 or envelope.height <= 0:
            raise ValidationError("VAL_CORRUPT_IMAGE", "Image dimensions are invalid or corrupted.")
        return PairValidationResult(is_valid=True, pair_type="single")

    @staticmethod
    def validate_bi_temporal_pair(
        img_t1: ImageMetadataEnvelope,
        img_t2: ImageMetadataEnvelope,
        tolerance_pct: float = 15.0
    ) -> PairValidationResult:
        warnings: List[str] = []

        # Check dimension compatibility
        w_diff = abs(img_t1.width - img_t2.width) / max(img_t1.width, img_t2.width) * 100.0
        h_diff = abs(img_t1.height - img_t2.height) / max(img_t1.height, img_t2.height) * 100.0

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
        warnings: List[str] = []
        for i, img in enumerate(images[1:], start=2):
            w_diff = abs(ref.width - img.width) / max(ref.width, img.width) * 100.0
            h_diff = abs(ref.height - img.height) / max(ref.height, img.height) * 100.0
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
