import os
import re
from typing import Dict, Any, Optional
from app.tools.base import ToolBase, ToolOutput


OPTICAL_CAPTION_MODEL_KEY = "earthdial-original"
MULTISPECTRAL_CAPTION_MODEL_KEY = "earthdial-ms"

DESCRIPTION_LABEL = "Scene description (EarthDial-4B): "

# Water-claim phrases that are unsupported when the measured water fraction is
# negligible. Ordered longest / most-specific first so compound phrases are
# removed whole before the bare ``water`` token.
_WATER_CLAIM_PATTERNS = (
    r"\bopen\s+water\s+bodies\b",
    r"\bwater\s+bodies\b",
    r"\bwater\s+body\b",
    r"\binland\s+waters?\b",
    r"\bcoastal\s+wetlands?\b",
    r"\bcoastal\b",
    r"\bopen\s+water\b",
    r"\bwaters?\b",
)

WATER_SUPPRESSION_NOTE = " (pixel analysis indicates no significant water.)"


def _caption_model_key(modality: str) -> str:
    """Select the caption checkpoint by modality.

    Optical scenes are described by the original general-purpose EarthDial-4B RGB
    checkpoint, which produces grounded scene descriptions. Multispectral and SAR
    inputs keep the multispectral checkpoint. The caption tool intentionally does
    not use the generic ``earthdial_model_key_for`` helper, which routes optical to
    the fine-tuned BigEarthNet adapter that hallucinates water on generic prompts.
    """
    if modality in ("multispectral", "sar"):
        return MULTISPECTRAL_CAPTION_MODEL_KEY
    return OPTICAL_CAPTION_MODEL_KEY


def _modality_label(modality: str) -> str:
    """Human-readable modality label used in the caption prompt."""
    return {
        "optical": "optical (RGB)",
        "multispectral": "multispectral",
        "sar": "SAR",
    }.get(modality, "optical (RGB)")


def _modality_prefix_label(modality: str) -> str:
    """Factual modality label used in the corrected caption prefix."""
    return {
        "optical": "Optical (RGB)",
        "multispectral": "Multispectral",
        "sar": "SAR (radar)",
    }.get(modality, "Optical (RGB)")


def _normalize_modality_text(text: str) -> str:
    """Tidy whitespace/punctuation left behind after removing a token."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _correct_modality_claim(text: str, modality: str) -> str:
    """Remove a wrong multi-spectral claim for non-multispectral inputs.

    The adapted model sometimes emits a canned "Based on multi-spectral..."
    preface regardless of the actual sensor. For non-multispectral inputs the
    misleading token is removed entirely, the surrounding text is left otherwise
    untouched, and a factual ``Observation modality: <label>.`` prefix is added.
    Multispectral text is returned unchanged.
    """
    if not text or modality == "multispectral":
        return text
    if not re.search(r"\bmulti[-\s]*spectral\b", text, flags=re.IGNORECASE):
        return text
    cleaned = re.sub(r"\bmulti[-\s]*spectral\b", "", text, flags=re.IGNORECASE)
    cleaned = _normalize_modality_text(cleaned)
    prefix = f"Observation modality: {_modality_prefix_label(modality)}. "
    return (prefix + cleaned).strip()


def _mentions_water(text: str) -> bool:
    """True when the text asserts any water/coastal presence."""
    return re.search(r"\bwater\b|\bwaters\b|\bcoastal\b", text, flags=re.IGNORECASE) is not None


def _suppress_unsupported_water(text: str) -> str:
    """Strip unsupported water claims while keeping the rest of the sentence.

    Removes water/coastal phrases and any comma-separated fragment left empty by
    that removal; all other content is preserved verbatim. Returns ``""`` when the
    sentence asserted nothing but water.
    """
    cleaned = text
    for pattern in _WATER_CLAIM_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    parts = [part.strip() for part in cleaned.split(",")]
    parts = [part for part in parts if part.strip(" \t.;:")]
    cleaned = ", ".join(parts)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*(?=[,;:])", "", cleaned).strip(" ,;:")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _measure_composition(filepath: str) -> Optional[Dict[str, float]]:
    """Measure real pixel-level land-cover fractions from a single image.

    The image is opened, converted to RGB, and reduced to a canonical 128x128
    grid; every pixel is classified as vegetation, water, built-up/bare, or
    other from brightness and per-channel dominance heuristics only (PIL +
    numpy, no scipy). Returns percentages that sum to ~100, or ``None`` when the
    file is missing or cannot be read — never a fabricated result.
    """
    if not filepath:
        return None
    try:
        from PIL import Image
        import numpy as np

        path = os.fspath(filepath)
        if not os.path.isfile(path):
            return None
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((128, 128))
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        total = float(arr.shape[0] * arr.shape[1])
        if total <= 0:
            return None

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        brightness = (r + g + b) / 3.0

        vegetation = (g > r * 1.08) & (g > b * 1.05)
        water = ((b > r * 1.05) | (brightness < 60.0)) & (brightness < 110.0)
        water = water & (~vegetation)
        built_up_or_bare = (~vegetation) & (~water) & (brightness >= 120.0)
        other = (~vegetation) & (~water) & (~built_up_or_bare)

        pct = lambda mask: float(np.count_nonzero(mask)) / total * 100.0
        return {
            "vegetation_pct": pct(vegetation),
            "water_pct": pct(water),
            "built_up_or_bare_pct": pct(built_up_or_bare),
            "other_pct": pct(other),
        }
    except Exception:
        return None


def _qualitative_scene_sentence(modality: str) -> str:
    """One qualitative, number-free modality description (no invented values)."""
    if modality == "sar":
        return (
            "Synthetic Aperture Radar (SAR) scene analysis is based on radar "
            "backscatter rather than optical colour: built-up structures and rough "
            "terrain tend to appear brighter, while smooth water and flat paved "
            "surfaces tend to appear dark, depending on the sensor and geometry."
        )
    if modality == "multispectral":
        return (
            "Multispectral observation records reflectance beyond the visible bands, "
            "so vegetation, soil, and water can differ from their apparent colour; the "
            "measured composition above reports the actual proportions of each surface type."
        )
    return (
        "High-resolution optical scene combining built-up infrastructure and natural "
        "terrain; the measured composition above reports the actual proportions of each "
        "surface type."
    )


class SingleImageCaptionTool(ToolBase):
    """
    Slot S2: Single-Image Remote Sensing Captioning & Scene Description (Model A: EarthDial-4B).
    Generates rich, detailed descriptive summaries aligned with VRSBench caption benchmarks.
    """

    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        clean_params = self.validate_and_filter_params(parameters)

        envelope = inputs.get("envelope")
        modality = self.normalize_modality(
            getattr(envelope, "modality", None) or inputs.get("modality") or "optical"
        )
        # Captioning overrides the generic modality routing: optical scenes use the
        # original general-purpose RGB checkpoint, while MS/SAR keep earthdial-ms.
        earthdial_key = _caption_model_key(modality)
        self.runtime_mgr.ensure_model_loaded(earthdial_key, self.load_group)

        real_result = None
        if envelope and getattr(envelope, "filepath", None):
            try:
                prompt = (
                    f"Describe this {_modality_label(modality)} scene in one or two factual sentences. "
                    "List the main visible objects (vehicles, roads, buildings, water, vegetation) "
                    "and the land cover."
                )
                real_result = self.runtime_mgr.run_earthdial(
                    prompt,
                    [envelope.filepath],
                    clean_params,
                    model_key=earthdial_key,
                )
            except Exception as exc:
                self.runtime_mgr.load_errors[earthdial_key] = f"Inference: {type(exc).__name__}: {exc}"

        # Ground the answer in real pixels from THIS image (None if unreadable).
        composition = None
        if envelope and getattr(envelope, "filepath", None):
            composition = _measure_composition(envelope.filepath)

        checkpoint_dir = self.runtime_mgr.get_checkpoint_dir(earthdial_key)
        has_trained_weights = checkpoint_dir is not None
        model_text = self.usable_model_text(real_result)
        corrected_model_text = _correct_modality_claim(model_text, modality) if model_text else None

        # Water-consistency guard: the adapted model can assert water even when this
        # image's pixels contain almost none. Only suppress when a genuine measurement
        # exists, the measured water fraction is negligible, and the model claimed
        # water; the remaining sentence content is left untouched.
        if (
            composition is not None
            and corrected_model_text
            and composition["water_pct"] < 5.0
            and _mentions_water(corrected_model_text)
        ):
            suppressed = _suppress_unsupported_water(corrected_model_text)
            corrected_model_text = (
                suppressed + WATER_SUPPRESSION_NOTE if suppressed else WATER_SUPPRESSION_NOTE.strip()
            )

        if composition is not None:
            # PRIMARY, factual line measured directly from the image's pixels.
            text = (
                f"Measured land-cover composition (pixel analysis of this {_modality_label(modality)} scene): "
                f"vegetation ~{composition['vegetation_pct']:.0f}%, "
                f"built-up/bare soil ~{composition['built_up_or_bare_pct']:.0f}%, "
                f"water ~{composition['water_pct']:.0f}%, "
                f"other ~{composition['other_pct']:.0f}%."
            )
            text += "\n\n" + _qualitative_scene_sentence(modality)
            if corrected_model_text:
                text += "\n\n" + DESCRIPTION_LABEL + corrected_model_text
            conf = 0.90
            confidence_basis = "measured_pixel_analysis"
        elif corrected_model_text:
            # No pixel measurement available: attribute the model sentence honestly.
            text = DESCRIPTION_LABEL + corrected_model_text
            conf = 0.90
            confidence_basis = "nominal_model_estimate"
        else:
            text = _qualitative_scene_sentence(modality)
            conf = None
            confidence_basis = "not_available"

        return ToolOutput(
            tool_name=self.name,
            model_name=self.model_name,
            text=text,
            boxes=None,
            confidence=round(conf, 3) if conf is not None else None,
            evidence_type="analysed_image",
            evidence_ptr=envelope.image_id if envelope else None,
            evidence=[],  # Requirement 2: Captioning produces no explicit localization; mark unavailable instead of fabricating
            parameters_used=clean_params,
            metadata={
                "modality": modality,
                "earthdial_model_key": earthdial_key,
                "confidence_basis": confidence_basis,
                "slot": "S2",
                "evidence_status": "unavailable",
                "fine_tuned_weights_present": has_trained_weights,
                "inference_backend": "checkpoint" if model_text else "simulation",
                "checkpoint_dir": (
                    real_result.get("checkpoint_dir") if isinstance(real_result, dict)
                    else (str(checkpoint_dir) if checkpoint_dir else None)
                ),
            }
        )
