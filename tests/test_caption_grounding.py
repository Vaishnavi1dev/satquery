"""Grounding tests for the single-image caption tool.

Verifies the caption answer is grounded in measured pixel composition, that the
model sentence is clearly attributed, that optical captions use the original
general-purpose EarthDial-4B RGB checkpoint while MS/SAR use the multispectral
checkpoint, and that unsupported water claims are suppressed. Also confirms no
fabricated dB or band references leak into the templated fallback text.
"""

import os

import numpy as np
from PIL import Image

from app.tools.rs_caption import (
    SingleImageCaptionTool,
    _measure_composition,
)
from app.data.ingestion import ImageMetadataEnvelope


SIZE = 128
DESCRIPTION_LABEL = "Scene description (EarthDial-4B): "


class _StubRuntime:
    """Minimal runtime that never loads a real checkpoint but records call args."""

    def __init__(self, earthdial_result=None, checkpoint_dir=None):
        self._earthdial = earthdial_result
        self._ckpt = checkpoint_dir
        self.load_errors = {}
        self.loaded_keys = []
        self.run_model_keys = []
        self.prompts = []

    @staticmethod
    def earthdial_model_key_for(modalities, default_key):
        return default_key

    def ensure_model_loaded(self, model_key, load_group=None, **kwargs):
        self.loaded_keys.append(model_key)
        return None

    def run_earthdial(self, query, image_paths, parameters, model_key=None, **kwargs):
        self.prompts.append(query)
        self.run_model_keys.append(model_key)
        return self._earthdial

    def get_checkpoint_dir(self, *args, **kwargs):
        return self._ckpt


def _caption_tool(runtime):
    return SingleImageCaptionTool(
        {
            "name": "rs-caption",
            "model_key": "earthdial-4b",
            "model_name": "EarthDial-4B (Model A)",
            "load_group": "llm_primary",
            "permitted_params": {
                "max_new_tokens": {"type": "int", "default": 128, "min": 1, "max": 512}
            },
        },
        runtime,
    )


def _solid(path, rgb, size=SIZE):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = rgb
    Image.fromarray(arr).save(path)
    return str(path)


def _envelope(image_id, modality, filepath):
    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=os.path.basename(filepath),
        filepath=filepath,
        width=SIZE,
        height=SIZE,
        bands=1 if modality == "sar" else 3,
        dtype="uint8",
        file_size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 128,
        sha256="a" * 64,
        modality=modality,
        is_georeferenced=False,
        crs=None,
        bounds=None,
    )


# --- _measure_composition ---------------------------------------------------

def test_measure_composition_missing_file_returns_none(tmp_path):
    assert _measure_composition(str(tmp_path / "nope.png")) is None
    assert _measure_composition("") is None


def test_measure_composition_partitions_sum_to_100(tmp_path):
    comp = _measure_composition(_solid(tmp_path / "v.png", (30, 180, 40)))
    assert comp is not None
    assert abs(sum(comp.values()) - 100.0) < 0.01
    assert comp["vegetation_pct"] > 99.0


def test_measure_composition_classifies_water_and_built_up(tmp_path):
    water = _measure_composition(_solid(tmp_path / "w.png", (10, 10, 40)))
    assert water is not None and water["water_pct"] > 99.0

    built = _measure_composition(_solid(tmp_path / "b.png", (180, 180, 180)))
    assert built is not None and built["built_up_or_bare_pct"] > 99.0


# --- grounded caption composition -------------------------------------------

def test_caption_optical_uses_original_checkpoint_and_suppresses_water(tmp_path):
    path = _solid(tmp_path / "opt.png", (30, 180, 40))
    env = _envelope("opt", "optical", path)
    runtime = _StubRuntime(
        earthdial_result={
            "text": "Inland waters, agriculture, Coastal wetlands, open water bodies.",
            "checkpoint_dir": "ckpt",
        }
    )
    out = _caption_tool(runtime).invoke({"envelope": env})

    # 1. Real pixel measurement leads the answer.
    assert out.text.startswith("Measured land-cover composition")
    assert "vegetation ~100%" in out.text

    # 2. Optical captions use the original general-purpose RGB checkpoint.
    assert runtime.loaded_keys == ["earthdial-original"]
    assert runtime.run_model_keys == ["earthdial-original"]
    assert out.metadata["earthdial_model_key"] == "earthdial-original"

    # 3. Object-focused prompt (greedy from clean_params).
    prompt = runtime.prompts[0]
    assert prompt.startswith("Describe this optical (RGB) scene in one or two factual sentences.")
    assert "List the main visible objects (vehicles, roads, buildings, water, vegetation)" in prompt

    # 4. Attribution label is present.
    assert DESCRIPTION_LABEL in out.text

    # 5. Unsupported water claims are removed; the rest of the sentence survives.
    assert "Inland waters" not in out.text
    assert "Coastal" not in out.text
    assert "open water bodies" not in out.text
    assert "agriculture" in out.text
    assert "pixel analysis indicates no significant water." in out.text

    assert out.confidence == 0.9
    assert out.metadata["confidence_basis"] == "measured_pixel_analysis"
    assert out.boxes is None and out.evidence == []
    assert "-22 dB" not in out.text and "(B08)" not in out.text


def test_caption_keeps_water_claim_when_measured_water_is_present(tmp_path):
    path = _solid(tmp_path / "water.png", (10, 10, 40))
    env = _envelope("water", "optical", path)
    runtime = _StubRuntime(
        earthdial_result={"text": "Inland waters and water bodies.", "checkpoint_dir": "ckpt"}
    )
    out = _caption_tool(runtime).invoke({"envelope": env})

    assert "water ~100%" in out.text
    assert "Inland waters" in out.text
    assert "pixel analysis indicates no significant water." not in out.text


def test_caption_sar_uses_multispectral_checkpoint(tmp_path):
    path = _solid(tmp_path / "sar.png", (120, 120, 120))
    env = _envelope("sar", "sar", path)
    runtime = _StubRuntime(earthdial_result=None)
    out = _caption_tool(runtime).invoke({"envelope": env})

    assert out.text.startswith("Measured land-cover composition")
    assert "dB" not in out.text
    assert out.metadata["modality"] == "sar"
    assert runtime.loaded_keys == ["earthdial-ms"]
    assert out.metadata["earthdial_model_key"] == "earthdial-ms"
    if runtime.prompts:
        assert runtime.prompts[0].startswith("Describe this SAR scene")


def test_caption_multispectral_uses_multispectral_checkpoint(tmp_path):
    path = _solid(tmp_path / "ms.png", (30, 180, 40))
    env = _envelope("ms", "multispectral", path)
    out = _caption_tool(_StubRuntime(earthdial_result=None)).invoke({"envelope": env})

    assert "B08" not in out.text
    assert out.metadata["earthdial_model_key"] == "earthdial-ms"


def test_caption_without_measurement_attributes_model_text(tmp_path):
    missing = str(tmp_path / "missing.png")
    env = _envelope("m", "optical", missing)
    runtime = _StubRuntime(
        earthdial_result={"text": "A canned fallback caption.", "checkpoint_dir": None}
    )
    out = _caption_tool(runtime).invoke({"envelope": env})

    assert out.text.startswith(DESCRIPTION_LABEL)
    assert "A canned fallback caption." in out.text
    assert out.metadata["earthdial_model_key"] == "earthdial-original"
    assert out.confidence == 0.9
    assert out.metadata["confidence_basis"] == "nominal_model_estimate"


def test_caption_without_measurement_or_model_is_not_available(tmp_path):
    missing = str(tmp_path / "missing2.png")
    env = _envelope("m2", "optical", missing)
    out = _caption_tool(_StubRuntime(earthdial_result=None)).invoke({"envelope": env})

    assert out.confidence is None
    assert out.metadata["confidence_basis"] == "not_available"
    assert "Measured land-cover composition" not in out.text
    assert "dB" not in out.text