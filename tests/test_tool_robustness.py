"""Robustness regression tests for the specialist tools.

Covers the confirmed findings: fabricated vessel counts, null/non-envelope list
entries, over-long image lists, unsafe ``token_confidence`` coercion, degenerate
zero-area measurements, NaN/boolean parameter parsing, and model-text sentinels.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.tools.base import ToolBase, ToolParameterError
from app.tools.rs_vqa import SingleImageVQATool, _count_vessels
from app.tools.rs_caption import _correct_modality_claim
from app.tools.rs_ground import TextGuidedGroundingTool
from app.tools.change_vqa import BiTemporalChangeVQATool
from app.tools.opt_sar_fusion import OpticalSARFusionTool
from app.tools.sequence_analyzer import MultiTemporalSequenceTool, _measure_pair_change
from app.data.ingestion import ImageMetadataEnvelope
from app.runtime.manager import ModelExecutionError


SIZE = 512


class _DummyTool(ToolBase):
    def invoke(self, inputs, parameters=None):
        raise NotImplementedError


class _StubRuntime:
    """Minimal runtime that never loads a real checkpoint."""

    def __init__(self, dofa_result=None, earthdial_result=None, checkpoint_dir=None, water_gate_result=None):
        self._dofa = dofa_result
        self._earthdial = earthdial_result
        self._ckpt = checkpoint_dir
        self._water_gate = water_gate_result
        self.load_errors = {}

    @staticmethod
    def earthdial_model_key_for(modalities, default_key):
        return default_key

    def ensure_model_loaded(self, *args, **kwargs):
        return None

    def run_dofa_fusion(self, *args, **kwargs):
        return self._dofa

    def run_earthdial(self, query="", *args, **kwargs):
        # The water-presence gate uses a distinct prompt; serve it separately from
        # the tool's main-query inference so the two are independently testable.
        if isinstance(query, str) and "harbour, or port" in query:
            return self._water_gate
        return self._earthdial

    def get_checkpoint_dir(self, *args, **kwargs):
        return self._ckpt


def _write_rgb(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def _write_sar(path, size=SIZE):
    arr = np.full((size, size), 30, dtype=np.uint8)
    arr[0:64, 0:64] = 200
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


def _blob_image(blobs, extra=None, size=SIZE):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = (10, 10, 40)
    for (r0, c0) in blobs:
        arr[r0:r0 + 64, c0:c0 + 64] = (255, 255, 255)
    if extra is not None:
        arr[extra[0]:extra[1], :] = extra[2]
    return arr


def _plain_water(size=SIZE):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = (10, 10, 40)
    return arr


def _envelope(image_id, modality, filepath, width=SIZE, height=SIZE, filename=None):
    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=filename or os.path.basename(filepath),
        filepath=filepath,
        width=width,
        height=height,
        bands=1 if modality == "sar" else 3,
        dtype="uint8",
        file_size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 128,
        sha256="a" * 64,
        modality=modality,
        is_georeferenced=False,
        crs=None,
        bounds=None,
    )


def _fusion_tool(runtime):
    return OpticalSARFusionTool(
        {"name": "opt-sar-fusion", "model_key": "dofa-fusion", "model_name": "DOFA", "load_group": "encoder"},
        runtime,
    )


def _change_tool(runtime):
    return BiTemporalChangeVQATool(
        {"name": "change-vqa", "model_key": "earthdial-original", "model_name": "EarthDial", "load_group": "llm_secondary"},
        runtime,
    )


def _sequence_tool(runtime):
    return MultiTemporalSequenceTool(
        {"name": "temporal-sequence", "model_key": "earthdial-original", "model_name": "EarthDial", "load_group": "llm_secondary"},
        runtime,
    )


def _ground_tool(runtime):
    return TextGuidedGroundingTool(
        {"name": "rs-ground", "model_key": "earthdial-4b", "model_name": "EarthDial", "load_group": "llm_primary"},
        runtime,
    )


def _vqa_tool(runtime):
    return SingleImageVQATool(
        {"name": "rs-vqa", "model_key": "earthdial-4b", "model_name": "EarthDial", "load_group": "llm_primary"},
        runtime,
    )


def _param_tool():
    return _DummyTool(
        {
            "name": "dummy",
            "model_key": "k",
            "model_name": "m",
            "load_group": "g",
            "permitted_params": {
                "max_new_tokens": {"type": "int", "default": 256, "min": 1, "max": 1024},
                "temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
                "do_sample": {"type": "bool", "default": False},
            },
        },
        _StubRuntime(),
    )


# --- Vessel counting: real counts, no clamp --------------------------------

def test_vessel_count_reports_single_blob_as_one(tmp_path):
    path = _write_rgb(tmp_path / "one_blob.png", _blob_image([(0, 0)]))
    result = _count_vessels(path, 90.0)
    assert result["count"] == 1
    assert result["box"] is not None


def test_vessel_count_is_not_clamped(tmp_path):
    path = _write_rgb(tmp_path / "two_blobs.png", _blob_image([(0, 0), (384, 384)]))
    result = _count_vessels(path, 90.0)
    assert result["count"] == 2


def test_vessel_count_plain_water_is_zero(tmp_path):
    path = _write_rgb(tmp_path / "plain_water.png", _plain_water())
    result = _count_vessels(path, 90.0)
    assert result["count"] == 0
    assert result["box"] is None


def test_vessel_count_unreadable_is_none(tmp_path):
    result = _count_vessels(str(tmp_path / "missing.png"), 0.0)
    assert result["count"] is None
    assert result["box"] is None


# --- Vessel counting: real water-presence gate -----------------------------

VESSEL_QUERY = "Identify and count the cargo vessels in this image."
WATER_YES = "Yes, a harbour water body is clearly visible in this image."
WATER_NO = "No, there is no water body, harbour, or port visible in this scene."


def _vessel_env(tmp_path, name, arr):
    path = _write_rgb(tmp_path / name, arr)
    return _envelope(name, "optical", path)


def test_vessel_gate_non_water_no_count_no_box_no_evidence(tmp_path):
    """A non-water scene must never be counted or boxed (the reported bug)."""
    env = _vessel_env(tmp_path, "land.png", _blob_image([(0, 0), (384, 384)]))
    runtime = _StubRuntime(earthdial_result=None, water_gate_result={"text": WATER_NO})
    out = _vqa_tool(runtime).invoke({"query": VESSEL_QUERY, "envelope": env})
    assert out.metadata["water_detected"] is False
    assert out.metadata["count_method"] == "not_applicable"
    assert out.metadata["confidence_basis"] == "nominal_model_estimate"
    assert out.boxes is None
    assert not any(ev.get("type") == "vessel_count" for ev in out.evidence)
    assert "No significant water body or port is visible" in out.text


def test_vessel_gate_water_with_clusters_reports_real_count(tmp_path):
    env = _vessel_env(tmp_path, "harbour.png", _blob_image([(0, 0), (384, 384)]))
    runtime = _StubRuntime(earthdial_result=None, water_gate_result={"text": WATER_YES})
    out = _vqa_tool(runtime).invoke({"query": VESSEL_QUERY, "envelope": env})
    assert out.metadata["water_detected"] is True
    assert out.metadata["count_method"] == "bright-target clustering heuristic"
    vessel_ev = [ev for ev in out.evidence if ev.get("type") == "vessel_count"]
    assert len(vessel_ev) == 1
    assert vessel_ev[0]["description"].startswith("Counted 2 cargo vessels")
    assert out.boxes is not None and len(out.boxes) == 1
    assert "2 cargo vessels" in out.text
    x1, y1, x2, y2 = out.boxes[0]
    assert 0 <= x1 <= x2 <= 512 and 0 <= y1 <= y2 <= 512


def test_vessel_gate_water_without_clusters_is_zero_no_box(tmp_path):
    env = _vessel_env(tmp_path, "open_water.png", _plain_water())
    runtime = _StubRuntime(earthdial_result=None, water_gate_result={"text": WATER_YES})
    out = _vqa_tool(runtime).invoke({"query": VESSEL_QUERY, "envelope": env})
    assert out.metadata["water_detected"] is True
    assert out.metadata["count_method"] == "bright-target clustering heuristic"
    assert out.boxes is None
    assert not any(ev.get("type") == "vessel_count" for ev in out.evidence)
    assert "No cargo vessels detected on the water" in out.text


def test_vessel_gate_missing_model_is_not_applicable(tmp_path):
    """No gate model output -> refuse to count instead of guessing."""
    env = _vessel_env(tmp_path, "no_model.png", _blob_image([(0, 0)]))
    runtime = _StubRuntime(earthdial_result=None, water_gate_result=None)
    out = _vqa_tool(runtime).invoke({"query": VESSEL_QUERY, "envelope": env})
    assert out.metadata["water_detected"] is False
    assert out.metadata["count_method"] == "not_applicable"
    assert out.metadata["confidence_basis"] == "not_available"
    assert out.boxes is None
    assert not any(ev.get("type") == "vessel_count" for ev in out.evidence)


def test_water_present_from_gate_rejects_negations_and_ambiguity():
    from app.tools.rs_vqa import _water_present_from_gate
    assert _water_present_from_gate("Yes, a river is visible.") is True
    assert _water_present_from_gate("No, there is no water body.") is False
    assert _water_present_from_gate("No water or harbour is visible.") is False
    assert _water_present_from_gate("Maybe a small pond.") is False
    assert _water_present_from_gate(None) is False


# --- Null / extra image envelopes ------------------------------------------

def test_fusion_rejects_null_envelope(tmp_path):
    sar = _envelope("sar", "sar", _write_sar(tmp_path / "sar.png"))
    with pytest.raises(ModelExecutionError) as exc:
        _fusion_tool(_StubRuntime()).invoke({"images": [None, sar], "query": "q"})
    assert exc.value.code == "MDL_INVALID_IMAGE_ENVELOPE"


def test_fusion_rejects_more_than_two_images(tmp_path):
    opt = _envelope("opt", "optical", _write_rgb(tmp_path / "opt.png", _blob_image([(0, 0)])))
    sar = _envelope("sar", "sar", _write_sar(tmp_path / "sar.png"))
    with pytest.raises(ModelExecutionError) as exc:
        _fusion_tool(_StubRuntime()).invoke({"images": [opt, sar, sar], "query": "q"})
    assert exc.value.code == "MDL_JOINT_USE_VIOLATION"


def test_change_rejects_null_envelope(tmp_path):
    opt = _envelope("opt", "optical", _write_rgb(tmp_path / "opt.png", _blob_image([(0, 0)])))
    with pytest.raises(ModelExecutionError) as exc:
        _change_tool(_StubRuntime()).invoke({"images": [None, opt], "query": "q"})
    assert exc.value.code == "MDL_INVALID_IMAGE_ENVELOPE"


def test_sequence_rejects_null_envelope(tmp_path):
    opt = _envelope("opt", "optical", _write_rgb(tmp_path / "opt.png", _blob_image([(0, 0)])))
    with pytest.raises(ModelExecutionError) as exc:
        _sequence_tool(_StubRuntime()).invoke({"images": [opt, None, opt], "query": "q"})
    assert exc.value.code == "MDL_INVALID_IMAGE_ENVELOPE"


# --- token_confidence coercion ---------------------------------------------

def _fusion_inputs(tmp_path):
    opt = _envelope("opt", "optical", _write_rgb(tmp_path / "opt.png", _blob_image([(0, 0)])))
    sar = _envelope("sar", "sar", _write_sar(tmp_path / "sar.png"))
    return opt, sar


def test_fusion_none_token_confidence_falls_back_to_measured(tmp_path):
    opt, sar = _fusion_inputs(tmp_path)
    runtime = _StubRuntime(dofa_result={"token_confidence": None, "checkpoint_dir": "ckpt", "logits_shape": [1, 196, 1000]})
    out = _fusion_tool(runtime).invoke({"images": [opt, sar], "query": "identify built-up and water regions"})
    assert out.confidence == 0.9
    assert out.metadata["confidence_basis"] == "measured_pixel_analysis"
    assert out.metadata["inference_backend"] == "simulation"


def test_fusion_string_token_confidence_falls_back_to_measured(tmp_path):
    opt, sar = _fusion_inputs(tmp_path)
    runtime = _StubRuntime(dofa_result={"token_confidence": "0.9", "checkpoint_dir": "ckpt"})
    out = _fusion_tool(runtime).invoke({"images": [opt, sar], "query": "identify built-up and water regions"})
    assert out.metadata["confidence_basis"] == "measured_pixel_analysis"


def test_fusion_valid_token_confidence_uses_model_estimate(tmp_path):
    opt, sar = _fusion_inputs(tmp_path)
    runtime = _StubRuntime(dofa_result={"token_confidence": 0.10, "checkpoint_dir": "ckpt"})
    out = _fusion_tool(runtime).invoke({"images": [opt, sar], "query": "identify built-up and water regions"})
    assert out.confidence == 0.85
    assert out.metadata["confidence_basis"] == "nominal_model_estimate"
    assert out.metadata["inference_backend"] == "checkpoint"


# --- Zero-area measurements -------------------------------------------------

def test_change_zero_area_measurement_returns_none(tmp_path):
    first = _write_rgb(tmp_path / "a.png", _blob_image([(0, 0)]))
    second = _write_rgb(tmp_path / "b.png", _blob_image([(0, 0)], extra=(300, 400, (255, 0, 0))))
    e1 = SimpleNamespace(filepath=first)
    e2 = SimpleNamespace(filepath=second)
    assert BiTemporalChangeVQATool._measure_change(e1, e2, 0, 0) is None
    measured = BiTemporalChangeVQATool._measure_change(e1, e2, 512, 512)
    assert measured is not None
    x1, y1, x2, y2 = measured["box"]
    assert x2 - x1 >= 1 and y2 - y1 >= 1


def test_sequence_zero_area_measurement_returns_none(tmp_path):
    first = _write_rgb(tmp_path / "a.png", _blob_image([(0, 0)]))
    second = _write_rgb(tmp_path / "b.png", _blob_image([(0, 0)], extra=(300, 400, (255, 0, 0))))
    assert _measure_pair_change(first, second, 0, 0) is None


# --- Modality / parameter / model-text hardening ---------------------------

def test_modality_normalization_and_validation():
    tool = _param_tool()
    assert tool.normalize_modality("OPTICAL") == "optical"
    assert tool.normalize_modality(" Sar ") == "sar"
    with pytest.raises(ModelExecutionError):
        tool.normalize_modality("lidar")
    with pytest.raises(ModelExecutionError):
        tool.normalize_modality(None)


def test_nan_parameter_is_rejected():
    with pytest.raises(ToolParameterError):
        _param_tool().validate_and_filter_params({"temperature": float("nan")})


def test_boolean_parameter_parsing_is_explicit():
    assert _param_tool().validate_and_filter_params({"do_sample": "false"})["do_sample"] is False
    assert _param_tool().validate_and_filter_params({"do_sample": "true"})["do_sample"] is True
    with pytest.raises(ToolParameterError):
        _param_tool().validate_and_filter_params({"do_sample": "yes"})


def test_int_parameter_rejects_float_and_bool():
    with pytest.raises(ToolParameterError):
        _param_tool().validate_and_filter_params({"max_new_tokens": 32.5})
    with pytest.raises(ToolParameterError):
        _param_tool().validate_and_filter_params({"max_new_tokens": True})


def test_usable_model_text_treats_sentinels_as_no_output():
    tool = _param_tool()
    for sentinel in (None, "", "None", "none", " null "):
        assert tool.usable_model_text({"text": sentinel}) is None
    assert tool.usable_model_text({"text": "A real answer"}) == "A real answer"
    assert tool.usable_model_text(None) is None


def test_vqa_model_text_sentinel_is_not_surfaced(tmp_path):
    path = _write_rgb(tmp_path / "v.png", _blob_image([(0, 0)]))
    env = _envelope("v", "optical", path)
    runtime = _StubRuntime(earthdial_result={"text": "None", "checkpoint_dir": "ckpt"})
    out = _vqa_tool(runtime).invoke({"query": "Describe the scene", "envelope": env})
    assert out.text.strip().lower() != "none"
    assert isinstance(out.text, str) and len(out.text) > 0
    assert out.metadata["inference_backend"] == "simulation"


# --- Grounding hardening ----------------------------------------------------

def test_ground_parse_model_boxes_non_string_returns_empty():
    assert TextGuidedGroundingTool._parse_model_boxes(None) == []
    assert TextGuidedGroundingTool._parse_model_boxes(123) == []


def test_ground_empty_query_is_rejected():
    with pytest.raises(ModelExecutionError) as exc:
        _ground_tool(_StubRuntime()).invoke({"query": "   ", "envelope": None})
    assert exc.value.code == "MDL_EMPTY_QUERY"


def test_ground_incomplete_transform_meta_falls_back(tmp_path):
    path = _write_rgb(tmp_path / "g.png", _blob_image([(0, 0)]))
    env = _envelope("g", "optical", path, width=96, height=96)
    runtime = _StubRuntime(earthdial_result={"text": "region [100, 200, 300, 400]", "checkpoint_dir": "ckpt"})
    out = _ground_tool(runtime).invoke(
        {"query": "locate the water", "envelope": env, "transform_meta": {"target_size": 512}}
    )
    assert out.boxes is not None and len(out.boxes) == 1
    x1, y1, x2, y2 = out.boxes[0]
    assert x2 > x1 and y2 > y1


# --- Caption modality claim -------------------------------------------------

def test_caption_modality_claim_removes_multi_whitespace_token():
    cleaned = _correct_modality_claim("Based on multi  spectral analysis of the scene.", "optical")
    assert "spectral" not in cleaned.lower()
    assert cleaned.startswith("Observation modality: Optical (RGB).")


def test_caption_modality_claim_preserves_multispectrality_word():
    text = "The multispectrality of the sensor is notable."
    assert _correct_modality_claim(text, "optical") == text


# --- Sequence cumulative presentation ---------------------------------------

def test_sequence_labels_cumulative_sum_and_reports_mean(tmp_path):
    p1 = _write_rgb(tmp_path / "s1.png", _blob_image([(0, 0)]))
    p2 = _write_rgb(tmp_path / "s2.png", _blob_image([(0, 0)], extra=(300, 400, (255, 0, 0))))
    p3 = _write_rgb(tmp_path / "s3.png", _blob_image([(0, 0)], extra=(200, 320, (0, 255, 0))))
    envs = [
        _envelope("s1", "optical", p1),
        _envelope("s2", "optical", p2),
        _envelope("s3", "optical", p3),
    ]
    out = _sequence_tool(_StubRuntime(earthdial_result=None)).invoke(
        {"images": envs, "query": "analyze the cumulative urban change across the timeline"}
    )
    assert "cumulative" in out.text.lower()
    assert "can exceed 100%" in out.text.lower()
    assert out.metadata["cumulative_delta_label"]
    assert "mean_delta_pct" in out.metadata
