import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from app.data.ingestion import ImageMetadataEnvelope
from app.registry.store import ToolRegistryStore
from app.agent.controller import AgentController, QueryExecutionResult
from app.agent.aggregator import OutputAggregator
from app.tools.base import ToolOutput
from app.storage.sandbox import StorageSandbox


# The only honest bases a confidence estimate may carry. Anything else would
# reintroduce a fabricated / unsourced number.
ALLOWED_CONFIDENCE_BASIS = {
    "nominal_model_estimate",
    "measured_pixel_analysis",
    "not_available",
}

IMAGE_SIZE = 96


def assert_confidence_contract(value):
    """Confidence is either None (nothing genuine ran) or a probability in [0, 1]."""
    assert value is None or (isinstance(value, float) and 0.0 <= value <= 1.0), value


def assert_confidence_basis(metadata):
    """When an estimate exists its basis must be one of the honest labels."""
    assert "confidence_basis" in metadata
    assert metadata["confidence_basis"] in ALLOWED_CONFIDENCE_BASIS


def assert_boxes_contract(boxes, width=IMAGE_SIZE, height=IMAGE_SIZE):
    """Boxes are only ever None or a non-empty list of in-bounds 4-int pixel boxes."""
    if boxes is None:
        return
    assert isinstance(boxes, list) and len(boxes) > 0
    for box in boxes:
        assert isinstance(box, list) and len(box) == 4, box
        assert all(isinstance(c, int) for c in box), box
        x1, y1, x2, y2 = box
        assert 0 <= x1 <= x2 <= width, box
        assert 0 <= y1 <= y2 <= height, box


def assert_no_fabricated_db(text):
    lowered = text.lower()
    assert "db" not in lowered and "decibel" not in lowered


def _write_real_image(path: Path, modality: str, variant: int = 0) -> None:
    """Write a small deterministic land-cover image so real pixel math can run."""
    size = IMAGE_SIZE
    if modality == "sar":
        arr = np.full((size, size), 20, dtype=np.uint8)
        band = 200
        if variant % 2 == 0:
            arr[0:size // 5, :] = band
        else:
            arr[size // 5:size // 2, :] = band
        Image.fromarray(arr, mode="L").save(path)
        return
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = (40, 150, 40)  # vegetation
    if variant % 2 == 0:
        arr[0:size // 5, :] = (225, 225, 225)  # built-up
    else:
        arr[size // 5:size // 2, :] = (225, 225, 225)  # built-up (shifted)
        arr[(size * 55) // 96:(size * 65) // 96, :] = (200, 180, 60)  # bare soil
    arr[(size * 70) // 96:, :] = (25, 50, 170)  # water
    Image.fromarray(arr).save(path)


def make_real_envelope(tmp_path, image_id, modality="optical", variant=0):
    filepath = Path(tmp_path) / f"{image_id}.png"
    _write_real_image(filepath, modality, variant)
    return ImageMetadataEnvelope(
        image_id=image_id,
        session_id="test_sess",
        filepath=str(filepath),
        filename=f"{image_id}.png",
        sha256="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        file_size_bytes=filepath.stat().st_size,
        modality=modality,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        bands=3 if modality in ("optical", "multispectral") else 1,
        dtype="uint8",
        is_georeferenced=False,
        crs=None,
        bounds=None,
    )


@pytest.fixture
def agent():
    registry = ToolRegistryStore()
    sandbox = StorageSandbox()
    return AgentController(registry=registry, sandbox=sandbox)


def test_single_image_grounding_evidence(tmp_path, agent):
    """Grounding reports a region only when the model genuinely produced one."""
    img = make_real_envelope(tmp_path, "img_opt_01", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_01",
        query="Highlight and locate the water body",
        images=[img]
    )

    assert isinstance(res, QueryExecutionResult)
    assert res.task == "grounding"
    assert "EarthDial-4B" in res.selected_model
    assert_confidence_contract(res.confidence)
    assert_boxes_contract(res.boxes, img.width, img.height)

    if res.boxes is not None:
        # A genuine model localization was parsed: boxes must be real, in-bounds ints.
        assert len(res.evidence) > 0
        box_ev = res.evidence[0]
        assert box_ev["type"] == "bounding_box"
        assert box_ev["source_model"] == "earthdial"
        assert len(box_ev["region"]) == 4
        assert all(isinstance(c, int) for c in box_ev["region"])
    else:
        # Honest unlocalized behaviour: nothing is invented and nothing is claimed.
        assert res.evidence == [] or all(ev.get("region") is None for ev in res.evidence)
        assert "high localization confidence" not in res.answer.lower()

    # Trace verification
    assert isinstance(res.trace, list)
    assert any("grounding" in t.lower() for t in res.trace)


def test_single_image_captioning_no_hallucinated_evidence(tmp_path, agent):
    """Requirement 2: Captioning must not invent fake bounding boxes or regions."""
    img = make_real_envelope(tmp_path, "img_opt_02", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_02",
        query="Describe the land-cover distribution in this scene",
        images=[img]
    )

    assert res.task == "caption"
    assert isinstance(res.answer, str) and len(res.answer) > 20
    assert_confidence_contract(res.confidence)
    # Must NOT invent visual regions/bounding boxes
    assert res.boxes is None
    assert res.evidence == []


def test_single_image_vqa_evidence(tmp_path, agent):
    """Requirement 2: VQA preserves provenance without fabricating regions."""
    img = make_real_envelope(tmp_path, "img_sar_01", modality="sar")
    res = agent.execute_query(
        session_id="sess_test_03",
        query="What is the backscatter characteristic of the urban area?",
        images=[img]
    )

    assert res.task == "vqa"
    assert isinstance(res.answer, str) and len(res.answer) > 0
    assert_confidence_contract(res.confidence)
    assert_boxes_contract(res.boxes, img.width, img.height)
    assert len(res.evidence) >= 1
    assert res.evidence[0]["source_model"] == "earthdial"
    assert res.evidence[0]["modality"] == "sar"
    assert res.evidence[0].get("region") is None

    # The tool metadata still labels the basis of whatever confidence was produced.
    tool = agent.registry.get_tool("rs-vqa")
    out = tool.invoke(
        inputs={
            "query": "What is the backscatter characteristic of the urban area?",
            "images": [img],
            "modality": "sar",
            "envelope": img,
        },
        parameters={"max_new_tokens": 32},
    )
    assert_confidence_contract(out.confidence)
    assert_confidence_basis(out.metadata)


def test_cross_modal_dofa_evidence_provenance(tmp_path, agent):
    """Requirement 3: DOFA distinguishes optical, sar, and joint contributions."""
    opt_img = make_real_envelope(tmp_path, "img_opt_pair", modality="optical")
    sar_img = make_real_envelope(tmp_path, "img_sar_pair", modality="sar")

    res = agent.execute_query(
        session_id="sess_test_04",
        query="Identify built-up areas and water bodies using optical and SAR together",
        images=[opt_img, sar_img]
    )

    assert res.task == "opt_sar_fusion"
    assert "DOFA" in res.selected_model
    assert_confidence_contract(res.confidence)
    assert_boxes_contract(res.boxes, opt_img.width, opt_img.height)

    types = [e["type"] for e in res.evidence]
    assert "optical" in types
    assert "sar" in types
    assert "joint" in types

    # Real pixel-derived land-cover classes must be present and quantified.
    measured = [e for e in res.evidence if e.get("coverage_pct") is not None]
    assert measured, "expected genuine pixel-derived class coverage"
    for ev in measured:
        assert isinstance(ev["coverage_pct"], float)
        assert 0.0 <= ev["coverage_pct"] <= 100.0

    for ev in res.evidence:
        assert ev["source_model"] == "dofa"

    assert_no_fabricated_db(res.answer)

    # Verify execution trace contains cross-modal steps
    trace_text = " ".join(res.trace).lower()
    assert "cross-modal" in trace_text or "optical" in trace_text

    # Direct tool metadata carries the honest confidence basis and real fractions.
    tool = agent.registry.get_tool("opt-sar-fusion")
    out = tool.invoke(
        inputs={"query": "identify built-up and water regions", "images": [opt_img, sar_img]},
        parameters={"max_new_tokens": 32},
    )
    assert_confidence_contract(out.confidence)
    assert_confidence_basis(out.metadata)
    assert out.boxes is not None and len(out.boxes) > 0
    for key in ("water_coverage_pct", "builtup_coverage_pct", "vegetation_coverage_pct"):
        assert isinstance(out.metadata[key], float)
    assert_no_fabricated_db(out.text)


def test_bitemporal_earthdial_evidence(tmp_path, agent):
    """Requirement 4: bi-temporal change reports only genuine pixel measurements."""
    # Two genuinely different observations so a real pixel difference exists.
    t1_img = make_real_envelope(tmp_path, "img_t1", modality="optical", variant=0)
    t2_img = make_real_envelope(tmp_path, "img_t2", modality="optical", variant=1)

    res = agent.execute_query(
        session_id="sess_test_05",
        query="What changed between Observation T1 and Observation T2? Has the built-up area increased?",
        images=[t1_img, t2_img]
    )

    assert res.task == "change_vqa"
    assert "EarthDial-4B" in res.selected_model
    assert_confidence_contract(res.confidence)
    assert res.boxes is not None and len(res.boxes) > 0
    assert_boxes_contract(res.boxes, t1_img.width, t1_img.height)

    # Check evidence structure
    assert len(res.evidence) > 0
    change_ev = res.evidence[0]
    assert change_ev["type"] == "change_region"
    assert change_ev["source_model"] == "earthdial-4b"
    assert change_ev["time_from"] == "T1"
    assert change_ev["time_to"] == "T2"
    assert change_ev["change_detected"] is True
    assert change_ev["change_type"] == "measured_surface_difference"
    assert isinstance(change_ev["change_ratio"], float)
    assert 0.0 < change_ev["change_ratio"] <= 1.0
    # The removed metric must no longer be asserted or present.
    assert "cycle_consistency" not in change_ev
    # Non-georeferenced inputs must never yield fabricated WGS84 coordinates.
    assert "\u00b0N" not in res.answer
    assert "\u00b0E" not in res.answer
    for ev in res.evidence:
        assert ev.get("geo_coordinates") is None

    # Check execution trace
    trace_text = " ".join(res.trace).lower()
    assert "bi-temporal" in trace_text or "compared t1 and t2" in trace_text

    tool = agent.registry.get_tool("change-vqa")
    out = tool.invoke(
        inputs={
            "query": "What changed between the two observations?",
            "images": [t1_img, t2_img],
            "modality": "optical",
            "envelope": t1_img,
        },
        parameters={"max_new_tokens": 32},
    )
    assert_confidence_contract(out.confidence)
    assert_confidence_basis(out.metadata)
    assert out.metadata.get("change_type") == "measured_surface_difference"
    assert isinstance(out.metadata.get("change_ratio"), float)
    assert "cycle_consistency" not in out.metadata


def test_common_result_json_serializable(tmp_path, agent):
    """Requirement 1: QueryExecutionResult is strictly JSON-serializable."""
    img = make_real_envelope(tmp_path, "img_serial", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_serial",
        query="Highlight and locate roads",
        images=[img]
    )

    # Must serialize to clean JSON string without type error
    json_str = res.model_dump_json()
    parsed = json.loads(json_str)

    assert "answer" in parsed
    assert "confidence" in parsed
    assert parsed["confidence"] is None or 0.0 <= parsed["confidence"] <= 1.0
    assert parsed["calibration_method"] == "none"
    assert parsed["calibration_temperature"] is None
    assert parsed["confidence"] == parsed["raw_confidence"]
    assert "evidence" in parsed
    assert isinstance(parsed["evidence"], list)
    assert "trace" in parsed
    assert isinstance(parsed["trace"], list)


def test_multi_specialist_aggregation_conflict_detection():
    """Requirement 9: Aggregator identifies supporting vs conflicting evidence."""
    # Specialist 1: EarthDial claims built-up expansion with high confidence
    out1 = ToolOutput(
        tool_name="change-vqa",
        model_name="EarthDial-4B (Multi-Image Engine)",
        text="Significant increase in built-up infrastructure (+18.4%).",
        boxes=[[10, 10, 100, 100]],
        confidence=0.92,
        evidence_type="bi_temporal_pair",
        evidence=[{
            "type": "change_region",
            "source_model": "earthdial-4b",
            "description": "Built-up expansion detected",
            "region": [10, 10, 100, 100],
            "score": 0.92
        }]
    )

    # Specialist 2 (Supporting): DOFA corroborates with radar backscatter
    out2_supporting = ToolOutput(
        tool_name="opt-sar-fusion",
        model_name="DOFA ViT-B",
        text="High radar backscatter confirms vertical metallic/concrete structures.",
        confidence=0.90,
        evidence_type="opt_sar_pair",
        evidence=[{
            "type": "sar",
            "source_model": "dofa",
            "description": "High double-bounce backscatter confirms urban structures",
            "score": 0.90
        }]
    )

    agg_supp = OutputAggregator.aggregate([out1, out2_supporting])
    assert len(agg_supp.supporting_evidence) == 1
    assert len(agg_supp.conflicting_evidence) == 0
    assert "Supporting Evidence" in agg_supp.answer

    # Specialist 3 (Conflicting): Model triggered honesty gate or low confidence divergence
    out3_conflicting = ToolOutput(
        tool_name="rs-ground",
        model_name="EarthDial-4B",
        text="Target structures not located in scene.",
        confidence=0.10,
        evidence_type="analysed_image",
        evidence=[{
            "type": "grounding_fallback",
            "source_model": "earthdial",
            "description": "Honesty gate triggered: structures not verified",
            "score": 0.10
        }],
        metadata={"honesty_gate_triggered": True}
    )

    agg_conf = OutputAggregator.aggregate([out1, out3_conflicting])
    assert len(agg_conf.conflicting_evidence) == 1
    assert "Conflicting Evidence / Discrepancies" in agg_conf.answer


def test_agent_multi_specialist_composition_execution(tmp_path, agent):
    """Requirement 9: End-to-end query scheduling both EarthDial-4B and DOFA when asking for change and SAR confirmation."""
    opt_img = make_real_envelope(tmp_path, "img_multi_opt", modality="optical", variant=0)
    sar_img = make_real_envelope(tmp_path, "img_multi_sar", modality="sar", variant=0)

    res = agent.execute_query(
        session_id="sess_test_multi",
        query="Has construction increased between these two dates, and does the SAR evidence support it?",
        images=[opt_img, sar_img]
    )

    assert res.task == "change_vqa"
    assert_confidence_contract(res.confidence)
    assert "Specialist 1" in res.answer or "Primary Result" in res.answer or "Observation" in res.answer
    # Should contain evidence from both change model (earthdial-4b) and dofa
    models_in_evidence = {e["source_model"] for e in res.evidence}
    assert "earthdial-4b" in models_in_evidence
    assert "dofa" in models_in_evidence
    assert any("SAR" in t or "cross-modal" in t.lower() or "dofa" in t.lower() for t in res.trace)


def test_multi_temporal_sequence_analysis(tmp_path, agent):
    """Test multi-temporal sequence analysis across T1 -> T2 -> T3 epochs."""
    t1 = make_real_envelope(tmp_path, "img_epoch_t1", modality="optical", variant=0)
    t2 = make_real_envelope(tmp_path, "img_epoch_t2", modality="optical", variant=1)
    t3 = make_real_envelope(tmp_path, "img_epoch_t3", modality="optical", variant=0)

    res = agent.execute_query(
        session_id="sess_test_seq",
        query="Analyze the multi-temporal timeline progression and cumulative urban expansion from T1 to T3",
        images=[t1, t2, t3]
    )

    assert res.task == "temporal_sequence"
    assert "EarthDial-4B" in res.selected_model
    assert_confidence_contract(res.confidence)
    assert res.temporal_events is not None
    assert len(res.temporal_events) == 2
    assert res.temporal_events[0]["transition"] == "T1 \u2192 T2"
    assert res.temporal_events[1]["transition"] == "T2 \u2192 T3"
    assert "cumulative" in res.answer.lower()
    assert res.evidence_url is not None
    assert_boxes_contract(res.boxes, t1.width, t1.height)


def test_query_decomposition_grounding_and_vqa(tmp_path, agent):
    """Test query decomposition when query asks for both localization and counting."""
    img = make_real_envelope(tmp_path, "img_decomp_01", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_decomp",
        query="Highlight the harbor pier and count the cargo vessels docked there",
        images=[img]
    )

    assert res.is_decomposed is True
    assert res.decomposition_reasoning is not None
    assert len(res.subtasks) >= 2
    assert any(s["tool"] == "rs-ground" for s in res.subtasks)
    assert any(s["tool"] == "rs-vqa" for s in res.subtasks)


def test_explainable_6_phase_trace(tmp_path, agent):
    """Test that all 6 phases are present in the explainable execution trace."""
    img = make_real_envelope(tmp_path, "img_trace_01", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_trace6",
        query="Describe the land cover",
        images=[img]
    )

    assert res.trace_view is not None
    step_names = [s.step_name for s in res.trace_view.steps]
    expected_phases = ["InputValidation", "TaskIdentification", "ModelSelection", "Execution", "EvidenceCollection", "FinalResult"]
    for phase in expected_phases:
        assert phase in step_names, f"Phase '{phase}' missing from trace steps: {step_names}"


def test_aggregator_reports_raw_confidence_without_temperature_scaling():
    """New contract: confidence is the raw aggregate; no post-hoc temperature scaling."""
    single = ToolOutput(
        tool_name="rs-caption",
        model_name="EarthDial-4B",
        text="A scene description.",
        confidence=0.62,
        evidence_type="analysed_image",
    )
    agg_single = OutputAggregator.aggregate([single])
    assert agg_single.confidence == agg_single.raw_confidence == 0.62
    assert agg_single.calibration_method == "none"
    assert agg_single.calibration_temperature is None
    # Raw 0.62 is below the 0.65 threshold, so the low-confidence flag must fire.
    assert agg_single.uncertainty_flag is True

    high = ToolOutput(
        tool_name="rs-vqa",
        model_name="EarthDial-4B",
        text="Answer.",
        confidence=0.80,
        evidence_type="analysed_image",
    )
    low = ToolOutput(
        tool_name="opt-sar-fusion",
        model_name="DOFA ViT-B",
        text="Cross-check.",
        confidence=0.20,
        evidence_type="opt_sar_pair",
        metadata={"honesty_gate_triggered": True},
    )
    agg_multi = OutputAggregator.aggregate([high, low])
    assert agg_multi.confidence == agg_multi.raw_confidence == 0.5
    assert agg_multi.calibration_method == "none"
    assert agg_multi.calibration_temperature is None
    assert agg_multi.conflict_detected is True
    assert agg_multi.uncertainty_flag is True


def test_vessel_count_reports_actual_count_not_fabricated(tmp_path, agent):
    """High finding: the maritime count is the real cluster count, never a floor of 3."""
    img = make_real_envelope(tmp_path, "img_vessel_count", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_vessel",
        query="How many cargo vessels are docked in the port basin?",
        images=[img],
    )

    assert res.task == "vqa"
    vessel_ev = [ev for ev in res.evidence if ev.get("type") == "vessel_count"]
    assert vessel_ev, "expected a vessel_count evidence item for this maritime query"
    count = int(vessel_ev[0]["description"].split()[1])
    assert count == 1
    assert count < 3  # the removed clamp would have forced at least 3
    assert "No cargo vessels detected on the water" not in res.answer

