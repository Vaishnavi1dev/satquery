import json
import pytest
from pathlib import Path
from app.data.ingestion import ImageMetadataEnvelope
from app.registry.store import ToolRegistryStore
from app.agent.controller import AgentController, QueryExecutionResult
from app.agent.aggregator import OutputAggregator
from app.tools.base import ToolOutput
from app.storage.sandbox import StorageSandbox


def make_dummy_envelope(image_id: str, modality: str = "optical", filepath: str = "dummy.png") -> ImageMetadataEnvelope:
    return ImageMetadataEnvelope(
        image_id=image_id,
        session_id="test_sess",
        filepath=filepath,
        filename=f"{image_id}.png",
        sha256="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        file_size_bytes=1024,
        modality=modality,
        width=512,
        height=512,
        bands=3 if modality in ("optical", "multispectral") else 1,
        dtype="uint8",
        is_georeferenced=False,
        crs=None,
        bounds=None
    )


@pytest.fixture
def agent():
    registry = ToolRegistryStore()
    sandbox = StorageSandbox()
    return AgentController(registry=registry, sandbox=sandbox)


def test_single_image_grounding_evidence(agent):
    """Requirement 2: Grounding returns bounding boxes with source_model earthdial."""
    img = make_dummy_envelope("img_opt_01", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_01",
        query="Highlight and locate the water body",
        images=[img]
    )

    assert isinstance(res, QueryExecutionResult)
    assert res.task == "grounding"
    assert "EarthDial-4B" in res.selected_model
    assert res.confidence is not None and res.confidence > 0.8
    assert res.boxes is not None and len(res.boxes) > 0

    # Evidence verification
    assert len(res.evidence) > 0
    box_ev = res.evidence[0]
    assert box_ev["type"] == "bounding_box"
    assert box_ev["source_model"] == "earthdial"
    assert "region" in box_ev
    assert len(box_ev["region"]) == 4
    assert box_ev["score"] is not None

    # Trace verification
    assert isinstance(res.trace, list)
    assert any("grounding" in t.lower() for t in res.trace)


def test_single_image_captioning_no_hallucinated_evidence(agent):
    """Requirement 2: Captioning must not invent fake bounding boxes or regions."""
    img = make_dummy_envelope("img_opt_02", modality="optical")
    res = agent.execute_query(
        session_id="sess_test_02",
        query="Describe the land-cover distribution in this scene",
        images=[img]
    )

    assert res.task == "caption"
    assert len(res.answer) > 20
    # Must NOT invent visual regions/bounding boxes
    assert res.boxes is None
    assert res.evidence == []


def test_single_image_vqa_evidence(agent):
    """Requirement 2: VQA preserves answer and confidence."""
    img = make_dummy_envelope("img_sar_01", modality="sar")
    res = agent.execute_query(
        session_id="sess_test_03",
        query="What is the backscatter characteristic of the urban area?",
        images=[img]
    )

    assert res.task == "vqa"
    assert "backscatter" in res.answer.lower()
    assert res.confidence is not None
    assert len(res.evidence) == 1
    assert res.evidence[0]["source_model"] == "earthdial"
    assert res.evidence[0]["modality"] == "sar"


def test_cross_modal_dofa_evidence_provenance(agent):
    """Requirement 3: DOFA distinguishes optical, sar, and joint contributions."""
    opt_img = make_dummy_envelope("img_opt_pair", modality="optical")
    sar_img = make_dummy_envelope("img_sar_pair", modality="sar")

    res = agent.execute_query(
        session_id="sess_test_04",
        query="Identify built-up areas and water bodies using optical and SAR together",
        images=[opt_img, sar_img]
    )

    assert res.task == "opt_sar_fusion"
    assert "DOFA" in res.selected_model
    assert res.confidence is not None

    types = [e["type"] for e in res.evidence]
    assert "optical" in types
    assert "sar" in types
    assert "joint" in types

    for ev in res.evidence:
        assert ev["source_model"] == "dofa"

    # Verify execution trace contains cross-modal steps
    trace_text = " ".join(res.trace).lower()
    assert "cross-modal" in trace_text or "optical" in trace_text


def test_bitemporal_earthdial_evidence(agent):
    """Requirement 4: EarthDial-4B returns change regions, temporal relationship, and cycle consistency."""
    t1_img = make_dummy_envelope("img_t1", modality="optical")
    t2_img = make_dummy_envelope("img_t2", modality="optical")

    res = agent.execute_query(
        session_id="sess_test_05",
        query="What changed between Observation T1 and Observation T2? Has the built-up area increased?",
        images=[t1_img, t2_img]
    )

    assert res.task == "change_vqa"
    assert "EarthDial-4B" in res.selected_model
    assert res.confidence is not None

    # Check evidence structure
    assert len(res.evidence) > 0
    change_ev = res.evidence[0]
    assert change_ev["type"] == "change_region"
    assert change_ev["source_model"] == "earthdial-4b"
    assert change_ev["time_from"] == "T1"
    assert change_ev["time_to"] == "T2"
    assert change_ev["change_detected"] is True
    assert "cycle_consistency" in change_ev
    assert "change_type" in change_ev

    # Check execution trace
    trace_text = " ".join(res.trace).lower()
    assert "bi-temporal" in trace_text or "compared t1 and t2" in trace_text


def test_common_result_json_serializable(agent):
    """Requirement 1: QueryExecutionResult is strictly JSON-serializable."""
    img = make_dummy_envelope("img_serial", modality="optical")
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


def test_agent_multi_specialist_composition_execution(agent):
    """Requirement 9: End-to-end query scheduling both EarthDial-4B and DOFA when asking for change and SAR confirmation."""
    opt_img = make_dummy_envelope("img_multi_opt", modality="optical")
    sar_img = make_dummy_envelope("img_multi_sar", modality="sar")

    res = agent.execute_query(
        session_id="sess_test_multi",
        query="Has construction increased between these two dates, and does the SAR evidence support it?",
        images=[opt_img, sar_img]
    )

    assert res.task == "change_vqa"
    assert "Specialist 1" in res.answer or "Primary Result" in res.answer or "Observation" in res.answer
    # Should contain evidence from both change model (earthdial-4b) and dofa
    models_in_evidence = {e["source_model"] for e in res.evidence}
    assert "earthdial-4b" in models_in_evidence
    assert "dofa" in models_in_evidence
    assert any("SAR" in t or "cross-modal" in t.lower() or "dofa" in t.lower() for t in res.trace)


def test_multi_temporal_sequence_analysis(agent):
    """Test multi-temporal sequence analysis across T1 -> T2 -> T3 epochs."""
    t1 = make_dummy_envelope("img_epoch_t1", modality="optical")
    t2 = make_dummy_envelope("img_epoch_t2", modality="optical")
    t3 = make_dummy_envelope("img_epoch_t3", modality="optical")

    res = agent.execute_query(
        session_id="sess_test_seq",
        query="Analyze the multi-temporal timeline progression and cumulative urban expansion from T1 to T3",
        images=[t1, t2, t3]
    )

    assert res.task == "temporal_sequence"
    assert "EarthDial-4B" in res.selected_model
    assert res.temporal_events is not None
    assert len(res.temporal_events) == 2
    assert res.temporal_events[0]["transition"] == "T1 → T2"
    assert res.temporal_events[1]["transition"] == "T2 → T3"
    assert "cumulative" in res.answer.lower()
    assert res.evidence_url is not None


def test_query_decomposition_grounding_and_vqa(agent):
    """Test query decomposition when query asks for both localization and counting."""
    img = make_dummy_envelope("img_decomp_01", modality="optical")
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


def test_explainable_6_phase_trace(agent):
    """Test that all 6 phases are present in the explainable execution trace."""
    img = make_dummy_envelope("img_trace_01", modality="optical")
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

