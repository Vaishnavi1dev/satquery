import pytest

from app.data.ingestion import ImageMetadataEnvelope
from app.data.pair_validator import ValidationError
from app.agent.classifier import TaskClassifier
from app.agent.validator import AgentInputValidator


SEQUENCE_QUERY = (
    "Analyze the multi-temporal timeline progression and cumulative land "
    "transformation from T1 to T3."
)
CHANGE_QUERY = "What changed between these two dates, and where did the change occur?"
CAPTION_QUERY = "Describe the land-cover and major visible objects in this image."
FUSION_QUERY = "Use the optical and SAR images together to identify built-up and water-covered regions."


def make_env(image_id: str, modality: str = "optical") -> ImageMetadataEnvelope:
    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=f"{image_id}.png",
        filepath=f"{image_id}.png",
        width=256,
        height=256,
        bands=1 if modality == "sar" else 3,
        dtype="uint8",
        file_size_bytes=1024,
        sha256="0" * 64,
        modality=modality,
        sensor={"optical": "Optical (RGB)", "multispectral": "Multispectral", "sar": "SAR (radar)"}[modality],
        crs=None,
        resolution_m=None,
        bounds=None,
        geo_bbox=None,
    )


def classify(query, images):
    return TaskClassifier.classify(query, images)[0]


def test_sequence_query_with_one_image_is_temporal_sequence():
    assert classify(SEQUENCE_QUERY, [make_env("o1")]) == "temporal_sequence"


def test_sequence_query_with_two_images_is_temporal_sequence():
    assert classify(SEQUENCE_QUERY, [make_env("o1"), make_env("o2")]) == "temporal_sequence"


def test_change_query_with_one_image_is_change_vqa():
    assert classify(CHANGE_QUERY, [make_env("o1")]) == "change_vqa"


def test_valid_two_optical_change_is_change_vqa():
    assert classify(CHANGE_QUERY, [make_env("o1"), make_env("o2")]) == "change_vqa"


def test_valid_two_optical_sar_fusion_is_opt_sar_fusion():
    imgs = [make_env("o1", "optical"), make_env("s1", "sar")]
    assert classify(FUSION_QUERY, imgs) == "opt_sar_fusion"


def test_valid_three_optical_sequence_is_temporal_sequence():
    imgs = [make_env("o1"), make_env("o2"), make_env("o3")]
    assert classify(SEQUENCE_QUERY, imgs) == "temporal_sequence"


def test_valid_single_optical_caption():
    assert classify(CAPTION_QUERY, [make_env("o1")]) == "caption"


def test_two_image_change_classification_passes_validator():
    imgs = [make_env("o1"), make_env("o2")]
    task = classify(CHANGE_QUERY, imgs)
    assert task == "change_vqa"
    AgentInputValidator.validate(task, imgs)


def test_validator_rejects_sequence_with_one_image_message():
    with pytest.raises(ValidationError) as exc:
        AgentInputValidator.validate("temporal_sequence", [make_env("o1")])
    assert exc.value.code == "VAL_INVALID_INPUT_COUNT"
    assert "at least 3 sequential epochs" in exc.value.message
    assert "Provided: 1 image" in exc.value.message


def test_validator_rejects_change_with_one_image_message():
    with pytest.raises(ValidationError) as exc:
        AgentInputValidator.validate("change_vqa", [make_env("o1")])
    assert exc.value.code == "VAL_INVALID_INPUT_COUNT"
    assert "bi-temporal pair (2 images)" in exc.value.message
    assert "Provided: 1 image" in exc.value.message


def test_validator_rejects_opt_sar_fusion_with_one_image_message():
    with pytest.raises(ValidationError) as exc:
        AgentInputValidator.validate("opt_sar_fusion", [make_env("o1")])
    assert exc.value.code == "VAL_INVALID_INPUT_COUNT"
    assert "optical + SAR pairing (2 images)" in exc.value.message
    assert "Provided: 1 image" in exc.value.message


def test_validate_endpoint_rejects_sequence_with_one_image(monkeypatch):
    from app.api import routes

    single = make_env("o1")
    monkeypatch.setattr(routes, "_get_envelope", lambda sid, iid: single if iid == "o1" else None)

    res = routes.validate_inputs(
        routes.ValidateRequest(session_id="sess", image_ids=["o1"], query=SEQUENCE_QUERY)
    )
    assert res["valid"] is False
    assert res["error_code"] == "VAL_INVALID_INPUT_COUNT"
    assert "at least 3" in res["message"]


def test_validate_endpoint_accepts_change_with_two_images(monkeypatch):
    from app.api import routes

    pairs = {"o1": make_env("o1"), "o2": make_env("o2")}
    monkeypatch.setattr(routes, "_get_envelope", lambda sid, iid: pairs.get(iid))

    res = routes.validate_inputs(
        routes.ValidateRequest(session_id="sess", image_ids=["o1", "o2"], query=CHANGE_QUERY)
    )
    assert res["valid"] is True
    assert res["task"] == "change_vqa"


def test_validate_endpoint_without_query_keeps_count_based_behavior(monkeypatch):
    from app.api import routes

    single = make_env("o1")
    monkeypatch.setattr(routes, "_get_envelope", lambda sid, iid: single if iid == "o1" else None)

    res = routes.validate_inputs(routes.ValidateRequest(session_id="sess", image_ids=["o1"]))
    assert res["valid"] is True
    assert res["pair_type"] == "single"
