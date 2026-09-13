"""Robustness tests for the runtime manager and data layer edge cases."""

import json
import threading

import pytest

from app.data.ingestion import ImageMetadataEnvelope
from app.data.pair_validator import PairValidator, ValidationError
from app.runtime.manager import EARTHDIAL_MODEL_KEYS, ModelRuntimeManager


def make_env(image_id, width=64, height=64, modality="optical"):
    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=f"{image_id}.png",
        filepath=f"{image_id}.png",
        width=width,
        height=height,
        bands=3,
        dtype="uint8",
        file_size_bytes=1024,
        sha256="0" * 64,
        modality=modality,
    )


def _fake_earthdial_handle(tmp_path):
    return {
        "kind": "earthdial",
        "model": object(),
        "tokenizer": object(),
        "checkpoint_dir": str(tmp_path),
    }


def test_zero_dimension_pair_validator_raises():
    with pytest.raises(ValidationError) as exc:
        PairValidator.validate_bi_temporal_pair(make_env("a", width=0), make_env("b"))
    assert exc.value.code == "VAL_CORRUPT_IMAGE"

    with pytest.raises(ValidationError):
        PairValidator.validate_bi_temporal_pair(make_env("a"), make_env("b", height=0))

    with pytest.raises(ValidationError):
        PairValidator.validate_temporal_sequence(
            [make_env("a", width=0), make_env("b"), make_env("c")]
        )

    with pytest.raises(ValidationError):
        PairValidator.validate_single(make_env("a", height=-1))

    # Valid inputs still validate.
    assert PairValidator.validate_bi_temporal_pair(make_env("a"), make_env("b")).is_valid


def test_concurrent_ensure_model_loaded_leaves_single_earthdial_resident(monkeypatch, tmp_path):
    manager = ModelRuntimeManager()
    manager.mode = "real"
    monkeypatch.setattr(manager, "_artifact_dir", lambda key: tmp_path)

    # On the unfixed code path both callers reach the loader together, so the barrier
    # releases; on the fixed path the lock serializes them and it simply times out.
    barrier = threading.Barrier(2, timeout=2.0)

    def fake_load(key):
        try:
            barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass
        return _fake_earthdial_handle(tmp_path)

    monkeypatch.setattr(manager, "_load_real_model", fake_load)

    errors = []

    def worker(key):
        try:
            manager.ensure_model_loaded(key, "llm_primary")
        except Exception as exc:  # pragma: no cover - diagnostic only
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(key,))
        for key in ("earthdial-4b", "earthdial-ms")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    resident = [key for key in manager.active_models if key in EARTHDIAL_MODEL_KEYS]
    assert len(resident) == 1, resident


def test_failed_real_load_is_retried_and_error_cleared(monkeypatch, tmp_path):
    manager = ModelRuntimeManager()
    manager.mode = "real"
    monkeypatch.setattr(manager, "_artifact_dir", lambda key: tmp_path)

    attempts = {"count": 0}

    def flaky_load(key):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("checkpoint missing")
        return _fake_earthdial_handle(tmp_path)

    monkeypatch.setattr(manager, "_load_real_model", flaky_load)

    assert manager.ensure_model_loaded("earthdial-4b", "llm_primary") is None
    assert "earthdial-4b" not in manager.active_models
    assert manager.load_errors.get("earthdial-4b")

    handle = manager.ensure_model_loaded("earthdial-4b", "llm_primary")
    assert handle is not None
    assert attempts["count"] == 2
    assert "earthdial-4b" in manager.active_models
    assert "earthdial-4b" not in manager.load_errors


def test_ms_failure_does_not_alias_rgb_handle(monkeypatch, tmp_path):
    manager = ModelRuntimeManager()
    manager.mode = "real"
    monkeypatch.setattr(manager, "_artifact_dir", lambda key: tmp_path)

    def fake_load(key):
        if key == "earthdial-ms":
            raise RuntimeError("ms checkpoint unavailable")
        return _fake_earthdial_handle(tmp_path)

    monkeypatch.setattr(manager, "_load_real_model", fake_load)

    assert manager.ensure_model_loaded("earthdial-ms", "llm_primary") is None
    # Cross-modality aliasing is forbidden: no RGB handle may appear under the MS key.
    assert "earthdial-ms" not in manager.active_models
    assert "earthdial-4b" not in manager.active_models
    assert manager.load_errors.get("earthdial-ms")


def test_rgb_original_alias_is_same_family_only(monkeypatch, tmp_path):
    manager = ModelRuntimeManager()
    manager.mode = "real"
    monkeypatch.setattr(manager, "_artifact_dir", lambda key: tmp_path)

    def fake_load(key):
        if key == "earthdial-original":
            raise RuntimeError("original checkpoint unavailable")
        return _fake_earthdial_handle(tmp_path)

    monkeypatch.setattr(manager, "_load_real_model", fake_load)

    handle = manager.ensure_model_loaded("earthdial-original", "llm_secondary")
    assert handle is not None
    # The RGB original may alias the RGB fine-tuned adapter (same modality family).
    assert manager.active_models["earthdial-original"]["handle"] is handle


def test_artifact_dir_requires_complete_checkpoint(tmp_path):
    manager = ModelRuntimeManager()

    config_only = tmp_path / "config_only"
    config_only.mkdir()
    (config_only / "config.json").write_text(
        json.dumps({"model_type": "internvl"}), encoding="utf-8"
    )

    weights_only = tmp_path / "weights_only"
    weights_only.mkdir()
    (weights_only / "model.safetensors").write_bytes(b"weights")

    broken_config = tmp_path / "broken_config"
    broken_config.mkdir()
    (broken_config / "config.json").write_text("{not json", encoding="utf-8")
    (broken_config / "model.safetensors").write_bytes(b"weights")

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "config.json").write_text(
        json.dumps({"model_type": "internvl"}), encoding="utf-8"
    )
    (complete / "model.safetensors").write_bytes(b"weights")

    assert manager._ready_checkpoint_dir(config_only) is None
    assert manager._ready_checkpoint_dir(weights_only) is None
    assert manager._ready_checkpoint_dir(broken_config) is None
    assert manager._ready_checkpoint_dir(complete) == complete
