"""
Tests for Model B (DOFA ViT-B + Cross-Modal Attention Fusion Head)
Slot S4: opt-sar-fusion
"""

import os
import json
import pytest
import tempfile

try:
    import torch
    import torch.nn as nn
    from training.dofa.train_dofa_fusion import (
        WavelengthHypernetwork,
        FrozenDOFAEncoder,
        CrossModalFusionHead,
        ModelBDOFAFusion,
        CheckpointManager,
        DOFAFusionDataset,
        collate_dofa_batch
    )
except ImportError:
    torch = None
    nn = None

from training.dofa.prepare_dofa_pairs import download_and_prepare_dofa_pairs, SENSOR_WAVELENGTHS
from app.runtime.manager import ModelRuntimeManager, ModelExecutionError
from app.tools.opt_sar_fusion import OpticalSARFusionTool
from app.data.ingestion import ImageMetadataEnvelope

require_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed in the local environment")


@require_torch
def test_dofa_wavelength_hypernetwork():
    """Verify Fourier wavelength modulation produces correct dimensions."""
    hypernet = WavelengthHypernetwork(embed_dim=768, num_harmonics=16)
    wls = torch.tensor([[0.490, 0.560, 0.665, 0.842]], dtype=torch.float32)  # [1, 4]
    out = hypernet(wls)
    assert out.shape == (1, 4, 768)


@require_torch
def test_frozen_dofa_encoder_parameters():
    """Verify DOFA encoder parameters are 100% frozen per DEC-004."""
    model = ModelBDOFAFusion(embed_dim=768)
    for param in model.encoder.parameters():
        assert not param.requires_grad, "DOFA encoder parameters must be completely frozen!"


@require_torch
def test_cross_modal_fusion_head_forward():
    """Verify Multi-Head Cross-Attention fuses optical and SAR tokens."""
    fusion_head = CrossModalFusionHead(embed_dim=768, num_heads=4, vocab_size=1000)
    opt_tokens = torch.randn(2, 196, 768)
    sar_tokens = torch.randn(2, 196, 768)

    fused = fusion_head(opt_tokens, sar_tokens)
    assert fused.shape == (2, 196, 768)

    logits = fusion_head.lm_head(fused)
    assert logits.shape == (2, 196, 1000)


@require_torch
def test_model_b_end_to_end_forward():
    """Verify ModelB forward pass with loss computation."""
    model = ModelBDOFAFusion(embed_dim=768, vocab_size=1000)
    opt_imgs = torch.randn(2, 3, 224, 224)
    sar_imgs = torch.randn(2, 3, 224, 224)
    opt_wls = torch.tensor([[0.49, 0.56, 0.665], [0.49, 0.56, 0.665]])
    sar_wls = torch.tensor([[56000.0, 56000.0, 56000.0], [56000.0, 56000.0, 56000.0]])
    targets = torch.randint(0, 1000, (2, 196))

    out = model(opt_imgs, sar_imgs, opt_wls, sar_wls, targets=targets)
    assert "logits" in out
    assert "fused_embeds" in out
    assert out["loss"] is not None
    assert out["loss"].item() > 0.0


@require_torch
def test_checkpoint_manager_save_and_load():
    """Verify rolling checkpoint save, manifest emission, and weight restoration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(output_dir=tmpdir)
        model = ModelBDOFAFusion(embed_dim=768, vocab_size=1000)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        ckpt_mgr.save_checkpoint(model, optimizer, None, step=150, epoch=2, tag="ckpt_test")

        saved_path = os.path.join(tmpdir, "ckpt_test")
        assert os.path.exists(os.path.join(saved_path, "fusion_head.pt"))
        assert os.path.exists(os.path.join(saved_path, "adapter_manifest.json"))
        assert os.path.exists(os.path.join(saved_path, "training_state.pt"))

        # Load back
        model_new = ModelBDOFAFusion(embed_dim=768, vocab_size=1000)
        resumed_step = ckpt_mgr.load_checkpoint(saved_path, model_new)
        assert resumed_step == 150


def test_dofa_pair_data_preparation():
    """Verify generation of calibrated Optical + SAR pair instructions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "test_pairs.json")
        img_dir = os.path.join(tmpdir, "images")

        records = download_and_prepare_dofa_pairs(
            output_json=out_json,
            image_dir=img_dir,
            num_samples=4
        )

        assert len(records) == 4
        assert os.path.exists(out_json)
        for r in records:
            assert "optical_image" in r and os.path.exists(r["optical_image"])
            assert "sar_image" in r and os.path.exists(r["sar_image"])
            assert "conversations" in r
            assert len(r["conversations"]) == 2
            assert "<image_optical>" in r["conversations"][0]["value"]
            assert "<image_sar>" in r["conversations"][0]["value"]



def _write_fusion_images(tmp_path, opt_variant=0, sar_variant=0):
    """Create small deterministic optical/SAR PNGs so real pixel analysis runs."""
    import numpy as np
    from PIL import Image

    size = 96
    opt_arr = np.zeros((size, size, 3), dtype=np.uint8)
    opt_arr[:, :] = (40, 150, 40)  # vegetation
    if opt_variant % 2 == 0:
        opt_arr[0:size // 5, :] = (225, 225, 225)  # built-up
    else:
        opt_arr[size // 5:size // 2, :] = (225, 225, 225)
    opt_arr[(size * 70) // 96:, :] = (25, 50, 170)  # water
    opt_path = tmp_path / "opt_real.png"
    Image.fromarray(opt_arr).save(opt_path)

    sar_arr = np.full((size, size), 20, dtype=np.uint8)
    band = 200
    if sar_variant % 2 == 0:
        sar_arr[0:size // 5, :] = band
    else:
        sar_arr[size // 5:size // 2, :] = band
    sar_path = tmp_path / "sar_real.png"
    Image.fromarray(sar_arr, mode="L").save(sar_path)
    return str(opt_path), str(sar_path)


def _make_fusion_envelope(image_id, modality, filepath):
    return ImageMetadataEnvelope(
        image_id=image_id,
        session_id="test_sess",
        filepath=filepath,
        filename=os.path.basename(filepath),
        sha256=("a" if modality == "optical" else "b") * 64,
        file_size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 1024,
        modality=modality,
        width=96,
        height=96,
        bands=3 if modality == "optical" else 1,
        dtype="uint8",
        is_georeferenced=False,
        crs=None,
        bounds=None,
    )


def _fusion_tool():
    mgr = ModelRuntimeManager()
    descriptor = {
        "name": "opt-sar-fusion",
        "model_key": "dofa-fusion",
        "model_name": "DOFA ViT-B + Head + EarthDial LLM (Model B)",
        "slot": "S4",
        "load_group": "encoder"
    }
    return OpticalSARFusionTool(descriptor, mgr)


def test_optical_sar_fusion_tool_integration(tmp_path):
    """Verify opt-sar-fusion executes with real imagery and honest provenance."""
    tool = _fusion_tool()

    opt_path, sar_path = _write_fusion_images(tmp_path, opt_variant=0, sar_variant=0)
    opt_env = _make_fusion_envelope("img_opt_01", "optical", opt_path)
    sar_env = _make_fusion_envelope("img_sar_01", "sar", sar_path)

    out = tool.invoke(
        inputs={"images": [opt_env, sar_env], "query": "identify built-up and water regions"},
        parameters={"max_new_tokens": 128}
    )

    assert out.tool_name == "opt-sar-fusion"
    assert out.confidence is None or (isinstance(out.confidence, float) and 0.0 <= out.confidence <= 1.0)
    assert out.metadata["confidence_basis"] in {
        "nominal_model_estimate",
        "measured_pixel_analysis",
        "not_available",
    }
    # Real pixel-derived class fractions must be genuine floats.
    for key in ("water_coverage_pct", "builtup_coverage_pct", "vegetation_coverage_pct"):
        assert isinstance(out.metadata[key], float)
        assert 0.0 <= out.metadata[key] <= 100.0
    assert out.boxes is not None and len(out.boxes) > 0
    for box in out.boxes:
        assert isinstance(box, list) and len(box) == 4
        assert all(isinstance(c, int) for c in box)
        assert 0 <= box[0] <= box[2] <= 96
        assert 0 <= box[1] <= box[3] <= 96
    assert "optical" in [e["type"] for e in out.evidence]
    assert "sar" in [e["type"] for e in out.evidence]
    assert "joint" in [e["type"] for e in out.evidence]
    assert "fine_tuned_fusion_head_present" in out.metadata
    # No fabricated radar decibel values are synthesized.
    assert "db" not in out.text.lower()
    assert "decibel" not in out.text.lower()

    # Unreadable inputs must not silently fabricate numbers either.
    missing_opt = _make_fusion_envelope("missing_opt", "optical", str(tmp_path / "does_not_exist_opt.png"))
    missing_sar = _make_fusion_envelope("missing_sar", "sar", str(tmp_path / "does_not_exist_sar.png"))
    out_missing = tool.invoke(
        inputs={"images": [missing_opt, missing_sar], "query": "identify built-up and water regions"},
        parameters={"max_new_tokens": 16}
    )
    assert out_missing.confidence is None
    assert out_missing.boxes is None
    assert out_missing.metadata["confidence_basis"] == "not_available"
    for key in ("water_coverage_pct", "builtup_coverage_pct", "vegetation_coverage_pct"):
        assert out_missing.metadata[key] is None
    assert "db" not in out_missing.text.lower()
    assert "decibel" not in out_missing.text.lower()


def test_optical_sar_fusion_rejects_more_than_two_images(tmp_path):
    """Only a genuine two-image pair is a valid joint-use request."""
    tool = _fusion_tool()
    opt_path, sar_path = _write_fusion_images(tmp_path)
    opt_env = _make_fusion_envelope("img_opt_x", "optical", opt_path)
    sar_env = _make_fusion_envelope("img_sar_x", "sar", sar_path)
    with pytest.raises(ModelExecutionError) as exc:
        tool.invoke(
            inputs={"images": [opt_env, sar_env, sar_env], "query": "identify built-up and water regions"},
            parameters={"max_new_tokens": 16},
        )
    assert exc.value.code == "MDL_JOINT_USE_VIOLATION"


def test_optical_sar_fusion_unusable_token_confidence_falls_back(tmp_path):
    """A None/string token_confidence must not crash and must fall back to the measurement."""
    tool = _fusion_tool()
    opt_path, sar_path = _write_fusion_images(tmp_path)
    opt_env = _make_fusion_envelope("img_opt_tc", "optical", opt_path)
    sar_env = _make_fusion_envelope("img_sar_tc", "sar", sar_path)

    tool.runtime_mgr.ensure_model_loaded = lambda *a, **k: None
    tool.runtime_mgr.get_checkpoint_dir = lambda *a, **k: None
    tool.runtime_mgr.run_dofa_fusion = lambda *a, **k: {
        "token_confidence": None, "checkpoint_dir": "ckpt", "logits_shape": [1, 196, 1000]
    }
    out = tool.invoke(
        inputs={"images": [opt_env, sar_env], "query": "identify built-up and water regions"},
        parameters={"max_new_tokens": 16},
    )
    assert out.confidence == 0.90
    assert out.metadata["confidence_basis"] == "measured_pixel_analysis"
    assert out.metadata["inference_backend"] == "simulation"

