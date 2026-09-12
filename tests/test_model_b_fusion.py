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
from app.runtime.manager import ModelRuntimeManager
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


def test_optical_sar_fusion_tool_integration():
    """Verify opt-sar-fusion tool executes and populates Model B provenance."""
    mgr = ModelRuntimeManager()
    descriptor = {
        "name": "opt-sar-fusion",
        "model_key": "dofa-fusion",
        "model_name": "DOFA ViT-B + Head + EarthDial LLM (Model B)",
        "slot": "S4",
        "load_group": "encoder"
    }
    tool = OpticalSARFusionTool(descriptor, mgr)

    opt_env = ImageMetadataEnvelope(
        image_id="img_opt_01",
        session_id="test_sess",
        filepath="mock_opt.png",
        filename="img_opt_01.png",
        sha256="a" * 64,
        file_size_bytes=1024,
        modality="optical",
        width=512,
        height=512,
        bands=3,
        dtype="uint8",
        is_georeferenced=False,
        crs=None,
        bounds=None
    )
    sar_env = ImageMetadataEnvelope(
        image_id="img_sar_01",
        session_id="test_sess",
        filepath="mock_sar.png",
        filename="img_sar_01.png",
        sha256="b" * 64,
        file_size_bytes=1024,
        modality="sar",
        width=512,
        height=512,
        bands=1,
        dtype="uint8",
        is_georeferenced=False,
        crs=None,
        bounds=None
    )

    out = tool.invoke(
        inputs={"images": [opt_env, sar_env], "query": "identify built-up and water regions"},
        parameters={"max_new_tokens": 128}
    )

    assert out.tool_name == "opt-sar-fusion"
    assert out.confidence >= 0.90
    assert "optical" in [e["type"] for e in out.evidence]
    assert "sar" in [e["type"] for e in out.evidence]
    assert "joint" in [e["type"] for e in out.evidence]
    assert "fine_tuned_fusion_head_present" in out.metadata
