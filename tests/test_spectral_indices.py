import pytest
import numpy as np
from app.data.preprocessing import compute_spectral_indices

def test_compute_spectral_indices_synthetic_msi():
    # Create synthetic 4+ band MSI array (H=64, W=64, C=4)
    # Channel 0: Blue, Channel 1: Green, Channel 2: Red, Channel 3: NIR
    raw = np.zeros((64, 64, 4), dtype=np.float32)
    raw[:, :, 0] = 0.2 # Blue
    raw[:, :, 1] = 0.3 # Green
    raw[:, :, 2] = 0.1 # Red
    raw[:, :, 3] = 0.6 # NIR (high vegetation reflectance)

    indices = compute_spectral_indices(raw, modality="multispectral")
    assert indices is not None
    assert "mean_ndvi" in indices
    assert "mean_ndwi" in indices
    assert indices["mean_ndvi"] > 0.6  # (0.6 - 0.1) / (0.6 + 0.1) = 0.714
    assert indices["dense_vegetation_pct"] > 90.0
    assert indices["vegetation_vigor"] == "High Active Canopy"
    assert "ndvi_overlay_b64" in indices
    assert indices["ndvi_overlay_b64"].startswith("data:image/png;base64,")
    assert "ndwi_overlay_b64" in indices
    assert indices["ndwi_overlay_b64"].startswith("data:image/png;base64,")

def test_compute_spectral_indices_rgb():
    # Test with standard 3-channel RGB image (H=64, W=64, C=3)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200 # Green dominant
    rgb[:, :, 0] = 50  # Red
    rgb[:, :, 2] = 50  # Blue

    indices = compute_spectral_indices(rgb, modality="optical")
    assert indices is not None
    assert "mean_ndvi" in indices
    assert "ndvi_overlay_b64" in indices
    assert "ndwi_overlay_b64" in indices


def test_compute_spectral_indices_sar_returns_not_applicable():
    raw = np.zeros((8, 8, 4), dtype=np.float32)
    result = compute_spectral_indices(raw, modality="sar")
    assert result["applicable"] is False
    assert result["error"] == "SPECTRAL_INDICES_NOT_APPLICABLE"
    assert result["mean_ndvi"] is None
    assert result["ndvi_overlay_b64"] is None


def test_compute_spectral_indices_rejects_degenerate_shapes():
    zero_size = np.zeros((0, 0, 4), dtype=np.float32)
    assert compute_spectral_indices(zero_size)["error"] == "SPECTRAL_INDICES_INVALID_INPUT"
    two_d = np.zeros((4, 4), dtype=np.float32)
    assert compute_spectral_indices(two_d, modality="optical")["error"] == "SPECTRAL_INDICES_INVALID_INPUT"
    two_band = np.zeros((4, 4, 2), dtype=np.float32)
    assert compute_spectral_indices(two_band)["error"] == "SPECTRAL_INDICES_INVALID_INPUT"


def test_compute_spectral_indices_masks_non_finite_values():
    import math

    raw = np.zeros((8, 8, 4), dtype=np.float32)
    raw[:, :, 0] = 0.2  # Blue
    raw[:, :, 1] = 0.3  # Green
    raw[:, :, 2] = 0.1  # Red
    raw[:, :, 3] = 0.6  # NIR
    raw[0, 0, 0] = np.nan
    raw[0, 0, 1] = np.inf
    raw[0, 0, 3] = -np.inf

    result = compute_spectral_indices(raw, modality="multispectral")
    assert result["applicable"] is True
    assert math.isfinite(result["mean_ndvi"])
    assert math.isfinite(result["mean_ndwi"])
    assert math.isfinite(result["water_body_pct"])
