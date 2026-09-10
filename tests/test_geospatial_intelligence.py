import json
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from app.data.ingestion import ImageMetadataEnvelope, pixel_box_to_geo, boxes_to_geojson
from app.evidence.renderer import EvidenceRenderer
from app.storage.sandbox import StorageSandbox
from app.registry.store import ToolRegistryStore
from app.agent.controller import AgentController


def make_test_envelope(
    tmp_path: Path, 
    image_id: str, 
    modality: str = "optical", 
    bounds=None
) -> ImageMetadataEnvelope:
    img_path = tmp_path / f"{image_id}.png"
    # Create test synthetic image
    if modality == "sar":
        arr = (np.random.rand(256, 256) * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    else:
        arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    pil_img.save(img_path)

    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=f"{image_id}.png",
        filepath=str(img_path),
        width=256,
        height=256,
        bands=1 if modality == "sar" else 3,
        dtype="uint8",
        file_size_bytes=1024,
        sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        modality=modality,
        sensor="Sentinel-2" if modality == "multispectral" else ("RISAT-1" if modality == "sar" else "Cartosat-2S"),
        crs="EPSG:4326",
        resolution_m=10.0,
        bounds=bounds or [78.4500, 17.3500, 78.5500, 17.4500]
    )


def test_pixel_box_to_geo_translation(tmp_path):
    """Verifies that pixel bounding boxes are mapped to real-world WGS84 Lat/Lon coordinates."""
    env = make_test_envelope(tmp_path, "test_geo", bounds=[78.400, 17.300, 78.500, 17.400])
    pixel_box = [64, 64, 192, 192] # Center zone (25% to 75%)

    geo_info = pixel_box_to_geo(pixel_box, env)

    assert "pixel_box" in geo_info
    assert geo_info["pixel_box"] == pixel_box
    assert "geo_box" in geo_info
    min_lon, min_lat, max_lon, max_lat = geo_info["geo_box"]

    # Verify interpolation within bounds
    assert 78.400 < min_lon < max_lon < 78.500
    assert 17.300 < min_lat < max_lat < 17.400
    assert "°N" in geo_info["formatted_coords"]
    assert "°E" in geo_info["formatted_coords"]


def test_boxes_to_geojson_generation(tmp_path):
    """Verifies standard GeoJSON FeatureCollection generation."""
    env = make_test_envelope(tmp_path, "test_geojson")
    boxes = [[20, 20, 100, 100], [120, 120, 200, 200]]

    geojson_data = boxes_to_geojson(boxes, env, label="Test Target")

    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) == 2
    f1 = geojson_data["features"][0]
    assert f1["geometry"]["type"] == "Polygon"
    assert len(f1["geometry"]["coordinates"][0]) == 5 # Closed ring
    assert f1["properties"]["label"] == "Test Target"


def test_render_difference_heatmap(tmp_path):
    """Verifies that difference heatmap overlay and transparent mask are properly generated."""
    sandbox = StorageSandbox(root_dir=tmp_path / "sandbox")
    sandbox.create_session("sess_01")
    renderer = EvidenceRenderer(sandbox)

    env1 = make_test_envelope(tmp_path, "env_t1")
    env2 = make_test_envelope(tmp_path, "env_t2")

    res = renderer.render_difference_heatmap("sess_01", env1, env2, change_boxes=[[50, 50, 150, 150]])

    assert res["overlay_path"].exists()
    assert res["mask_path"].exists()
    assert res["overlay_path"].name.endswith(".png")
    assert res["mask_path"].name.endswith(".png")


def test_sar_bitemporal_change_execution(tmp_path):
    """Verifies that SAR bi-temporal flood detection operates via EarthDial-4B with coordinates."""
    sandbox = StorageSandbox(root_dir=tmp_path / "sandbox")
    sandbox.create_session("sess_sar")
    registry = ToolRegistryStore()
    agent = AgentController(registry=registry, sandbox=sandbox)

    sar1 = make_test_envelope(tmp_path, "sar_t1", modality="sar")
    sar2 = make_test_envelope(tmp_path, "sar_t2", modality="sar")

    res = agent.execute_query(
        session_id="sess_sar",
        query="What changed between these two dates? Has flood water increased?",
        images=[sar1, sar2]
    )

    assert res.task == "change_vqa"
    assert "EarthDial" in res.selected_model
    assert "inundation" in res.answer.lower() or "flood" in res.answer.lower()
    assert res.boxes is not None and len(res.boxes) > 0
    assert res.diff_mask_url is not None
    assert res.geojson_url is not None


def test_multispectral_burn_scar_execution(tmp_path):
    """Verifies that Multispectral bi-temporal burn scar scan operates with GeoJSON and coordinates."""
    sandbox = StorageSandbox(root_dir=tmp_path / "sandbox")
    sandbox.create_session("sess_ms")
    registry = ToolRegistryStore()
    agent = AgentController(registry=registry, sandbox=sandbox)

    ms1 = make_test_envelope(tmp_path, "ms_t1", modality="multispectral")
    ms2 = make_test_envelope(tmp_path, "ms_t2", modality="multispectral")

    res = agent.execute_query(
        session_id="sess_ms",
        query="Map wildfire burn scars, canopy degradation, and vegetative index drop.",
        images=[ms1, ms2]
    )

    assert res.task == "change_vqa"
    assert "burn" in res.answer.lower() or "wildfire" in res.answer.lower()
    assert res.geojson_url is not None
    assert res.geo_boxes is not None

