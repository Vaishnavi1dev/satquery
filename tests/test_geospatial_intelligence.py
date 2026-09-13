import json
import numpy as np
from PIL import Image
from pathlib import Path

from app.data.ingestion import (
    ImageMetadataEnvelope,
    ImageIngestionService,
    pixel_box_to_geo,
    boxes_to_geojson,
)
from app.evidence.renderer import EvidenceRenderer
from app.storage.sandbox import StorageSandbox
from app.registry.store import ToolRegistryStore
from app.agent.controller import AgentController

# The old fabricated default AOI (Hyderabad). It must never appear for imagery
# that does not actually carry georeferencing.
FABRICATED_DEFAULT_AOI = [78.4500, 17.3500, 78.5500, 17.4500]


def make_test_envelope(
    tmp_path: Path,
    image_id: str,
    modality: str = "optical",
    bounds=None,
    crs=None,
) -> ImageMetadataEnvelope:
    img_path = tmp_path / f"{image_id}.png"
    if modality == "sar":
        arr = (np.random.rand(256, 256) * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    else:
        arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    pil_img.save(img_path)

    generic_sensor = {
        "optical": "Optical (RGB)",
        "multispectral": "Multispectral",
        "sar": "SAR (radar)",
    }[modality]

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
        sensor=generic_sensor,
        crs=crs,
        resolution_m=None,
        bounds=bounds,
        geo_bbox=bounds,
    )


def test_non_georeferenced_envelope_has_no_fabricated_geo(tmp_path):
    """A plain image must expose no AOI, CRS, resolution, or named satellite."""
    env = make_test_envelope(tmp_path, "no_geo")

    assert env.bounds is None
    assert env.geo_bbox is None
    assert env.crs is None
    assert env.resolution_m is None
    assert env.is_georeferenced is False

    assert env.sensor is not None
    for specific in ("Sentinel", "Cartosat", "RISAT", "Landsat", "EOS-04"):
        assert specific.lower() not in env.sensor.lower()

    assert pixel_box_to_geo([10, 10, 128, 128], env) is None
    assert env.bounds != FABRICATED_DEFAULT_AOI
    assert env.geo_bbox != FABRICATED_DEFAULT_AOI


def test_ingested_jpeg_is_not_georeferenced(tmp_path):
    """JPEG ingestion must not inject a default Hyderabad AOI or EPSG:4326."""
    src = tmp_path / "scene.jpg"
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(src, format="JPEG")
    payload = src.read_bytes()

    svc = ImageIngestionService()
    env = svc.ingest_image(payload, "scene.jpg", tmp_path / "store" / "scene.jpg")

    assert env.bounds is None
    assert env.geo_bbox is None
    assert env.crs is None
    assert env.resolution_m is None
    assert env.is_georeferenced is False
    assert env.sensor in ("Optical (RGB)", "Multispectral", "SAR (radar)")
    assert env.bounds != FABRICATED_DEFAULT_AOI
    assert pixel_box_to_geo([0, 0, env.width // 2, env.height // 2], env) is None

    serialized = json.dumps(env.model_dump(), default=str)
    assert "78.45" not in serialized
    assert "EPSG:4326" not in serialized


def test_pixel_box_to_geo_translation(tmp_path):
    """Real file-provided bounds still map pixel boxes to WGS84 coordinates."""
    env = make_test_envelope(
        tmp_path, "test_geo", bounds=[78.400, 17.300, 78.500, 17.400], crs="EPSG:4326"
    )
    pixel_box = [64, 64, 192, 192]  # Center zone (25% to 75%)

    geo_info = pixel_box_to_geo(pixel_box, env)

    assert geo_info is not None
    assert geo_info["pixel_box"] == pixel_box
    min_lon, min_lat, max_lon, max_lat = geo_info["geo_box"]

    assert 78.400 < min_lon < max_lon < 78.500
    assert 17.300 < min_lat < max_lat < 17.400
    assert "°N" in geo_info["formatted_coords"]
    assert "°E" in geo_info["formatted_coords"]
    assert geo_info["crs"] == "EPSG:4326"


def test_boxes_to_geojson_generation(tmp_path):
    """Real bounds produce a standard GeoJSON FeatureCollection."""
    env = make_test_envelope(
        tmp_path, "test_geojson", bounds=[78.400, 17.300, 78.500, 17.400], crs="EPSG:4326"
    )
    boxes = [[20, 20, 100, 100], [120, 120, 200, 200]]

    geojson_data = boxes_to_geojson(boxes, env, label="Test Target")

    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) == 2
    f1 = geojson_data["features"][0]
    assert f1["geometry"]["type"] == "Polygon"
    assert len(f1["geometry"]["coordinates"][0]) == 5  # Closed ring
    assert f1["properties"]["label"] == "Test Target"


def test_boxes_to_geojson_without_bounds_emits_no_features(tmp_path):
    """No georeferencing means no geographic features and no default coordinates."""
    env = make_test_envelope(tmp_path, "test_geojson_nogeo")

    geojson_data = boxes_to_geojson([[20, 20, 100, 100]], env)

    assert geojson_data["type"] == "FeatureCollection"
    assert geojson_data["features"] == []
    assert "78.45" not in json.dumps(geojson_data)
    assert "17.35" not in json.dumps(geojson_data)


def test_render_difference_heatmap(tmp_path):
    """Difference heatmap overlay and transparent mask are properly generated."""
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
    """SAR bi-temporal change keeps real pixel boxes but fabricates no WGS84 geometry."""
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
    # Non-georeferenced inputs: no GeoJSON and no fabricated WGS84 coordinates.
    assert res.geojson_url is None
    assert res.geo_boxes is None
    assert "78.45" not in res.answer
    assert "WGS84" not in res.answer
    for ev in res.evidence:
        assert ev.get("geo_coordinates") is None


def test_multispectral_burn_scar_execution(tmp_path):
    """Multispectral change analysis reports no geo footprints without real bounds."""
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
    assert res.boxes is not None and len(res.boxes) > 0
    assert res.geojson_url is None
    assert res.geo_boxes is None
    assert "78.45" not in res.answer


def test_grounding_honesty_gate_produces_no_fabricated_region(tmp_path):
    """When grounding fails, the controller must not synthesize a fallback box."""
    sandbox = StorageSandbox(root_dir=tmp_path / "sandbox")
    sandbox.create_session("sess_gate")
    registry = ToolRegistryStore()
    agent = AgentController(registry=registry, sandbox=sandbox)

    img = make_test_envelope(tmp_path, "img_gate", modality="optical")
    res = agent.execute_query(
        session_id="sess_gate",
        query="Highlight and locate the volcano in this scene",
        images=[img]
    )

    assert res.task == "grounding"
    assert not res.boxes  # None or empty, never a fabricated 12%-88% region
    assert res.geojson_url is None
    assert res.evidence_url is not None  # analysed image still rendered without boxes


def test_autonomous_modality_detection():
    """Autonomous detection of Optical, Multispectral, and SAR modalities."""
    svc = ImageIngestionService()

    # 1. 12-band Sentinel-2 datacube -> multispectral
    mod_msi = svc.detect_modality("datacube.tif", (256, 256, 12), {})
    assert mod_msi == "multispectral"

    # 2. 3-band natural color RGB image -> optical
    color_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    color_rgb[:, :, 0] = 40   # R
    color_rgb[:, :, 1] = 180  # G (high green vegetation)
    color_rgb[:, :, 2] = 70   # B
    mod_opt = svc.detect_modality("scene.png", color_rgb.shape, {}, data=color_rgb)
    assert mod_opt == "optical"

    # 3. 3-band grayscale speckled image (SAR radar product encoded as RGB) -> sar
    np.random.seed(42)
    speckle_ch = np.random.exponential(scale=50.0, size=(100, 100)).astype(np.uint8)
    sar_rgb = np.stack([speckle_ch, speckle_ch, speckle_ch], axis=-1)
    mod_sar = svc.detect_modality("observation.png", sar_rgb.shape, {}, data=sar_rgb)
    assert mod_sar == "sar"

    # 4. Explicit SAR markers in filename
    mod_fn_sar = svc.detect_modality("sentinel1_vv_cband.tif", (100, 100, 1), {})
    assert mod_fn_sar == "sar"

    # 5. Generic, non-satellite platform labels
    assert svc.generic_sensor_label("optical") == "Optical (RGB)"
    assert svc.generic_sensor_label("multispectral") == "Multispectral"
    assert svc.generic_sensor_label("sar") == "SAR (radar)"
    assert svc.generic_sensor_label("unknown") is None
