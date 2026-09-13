import base64
import hashlib
import io
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
import tifffile
from pydantic import BaseModel, Field

# Explicit decompression-bomb policy: PIL raises DecompressionBombError above this,
# and ``_open_raster`` pre-checks dimensions against it before allocating an array.
MAX_IMAGE_PIXELS = 512 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _corrupt_image(message: str):
    """Build a VAL_CORRUPT_IMAGE ValidationError without importing eagerly.

    ``pair_validator`` imports this module, so the shared ValidationError type is
    imported lazily at call time to avoid a circular import.
    """
    from app.data.pair_validator import ValidationError

    return ValidationError("VAL_CORRUPT_IMAGE", message)


def _finite_bounds(value) -> Optional[List[float]]:
    """Return four finite floats, or None when bounds are absent/malformed.

    Non-finite bounds otherwise propagate NaN/Inf into ``geo_box`` strings and
    GeoJSON, producing invalid JSON and a 500 on ``allow_nan=False``.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bounds = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in bounds):
        return None
    return bounds


def _finite_box(box) -> Optional[List[float]]:
    """Return four finite floats for a pixel box, or None when malformed."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


class ImageMetadataEnvelope(BaseModel):
    image_id: str
    filename: str
    filepath: str
    width: int
    height: int
    bands: int
    dtype: str
    file_size_bytes: int
    sha256: str
    modality: str  # "optical", "multispectral", "sar"
    sensor: Optional[str] = None
    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    geo_bbox: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    nodata_val: Optional[float] = None
    thumbnail_base64: Optional[str] = None
    tags: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_georeferenced(self) -> bool:
        """True only when the file itself supplied genuine geographic bounds."""
        return bool(self.bounds) and len(self.bounds) == 4


class ImageIngestionService:
    """Reads GeoTIFF, TIFF, PNG, and JPEG imagery, extracts metadata, and detects modality."""

    def __init__(self):
        pass

    def compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def detect_modality(self, filename: str, shape: tuple, tags: Dict[str, Any], data: Optional[np.ndarray] = None) -> str:
        """Infer modality from explicit evidence before falling back to heuristics.

        Ordering matters: real satellite filenames routinely contain content words
        such as ``urban``, ``water`` or ``rgb`` that describe the scene, not the
        sensor, so explicit SAR tokens and tag metadata are authoritative. Band count
        is the next signal, and scene-content words are only a last-resort tie-breaker.
        """
        fn_lower = filename.lower()
        bands = shape[2] if len(shape) == 3 else 1

        # 1. Explicit SAR / radar evidence in the filename.
        sar_tokens = ["sentinel-1", "sentinel1", "risat", "_sar", "sar_", "-sar", "c-band", "cband", "radar"]
        is_sar_name = (
            any(marker in fn_lower for marker in sar_tokens)
            or fn_lower.startswith("sar")
            or re.search(r"\bsar\b", fn_lower) is not None
        )

        # 2. SAR polarization / calibration metadata. The word-boundary regex avoids
        #    matching the substring 'sar' inside an unrelated token such as 'isarea'.
        tag_str = str(tags).lower()
        is_sar_tag = (
            any(p in tag_str for p in ["polarisation", "polarization", "c-band", "backscatter", "sigma0", "gamma0", "sentinel-1"])
            or re.search(r"\bsar\b", tag_str) is not None
        )
        if is_sar_name or is_sar_tag:
            return "sar"

        # 3. Genuine optical / multispectral sensor designators (never scene-content
        #    words such as water, urban, rgb, b08 or ndvi).
        optical_tokens = [
            "multispectral", "msi", "sentinel-2", "sentinel2", "ben-ge", "landsat",
            "b04_b08", "optical", "panchromatic", "pan", "cartosat", "hyperspectral",
        ]
        is_optical_name = any(marker in fn_lower for marker in optical_tokens)

        # 4. Band-count consistency with the designators above.
        if bands > 3:
            return "multispectral"
        if bands == 2:
            # Dual-polarization radar (e.g. VV + VH).
            return "sar"
        if bands == 1:
            # Single-band data is SAR unless a real optical designator is present.
            return "optical" if is_optical_name else "sar"

        # bands == 3: content words are only a tie-breaker.
        if is_optical_name and "multi" in fn_lower:
            return "multispectral"

        # 5. Autonomous pixel inspection for 3-band imagery: grayscale encoded as RGB
        #    is a common SAR radar product; strong chroma indicates natural color.
        if data is not None and bands == 3:
            ch_diff_rg = np.mean(np.abs(data[:, :, 0].astype(np.float32) - data[:, :, 1].astype(np.float32)))
            ch_diff_gb = np.mean(np.abs(data[:, :, 1].astype(np.float32) - data[:, :, 2].astype(np.float32)))
            is_monochrome = (ch_diff_rg < 3.0 and ch_diff_gb < 3.0)
            if is_monochrome:
                mean_val = float(np.mean(data))
                std_val = float(np.std(data))
                cv = std_val / (mean_val + 1e-6)
                if cv > 0.35:
                    return "sar"
            else:
                return "multispectral" if "multi" in fn_lower else "optical"

        return "optical"

    def _open_raster(self, file_path: Path) -> np.ndarray:
        """Decode a raster with PIL, rejecting empty/oversized/corrupt images.

        Dimensions are checked from the lazy header before ``np.array`` materializes
        the full buffer, so a decompression bomb is rejected without an allocation.
        """
        with Image.open(file_path) as pil_img:
            width, height = pil_img.size
            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions are invalid or corrupted.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image exceeds the maximum supported pixel count ({MAX_IMAGE_PIXELS})."
                )
            return np.array(pil_img)

    def read_image_data(self, file_path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
        suffix = file_path.suffix.lower()
        tags: Dict[str, Any] = {}

        try:
            if suffix in [".tif", ".tiff"]:
                try:
                    with tifffile.TiffFile(str(file_path)) as tif:
                        page = tif.pages[0]
                        pixel_count = 1
                        for dim in page.shape:
                            pixel_count *= int(dim)
                        if pixel_count > MAX_IMAGE_PIXELS:
                            raise ValueError(
                                f"Image exceeds the maximum supported pixel count ({MAX_IMAGE_PIXELS})."
                            )
                        data = tif.asarray()
                        if tif.geotiff_metadata:
                            tags["geotiff"] = tif.geotiff_metadata
                            meta = tif.geotiff_metadata
                            scale = meta.get("ModelPixelScale")
                            tie = meta.get("ModelTiepoint")
                            if tie and scale and len(tie) >= 5 and len(scale) >= 2:
                                x0, y0 = float(tie[3]), float(tie[4])
                                sx, sy = float(scale[0]), float(scale[1])
                                h_tif, w_tif = data.shape[:2]
                                x1 = x0 + w_tif * sx
                                y1 = y0 - h_tif * sy

                                proj_cs = meta.get("ProjectedCSTypeGeoKey")
                                if proj_cs == 3857 or "pseudo-mercator" in str(meta).lower():
                                    def _m2w(mx: float, my: float):
                                        lon = (mx / 20037508.342789244) * 180.0
                                        lat = 180.0 / math.pi * (2.0 * math.atan(math.exp((my / 20037508.342789244) * math.pi)) - math.pi / 2.0)
                                        return round(lon, 6), round(lat, 6)
                                    min_lon, min_lat = _m2w(x0, y1)
                                    max_lon, max_lat = _m2w(x1, y0)
                                    tags["bounds"] = [min_lon, min_lat, max_lon, max_lat]
                                    tags["crs"] = "EPSG:4326"
                                elif proj_cs == 4326 or "wgs 84" in str(meta).lower():
                                    tags["bounds"] = [round(x0, 6), round(y1, 6), round(x1, 6), round(y0, 6)]
                                    tags["crs"] = "EPSG:4326"
                except Exception:
                    # Fall back to PIL for non-GeoTIFF TIFFs, partial files, or
                    # compression bombs that PIL can still identify and reject cleanly.
                    data = self._open_raster(file_path)
            else:
                data = self._open_raster(file_path)
        except Exception as exc:
            raise _corrupt_image(
                f"Could not decode image '{file_path.name}': {type(exc).__name__}: {exc}"
            ) from exc

        if data is None or data.size == 0:
            raise _corrupt_image(f"Image '{file_path.name}' contains no pixel data.")

        # Standardize shape to (H, W, C)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        elif data.ndim == 3 and data.shape[0] in [1, 2, 3, 4, 8, 12, 13] and data.shape[2] > 16:
            # Channels first: (C, H, W) -> (H, W, C)
            data = np.transpose(data, (1, 2, 0))

        return data, tags

    def create_thumbnail_base64(self, data: np.ndarray, max_size: int = 256) -> str:
        try:
            # Handle multispectral or single channel
            if data.ndim == 3:
                if data.shape[2] >= 3:
                    rgb = data[:, :, :3].astype(np.float32)
                elif data.shape[2] == 2:
                    # SAR VV/VH composite: VV, VH, VV/VH
                    c1 = data[:, :, 0].astype(np.float32)
                    c2 = data[:, :, 1].astype(np.float32)
                    c3 = np.clip(c1 / (c2 + 1e-6), 0, 255)
                    rgb = np.stack([c1, c2, c3], axis=-1)
                else:
                    c1 = data[:, :, 0].astype(np.float32)
                    rgb = np.stack([c1, c1, c1], axis=-1)
            else:
                rgb = np.stack([data, data, data], axis=-1).astype(np.float32)

            # Min-max normalization for display
            min_val = np.percentile(rgb, 1)
            max_val = np.percentile(rgb, 99)
            if max_val > min_val:
                rgb_norm = np.clip((rgb - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_norm = np.zeros_like(rgb, dtype=np.uint8)

            img = Image.fromarray(rgb_norm)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return ""

    @staticmethod
    def generic_sensor_label(modality: str) -> Optional[str]:
        """Modality-honest platform label used only when the file names no sensor."""
        return {
            "optical": "Optical (RGB)",
            "multispectral": "Multispectral",
            "sar": "SAR (radar)",
        }.get(modality)

    def ingest_image(
        self,
        file_bytes: bytes,
        filename: str,
        dest_path: Path,
        forced_modality: Optional[str] = None
    ) -> ImageMetadataEnvelope:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        sha256_hash = self.compute_sha256(file_bytes)
        data, tags = self.read_image_data(dest_path)

        if data.ndim < 2 or data.shape[0] <= 0 or data.shape[1] <= 0 or data.size == 0:
            raise _corrupt_image(f"Image '{filename}' is empty or has invalid dimensions.")

        height, width = data.shape[0], data.shape[1]
        bands = data.shape[2] if data.ndim == 3 else 1
        dtype_str = str(data.dtype)

        modality = forced_modality or self.detect_modality(filename, data.shape, tags, data=data)
        thumbnail = self.create_thumbnail_base64(data)

        image_id = f"img_{sha256_hash[:12]}"

        # Carry only georeferencing the file actually provides. Never invent an
        # AOI, a CRS, a resolution, or a named satellite platform for plain imagery.
        # Only four genuinely finite bounds are ever carried through; a NaN/Inf bound
        # would otherwise poison geo_box strings and GeoJSON serialization.
        real_bounds = _finite_bounds(tags.get("bounds"))

        envelope = ImageMetadataEnvelope(
            image_id=image_id,
            filename=filename,
            filepath=str(dest_path),
            width=width,
            height=height,
            bands=bands,
            dtype=dtype_str,
            file_size_bytes=len(file_bytes),
            sha256=sha256_hash,
            modality=modality,
            sensor=tags.get("sensor") or self.generic_sensor_label(modality),
            crs=tags.get("crs"),
            resolution_m=tags.get("resolution"),
            bounds=real_bounds,
            geo_bbox=real_bounds,
            nodata_val=None,
            thumbnail_base64=thumbnail,
            tags=tags
        )
        return envelope


def pixel_box_to_geo(box: List[int], envelope: ImageMetadataEnvelope) -> Optional[Dict[str, Any]]:
    """
    Transforms pixel bounding box [x1, y1, x2, y2] into real-world geographic coordinates.

    Returns ``None`` when the source image carries no genuine (finite) georeferencing
    or when the box itself is malformed/non-finite; no default or inferred AOI is ever
    substituted.
    """
    bounds = _finite_bounds(envelope.bounds)
    if bounds is None:
        return None
    min_lon, min_lat, max_lon, max_lat = bounds

    box_values = _finite_box(box)
    if box_values is None:
        return None

    w = max(1, envelope.width)
    h = max(1, envelope.height)

    x1, y1, x2, y2 = box_values
    fx1 = max(0.0, min(1.0, x1 / w))
    fx2 = max(0.0, min(1.0, x2 / w))
    fy1 = max(0.0, min(1.0, y1 / h))
    fy2 = max(0.0, min(1.0, y2 / h))

    b_min_lon = round(min_lon + fx1 * (max_lon - min_lon), 5)
    b_max_lon = round(min_lon + fx2 * (max_lon - min_lon), 5)
    b_max_lat = round(max_lat - fy1 * (max_lat - min_lat), 5)
    b_min_lat = round(max_lat - fy2 * (max_lat - min_lat), 5)

    lat1_str = f"{abs(b_min_lat):.4f}\u00b0{'N' if b_min_lat >= 0 else 'S'}"
    lon1_str = f"{abs(b_min_lon):.4f}\u00b0{'E' if b_min_lon >= 0 else 'W'}"
    lat2_str = f"{abs(b_max_lat):.4f}\u00b0{'N' if b_max_lat >= 0 else 'S'}"
    lon2_str = f"{abs(b_max_lon):.4f}\u00b0{'E' if b_max_lon >= 0 else 'W'}"

    return {
        "pixel_box": list(box),
        "geo_box": [b_min_lon, b_min_lat, b_max_lon, b_max_lat],
        "formatted_coords": f"[{lat1_str}, {lon1_str}] to [{lat2_str}, {lon2_str}]",
        "crs": envelope.crs
    }


def boxes_to_geojson(
    boxes: List[List[int]], 
    envelope: ImageMetadataEnvelope, 
    label: str = "Grounding Target",
    evidence_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generates standard GeoJSON FeatureCollection for direct visualization in GIS tools (QGIS, ArcGIS)."""
    CATEGORY_COLORS = {
        "water_body": "#0284c7",          # Ocean/Lake Blue
        "forest_canopy": "#15803d",        # Deep Forest Green
        "vegetation_permeable": "#84cc16", # Agricultural / Crop Lime Green
        "built_up": "#f59e0b",             # Urban Amber / Orange
    }

    if _finite_bounds(envelope.bounds) is None:
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": [],
            "note": "No georeferencing available; no geographic features were generated."
        }

    # Evidence carrying a real region is matched to boxes by region equality. Generic
    # evidence without regions (e.g. per-image provenance) may still align by index.
    region_indexed = any(
        _finite_box(ev.get("region")) is not None for ev in (evidence_items or [])
    )

    def _match_evidence(box_values, idx):
        if not evidence_items:
            return None
        if region_indexed:
            for ev in evidence_items:
                if _finite_box(ev.get("region")) == box_values:
                    return ev
            return None
        if idx < len(evidence_items):
            return evidence_items[idx]
        return None

    features = []
    for idx, box in enumerate(boxes):
        box_values = _finite_box(box)
        if box_values is None:
            # Malformed / non-finite box: skip it rather than emit invalid geometry.
            continue
        geo_info = pixel_box_to_geo(list(box), envelope)
        if geo_info is None:
            continue
        min_lon, min_lat, max_lon, max_lat = geo_info["geo_box"]
        coordinates = [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]

        ev_item = _match_evidence(box_values, idx)
        if ev_item is None and region_indexed:
            # Region-keyed evidence exists but none matches this box: skip it instead
            # of applying another box's category/color by position.
            continue
        ev_item = ev_item or {}
        item_cat = ev_item.get("type") or ev_item.get("category") or "general_target"
        item_label = ev_item.get("label") or label
        item_desc = ev_item.get("description") or ""
        item_score = ev_item.get("score") or 0.90
        item_color = CATEGORY_COLORS.get(item_cat) or CATEGORY_COLORS.get(item_cat.lower(), "#06b6d4")

        features.append({
            "type": "Feature",
            "properties": {
                "id": f"zone_{idx + 1}",
                "label": item_label,
                "category": item_cat,
                "color": item_color,
                "description": item_desc,
                "confidence": item_score,
                "formatted_coords": geo_info["formatted_coords"],
                "pixel_box": list(box),
                "sensor": envelope.sensor,
                "modality": envelope.modality
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        })

    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
