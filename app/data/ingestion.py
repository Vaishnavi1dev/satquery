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
        fn_lower = filename.lower()
        bands = shape[2] if len(shape) == 3 else 1

        # 1. Check explicit filename markers for Optical / Multispectral first
        if any(marker in fn_lower for marker in ["multispectral", "msi", "sentinel-2", "sentinel2", "ben-ge", "landsat", "b04_b08", "b08", "ndvi", "ndwi", "water", "urban", "optical", "rgb"]):
            return "multispectral" if bands > 3 or "multi" in fn_lower else "optical"

        # 2. Check explicit filename markers for SAR / Radar
        sar_tokens = ["sentinel-1", "sentinel1", "risat", "_sar", "sar_", "-sar", "c-band", "cband", "radar"]
        if any(marker in fn_lower for marker in sar_tokens) or fn_lower.startswith("sar") or re.search(r"\bsar\b", fn_lower):
            return "sar"

        # 3. Check tag metadata for SAR polarization (do NOT match substring 'sar' inside 'isarea'!)
        tag_str = str(tags).lower()
        if any(p in tag_str for p in ["polarisation", "polarization", "c-band", "backscatter", "sigma0", "gamma0", "sentinel-1"]) or re.search(r"\bsar\b", tag_str):
            return "sar"

        # 4. Check band counts
        if bands > 3:
            return "multispectral"
        elif bands == 2:
            # Dual-polarization radar (e.g. VV + VH)
            return "sar"
        elif bands == 1:
            # Single band optical panchromatic or SAR
            if "pan" in fn_lower or "cartosat" in fn_lower or "opt" in fn_lower:
                return "optical"
            return "sar"

        # 5. Autonomous Pixel-Data Inspection (for 3-band / RGB encoded imagery)
        if data is not None and bands == 3:
            # Check if all 3 color channels are identical (grayscale encoded as RGB, standard in SAR radar products)
            ch_diff_rg = np.mean(np.abs(data[:, :, 0].astype(np.float32) - data[:, :, 1].astype(np.float32)))
            ch_diff_gb = np.mean(np.abs(data[:, :, 1].astype(np.float32) - data[:, :, 2].astype(np.float32)))
            is_monochrome = (ch_diff_rg < 3.0 and ch_diff_gb < 3.0)

            if is_monochrome:
                # In remote sensing, monochrome images are either panchromatic optical or SAR radar backscatter.
                mean_val = float(np.mean(data))
                std_val = float(np.std(data))
                cv = std_val / (mean_val + 1e-6)
                if cv > 0.35:
                    return "sar"
            else:
                # Significant color variance across RGB -> Natural/false-color optical
                return "multispectral" if "multi" in fn_lower else "optical"

        return "optical"

    def read_image_data(self, file_path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
        suffix = file_path.suffix.lower()
        tags: Dict[str, Any] = {}

        if suffix in [".tif", ".tiff"]:
            try:
                with tifffile.TiffFile(str(file_path)) as tif:
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
            except Exception as e:
                # Fallback to PIL
                pil_img = Image.open(file_path)
                data = np.array(pil_img)
        else:
            pil_img = Image.open(file_path)
            data = np.array(pil_img)

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

        height, width = data.shape[0], data.shape[1]
        bands = data.shape[2] if data.ndim == 3 else 1
        dtype_str = str(data.dtype)

        modality = forced_modality or self.detect_modality(filename, data.shape, tags, data=data)
        thumbnail = self.create_thumbnail_base64(data)

        image_id = f"img_{sha256_hash[:12]}"

        # Carry only georeferencing the file actually provides. Never invent an
        # AOI, a CRS, a resolution, or a named satellite platform for plain imagery.
        raw_bounds = tags.get("bounds")
        real_bounds = raw_bounds if isinstance(raw_bounds, list) and len(raw_bounds) == 4 else None

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

    Returns ``None`` when the source image carries no genuine georeferencing; no
    default or inferred AOI is ever substituted.
    """
    bounds = envelope.bounds
    if not bounds or len(bounds) != 4:
        return None
    min_lon, min_lat, max_lon, max_lat = bounds

    w = max(1, envelope.width)
    h = max(1, envelope.height)

    x1, y1, x2, y2 = box
    fx1 = max(0.0, min(1.0, x1 / w))
    fx2 = max(0.0, min(1.0, x2 / w))
    fy1 = max(0.0, min(1.0, y1 / h))
    fy2 = max(0.0, min(1.0, y2 / h))

    b_min_lon = round(min_lon + fx1 * (max_lon - min_lon), 5)
    b_max_lon = round(min_lon + fx2 * (max_lon - min_lon), 5)
    b_max_lat = round(max_lat - fy1 * (max_lat - min_lat), 5)
    b_min_lat = round(max_lat - fy2 * (max_lat - min_lat), 5)

    lat1_str = f"{abs(b_min_lat):.4f}°{'N' if b_min_lat >= 0 else 'S'}"
    lon1_str = f"{abs(b_min_lon):.4f}°{'E' if b_min_lon >= 0 else 'W'}"
    lat2_str = f"{abs(b_max_lat):.4f}°{'N' if b_max_lat >= 0 else 'S'}"
    lon2_str = f"{abs(b_max_lon):.4f}°{'E' if b_max_lon >= 0 else 'W'}"

    return {
        "pixel_box": [x1, y1, x2, y2],
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

    if not envelope.bounds or len(envelope.bounds) != 4:
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": [],
            "note": "No georeferencing available; no geographic features were generated."
        }

    features = []
    for idx, box in enumerate(boxes):
        geo_info = pixel_box_to_geo(box, envelope)
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

        ev_item = evidence_items[idx] if (evidence_items and idx < len(evidence_items)) else {}
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
                "pixel_box": box,
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

