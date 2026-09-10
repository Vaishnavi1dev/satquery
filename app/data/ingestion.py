import base64
import hashlib
import io
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
    nodata_val: Optional[float] = None
    thumbnail_base64: Optional[str] = None
    tags: Dict[str, Any] = Field(default_factory=dict)


class ImageIngestionService:
    """Reads GeoTIFF, TIFF, PNG, and JPEG imagery, extracts metadata, and detects modality."""

    def __init__(self):
        pass

    def compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def detect_modality(self, filename: str, shape: tuple, tags: Dict[str, Any]) -> str:
        fn_lower = filename.lower()
        bands = shape[2] if len(shape) == 3 else 1

        # Check explicit filename markers for SAR
        if any(marker in fn_lower for marker in ["sar", "s1", "sentinel-1", "sentinel1", "risat", "vv", "vh", "hh", "hv"]):
            return "sar"

        # Check tag metadata for SAR polarization
        tag_str = str(tags).lower()
        if any(p in tag_str for p in ["polarisation", "c-band", "backscatter", "sigma0", "gamma0"]):
            return "sar"

        # Check explicit filename markers for Multispectral
        if any(marker in fn_lower for marker in ["multispectral", "msi", "sentinel-2", "sentinel2", "ben-ge", "landsat", "b04_b08", "b08", "ndvi"]):
            return "multispectral"

        # Check band counts
        if bands in (1, 2) and any(kw in fn_lower for kw in ["radar", "amplitude", "intensity"]):
            return "sar"
        elif bands > 3:
            return "multispectral"
        elif bands == 3:
            return "optical"
        elif bands == 1:
            # Single band optical panchromatic or SAR
            if "pan" in fn_lower or "cartosat" in fn_lower:
                return "optical"
            return "sar"

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

        modality = forced_modality or self.detect_modality(filename, data.shape, tags)
        thumbnail = self.create_thumbnail_base64(data)

        image_id = f"img_{sha256_hash[:12]}"

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
            sensor=tags.get("sensor"),
            crs=tags.get("crs", "EPSG:4326" if "geotiff" in tags else None),
            resolution_m=tags.get("resolution", 10.0 if "sentinel" in filename.lower() else None),
            bounds=tags.get("bounds"),
            nodata_val=None,
            thumbnail_base64=thumbnail,
            tags=tags
        )
        return envelope
