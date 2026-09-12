from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image
from app.config import load_sensor_profiles


class PreprocessingService:
    """Prepares image tensors, applies sensor scaling, and handles spatial coordinate transforms."""

    def __init__(self):
        self.sensor_profiles = load_sensor_profiles()

    def process_sar_to_db(self, sar_data: np.ndarray) -> np.ndarray:
        """Converts SAR linear amplitude/intensity to calibrated decibel (dB) scale."""
        data = np.maximum(sar_data.astype(np.float32), 1e-6)
        db = 10.0 * np.log10(data)
        # Clip reasonable dB range [-35, +10]
        db_clipped = np.clip(db, -35.0, 10.0)
        return db_clipped

    def normalize_array(self, arr: np.ndarray, modality: str = "optical") -> np.ndarray:
        """Normalizes array to standard [0.0, 1.0] range."""
        if modality == "sar":
            # Scale from [-35, 10] dB to [0, 1]
            return (arr + 35.0) / 45.0
        else:
            p2 = np.percentile(arr, 2)
            p98 = np.percentile(arr, 98)
            if p98 > p2:
                return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
            return np.zeros_like(arr, dtype=np.float32)

    def prepare_tensor(
        self,
        raw_data: np.ndarray,
        modality: str,
        target_size: int = 512
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Converts raw multi-band array into standardized [3, target_size, target_size] float32 tensor.
        Returns tensor and coordinate mapping metadata for bounding box re-projection.
        """
        orig_h, orig_w = raw_data.shape[0], raw_data.shape[1]

        # Extract 3-channel composite
        if raw_data.ndim == 2:
            comp = np.stack([raw_data, raw_data, raw_data], axis=-1)
        elif raw_data.shape[2] == 1:
            comp = np.repeat(raw_data, 3, axis=-1)
        elif raw_data.shape[2] == 2:
            # Dual-pol SAR: VV, VH, ratio
            c1 = raw_data[:, :, 0]
            c2 = raw_data[:, :, 1]
            c3 = c1 / (c2 + 1e-6)
            comp = np.stack([c1, c2, c3], axis=-1)
        elif raw_data.shape[2] >= 3:
            comp = raw_data[:, :, :3]
        else:
            comp = raw_data

        if modality == "sar":
            comp = self.process_sar_to_db(comp)

        norm_comp = self.normalize_array(comp, modality=modality)

        # Convert to uint8 for high-quality PIL resizing
        uint8_img = (norm_comp * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(uint8_img)
        resized_pil = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

        # Standard float32 tensor (3, H, W)
        resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0
        tensor = np.transpose(resized_arr, (2, 0, 1))

        transform_meta = {
            "orig_width": orig_w,
            "orig_height": orig_h,
            "target_size": target_size,
            "scale_x": orig_w / float(target_size),
            "scale_y": orig_h / float(target_size)
        }

        return tensor, transform_meta

    def project_boxes_to_original(
        self,
        boxes_norm: List[List[float]],
        transform_meta: Dict[str, Any],
        format_range: float = 1000.0
    ) -> List[List[int]]:
        """
        Projects normalized bounding boxes [ymin, xmin, ymax, xmax] (e.g. 0-1000 from EarthDial)
        back to pixel coordinates [x1, y1, x2, y2] on the original full-resolution image.
        """
        orig_w = transform_meta["orig_width"]
        orig_h = transform_meta["orig_height"]

        projected = []
        for box in boxes_norm:
            if len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box

            # Convert from format_range (e.g. 1000) to unit [0, 1]
            ymin_u = max(0.0, min(1.0, ymin / format_range))
            xmin_u = max(0.0, min(1.0, xmin / format_range))
            ymax_u = max(0.0, min(1.0, ymax / format_range))
            xmax_u = max(0.0, min(1.0, xmax / format_range))

            x1 = int(round(xmin_u * orig_w))
            y1 = int(round(ymin_u * orig_h))
            x2 = int(round(xmax_u * orig_w))
            y2 = int(round(ymax_u * orig_h))

            projected.append([x1, y1, x2, y2])

        return projected

    def get_dofa_wavelengths(self, modality: str, sensor: Optional[str] = None) -> List[float]:
        """Returns standard center wavelengths in micrometers (µm) for DOFA conditioning."""
        if modality == "sar":
            # C-band radar: ~0.056 µm equivalent frequency scale
            return [0.056, 0.056]
        elif modality == "multispectral":
            return [0.490, 0.560, 0.665, 0.842]  # B2, B3, B4, B8
        else:
            # Optical RGB: Red, Green, Blue
            return [0.665, 0.560, 0.490]

    def compute_spectral_indices(self, raw_data: np.ndarray, modality: str = "multispectral") -> Dict[str, Any]:
        """
        Computes NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index).
        Generates statistical summaries, vegetation vigor ratings, and color-mapped visual raster overlays.
        """
        import io
        import base64

        data = raw_data.astype(np.float32)
        h, w = data.shape[0], data.shape[1]

        # Extract NIR, Red, Green channels based on data shape and modality
        if data.ndim == 3 and data.shape[2] >= 4:
            # Sentinel-2 MSI standard channel ordering: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
            blue = data[:, :, 0]
            green = data[:, :, 1]
            red = data[:, :, 2]
            nir = data[:, :, 3]
        elif data.ndim == 3 and data.shape[2] >= 3:
            # Optical RGB: simulate Green-Red / VARI indices for proxy vegetation analysis
            red = data[:, :, 0]
            green = data[:, :, 1]
            blue = data[:, :, 2]
            nir = green * 1.35  # proxy NIR from high vegetative reflectance
        elif data.ndim == 2:
            red = data
            green = data
            nir = data * 1.1
            blue = data
        else:
            red = data[:, :, 0]
            green = data[:, :, 0]
            nir = data[:, :, 0]
            blue = data[:, :, 0]

        # 1. NDVI Calculation: (NIR - Red) / (NIR + Red)
        denom_ndvi = nir + red + 1e-6
        ndvi = np.clip((nir - red) / denom_ndvi, -1.0, 1.0)

        # 2. NDWI Calculation: (Green - NIR) / (Green + NIR)
        denom_ndwi = green + nir + 1e-6
        ndwi = np.clip((green - nir) / denom_ndwi, -1.0, 1.0)

        # Statistical Metrics
        mean_ndvi = float(np.mean(ndvi))
        mean_ndwi = float(np.mean(ndwi))
        dense_veg_pct = float(np.mean(ndvi > 0.45) * 100.0)
        moderate_veg_pct = float(np.mean((ndvi >= 0.20) & (ndvi <= 0.45)) * 100.0)
        barren_pct = float(np.mean((ndvi >= 0.0) & (ndvi < 0.20)) * 100.0)
        water_body_pct = float(np.mean(ndwi > 0.15) * 100.0)

        # Colormap generation for NDVI: RdYlGn (Brown/Red -> Yellow -> Lush Green)
        ndvi_norm = np.clip((ndvi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        lut_ndvi = np.zeros((256, 3), dtype=np.uint8)
        lut_ndvi[:85, 0] = 210
        lut_ndvi[:85, 1] = np.linspace(40, 180, 85).astype(np.uint8)
        lut_ndvi[:85, 2] = 30
        lut_ndvi[85:170, 0] = np.linspace(210, 40, 85).astype(np.uint8)
        lut_ndvi[85:170, 1] = 200
        lut_ndvi[85:170, 2] = 40
        lut_ndvi[170:, 0] = 16
        lut_ndvi[170:, 1] = np.linspace(160, 245, 86).astype(np.uint8)
        lut_ndvi[170:, 2] = 50

        img_ndvi = Image.fromarray(lut_ndvi[ndvi_norm])
        img_ndvi.thumbnail((384, 384), Image.Resampling.LANCZOS)
        buf_ndvi = io.BytesIO()
        img_ndvi.save(buf_ndvi, format="PNG")
        ndvi_b64 = f"data:image/png;base64,{base64.b64encode(buf_ndvi.getvalue()).decode('utf-8')}"

        # Colormap generation for NDWI: YlGnBu / Cyan-Blue palette
        ndwi_norm = np.clip((ndwi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        lut_ndwi = np.zeros((256, 3), dtype=np.uint8)
        lut_ndwi[:110, 0] = np.linspace(190, 80, 110).astype(np.uint8)
        lut_ndwi[:110, 1] = np.linspace(160, 70, 110).astype(np.uint8)
        lut_ndwi[:110, 2] = 50
        lut_ndwi[110:180, 0] = 40
        lut_ndwi[110:180, 1] = np.linspace(160, 210, 70).astype(np.uint8)
        lut_ndwi[110:180, 2] = 220
        lut_ndwi[180:, 0] = np.linspace(20, 10, 76).astype(np.uint8)
        lut_ndwi[180:, 1] = np.linspace(80, 40, 76).astype(np.uint8)
        lut_ndwi[180:, 2] = np.linspace(220, 255, 76).astype(np.uint8)

        img_ndwi = Image.fromarray(lut_ndwi[ndwi_norm])
        img_ndwi.thumbnail((384, 384), Image.Resampling.LANCZOS)
        buf_ndwi = io.BytesIO()
        img_ndwi.save(buf_ndwi, format="PNG")
        ndwi_b64 = f"data:image/png;base64,{base64.b64encode(buf_ndwi.getvalue()).decode('utf-8')}"

        return {
            "mean_ndvi": round(mean_ndvi, 3),
            "mean_ndwi": round(mean_ndwi, 3),
            "dense_vegetation_pct": round(dense_veg_pct, 1),
            "moderate_vegetation_pct": round(moderate_veg_pct, 1),
            "barren_soil_pct": round(barren_pct, 1),
            "water_body_pct": round(water_body_pct, 1),
            "vegetation_vigor": "High Active Canopy" if mean_ndvi > 0.45 else ("Moderate Vegetative Growth" if mean_ndvi > 0.25 else "Sparse / Non-Vegetated"),
            "ndvi_overlay_b64": ndvi_b64,
            "ndwi_overlay_b64": ndwi_b64
        }


def compute_spectral_indices(raw_data: np.ndarray, modality: str = "multispectral") -> Dict[str, Any]:
    """Convenience module-level function to compute NDVI/NDWI spectral indices."""
    preprocessor = PreprocessingService()
    return preprocessor.compute_spectral_indices(raw_data, modality=modality)

