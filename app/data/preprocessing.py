import math
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image
from app.config import load_sensor_profiles


class PreprocessingService:
    """Prepares image tensors, applies sensor scaling, and handles spatial coordinate transforms."""

    def __init__(self):
        self.sensor_profiles = load_sensor_profiles()

    def process_sar_to_db(self, sar_data: np.ndarray) -> np.ndarray:
        """Converts SAR linear amplitude/intensity to calibrated decibel (dB) scale.

        Integer DN (e.g. uint8) is first scaled by the dtype maximum so it is treated
        as linear amplitude in [0, 1]; otherwise raw counts saturate the +10 dB clip.
        """
        arr = sar_data.astype(np.float32)
        if np.issubdtype(sar_data.dtype, np.integer):
            info = np.iinfo(sar_data.dtype)
            if info.max > 0:
                arr = arr / float(info.max)
        data = np.maximum(arr, 1e-6)
        db = 10.0 * np.log10(data)
        # Clip reasonable dB range [-35, +10]
        db_clipped = np.clip(db, -35.0, 10.0)
        return db_clipped

    def normalize_array(self, arr: np.ndarray, modality: str = "optical") -> np.ndarray:
        """Normalizes array to standard [0.0, 1.0] range."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            # ``np.percentile`` on an empty array raises; an empty scene normalizes to 0.
            return np.zeros_like(arr, dtype=np.float32)
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

        try:
            fr = float(format_range)
        except (TypeError, ValueError):
            fr = 1000.0
        if not math.isfinite(fr) or fr == 0.0:
            # Guard the division: a zero/NaN format_range must not raise ZeroDivisionError.
            fr = 1.0

        projected = []
        for box in boxes_norm:
            if len(box) != 4:
                continue
            try:
                coords = [float(v) for v in box]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in coords):
                continue
            ymin, xmin, ymax, xmax = coords

            # Convert from format_range (e.g. 1000) to unit [0, 1]
            ymin_u = max(0.0, min(1.0, ymin / fr))
            xmin_u = max(0.0, min(1.0, xmin / fr))
            ymax_u = max(0.0, min(1.0, ymax / fr))
            xmax_u = max(0.0, min(1.0, xmax / fr))

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

    @staticmethod
    def _spectral_indices_error(code: str, message: str) -> Dict[str, Any]:
        return {
            "error": code,
            "message": message,
            "applicable": False,
            "mean_ndvi": None,
            "mean_ndwi": None,
            "dense_vegetation_pct": None,
            "moderate_vegetation_pct": None,
            "barren_soil_pct": None,
            "water_body_pct": None,
            "vegetation_vigor": "Not applicable",
            "ndvi_overlay_b64": None,
            "ndwi_overlay_b64": None,
        }

    def compute_spectral_indices(self, raw_data: np.ndarray, modality: str = "multispectral") -> Dict[str, Any]:
        """
        Computes NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index).
        Generates statistical summaries, vegetation vigor ratings, and color-mapped visual raster overlays.

        NDVI/NDWI require visible + NIR bands: SAR and <3-band inputs return a clean
        ``applicable: False`` result instead of a misleading vegetation/water index.
        """
        import io
        import base64

        data = np.asarray(raw_data, dtype=np.float32)

        if modality == "sar":
            return self._spectral_indices_error(
                "SPECTRAL_INDICES_NOT_APPLICABLE",
                "NDVI/NDWI are not defined for SAR (radar) imagery.",
            )
        if data.ndim != 3 or data.shape[2] < 3 or data.shape[0] == 0 or data.shape[1] == 0:
            return self._spectral_indices_error(
                "SPECTRAL_INDICES_INVALID_INPUT",
                "Spectral indices require a non-empty HxWxC array with at least 3 bands.",
            )

        # Mask non-finite samples so NaN/Inf cannot poison any index or statistic.
        data = np.where(np.isfinite(data), data, 0.0)

        h, w = data.shape[0], data.shape[1]

        # Extract NIR, Red, Green channels based on data shape and modality
        if data.shape[2] >= 4:
            # Sentinel-2 MSI standard channel ordering: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
            blue = data[:, :, 0]
            green = data[:, :, 1]
            red = data[:, :, 2]
            nir = data[:, :, 3]
        else:
            # Optical RGB: simulate biophysical NIR proxy for vegetation vs water analysis
            red = data[:, :, 0].astype(np.float32)
            green = data[:, :, 1].astype(np.float32)
            blue = data[:, :, 2].astype(np.float32)

            max_val = float(np.max(data))
            r_scaled = red if max_val > 1.0 else red * 255.0
            g_scaled = green if max_val > 1.0 else green * 255.0
            b_scaled = blue if max_val > 1.0 else blue * 255.0
            lum = 0.299 * r_scaled + 0.587 * g_scaled + 0.114 * b_scaled

            # 1. Vegetation Chlorophyll metrics:
            # Excess Green index (2G - R - B) and Normalized Green-Red Difference Index (G - R)/(G + R)
            exg = 2.0 * g_scaled - r_scaled - b_scaled
            ngrdi = (g_scaled - r_scaled) / (g_scaled + r_scaled + 1e-6)

            # 2. Genuine Open Water criteria:
            # Water absorbs red light strongly (r < 35) with blue dominance (b > r*1.35 and b >= g*0.95),
            # low luminance, and strictly non-vegetative signature (exg <= 0).
            is_water = (
                (b_scaled > r_scaled * 1.35) &
                (b_scaled >= g_scaled * 0.95) &
                (r_scaled < 35.0) &
                (exg <= 0.0) &
                (lum < 70.0)
            )

            # 3. Continuous biophysical NIR estimation:
            # - Terrestrial base (soil, urban, asphalt, barren): NIR reflects higher than both red and green.
            #   This guarantees that on all dry land, (green - nir) is negative (NDWI < 0.0).
            base_nir = 1.15 * red + 0.20 * green

            # - Vegetative boost: leaf cellular structure scatters NIR strongly when chlorophyll is active.
            veg_strength = np.clip(ngrdi * 2.5, 0.0, 1.5)
            veg_nir = green * (1.30 + veg_strength) + red * 0.10

            nir_land = np.where((ngrdi > 0.02) & (exg > 0.0), veg_nir, base_nir)

            # - Water absorption: water absorbs NIR almost entirely.
            nir = np.where(is_water, green * 0.15, nir_land)

        # 1. NDVI Calculation: (NIR - Red) / (NIR + Red)
        denom_ndvi = nir + red + 1e-6
        ndvi = np.clip((nir - red) / denom_ndvi, -1.0, 1.0)

        # 2. NDWI Calculation: (Green - NIR) / (Green + NIR)
        denom_ndwi = green + nir + 1e-6
        ndwi = np.clip((green - nir) / denom_ndwi, -1.0, 1.0)

        # Statistical Metrics (clamped so a degenerate array can never emit NaN/Inf)
        def _safe_stat(value, default=0.0):
            v = float(value)
            return v if math.isfinite(v) else default

        mean_ndvi = _safe_stat(np.mean(ndvi))
        mean_ndwi = _safe_stat(np.mean(ndwi))
        dense_veg_pct = _safe_stat(np.mean(ndvi > 0.45) * 100.0)
        moderate_veg_pct = _safe_stat(np.mean((ndvi >= 0.20) & (ndvi <= 0.45)) * 100.0)
        barren_pct = _safe_stat(np.mean((ndvi >= 0.0) & (ndvi < 0.20)) * 100.0)
        water_body_pct = _safe_stat(np.mean(ndwi > 0.15) * 100.0)

        # Colormap generation for NDVI: RdYlGn (Brown/Red -> Yellow -> Lush Green)
        # Normalized range: 0 -> -1.0, 128 -> 0.0, 140 -> 0.10, 178 -> 0.40, 255 -> 1.0
        ndvi_norm = np.clip((ndvi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        lut_ndvi = np.zeros((256, 3), dtype=np.uint8)
        # < 0.10 (0..139): Barren / urban / soil (Terracotta Red)
        lut_ndvi[:140, 0] = np.linspace(200, 220, 140).astype(np.uint8)
        lut_ndvi[:140, 1] = np.linspace(40, 140, 140).astype(np.uint8)
        lut_ndvi[:140, 2] = 30
        # 0.10 .. 0.40 (140..178): Grassland / moderate vegetation (Golden Yellow / Lime)
        lut_ndvi[140:179, 0] = np.linspace(220, 120, 39).astype(np.uint8)
        lut_ndvi[140:179, 1] = np.linspace(180, 215, 39).astype(np.uint8)
        lut_ndvi[140:179, 2] = np.linspace(35, 45, 39).astype(np.uint8)
        # > 0.40 (179..255): Dense canopy / forest (Lush Forest Green)
        lut_ndvi[179:, 0] = np.linspace(60, 16, 77).astype(np.uint8)
        lut_ndvi[179:, 1] = np.linspace(190, 245, 77).astype(np.uint8)
        lut_ndvi[179:, 2] = np.linspace(50, 45, 77).astype(np.uint8)

        img_ndvi = Image.fromarray(lut_ndvi[ndvi_norm])
        img_ndvi.thumbnail((384, 384), Image.Resampling.LANCZOS)
        buf_ndvi = io.BytesIO()
        img_ndvi.save(buf_ndvi, format="PNG")
        ndvi_b64 = f"data:image/png;base64,{base64.b64encode(buf_ndvi.getvalue()).decode('utf-8')}"

        # Colormap generation for NDWI: Calibrated strictly to physical water absorption
        # 0..127 (NDWI < 0.0): Dry Land (Warm Earthy Soil / Ochre)
        # 128..153 (0.0 <= NDWI < 0.20): Moist / Transition zone (Soft Cyan-Teal)
        # 154..255 (NDWI >= 0.20): Open Water (Deep Ocean / Electric Blue)
        ndwi_norm = np.clip((ndwi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        lut_ndwi = np.zeros((256, 3), dtype=np.uint8)
        # Dry Land (NDWI < 0.0 -> idx 0..127):
        lut_ndwi[:128, 0] = np.linspace(175, 120, 128).astype(np.uint8)
        lut_ndwi[:128, 1] = np.linspace(140, 95, 128).astype(np.uint8)
        lut_ndwi[:128, 2] = np.linspace(70, 55, 128).astype(np.uint8)
        # Moist / transition (0.0 <= NDWI < 0.20 -> idx 128..153):
        lut_ndwi[128:154, 0] = np.linspace(50, 20, 26).astype(np.uint8)
        lut_ndwi[128:154, 1] = np.linspace(150, 190, 26).astype(np.uint8)
        lut_ndwi[128:154, 2] = np.linspace(180, 220, 26).astype(np.uint8)
        # Open Water (NDWI >= 0.20 -> idx 154..255):
        lut_ndwi[154:, 0] = np.linspace(15, 5, 102).astype(np.uint8)
        lut_ndwi[154:, 1] = np.linspace(70, 30, 102).astype(np.uint8)
        lut_ndwi[154:, 2] = np.linspace(220, 255, 102).astype(np.uint8)

        img_ndwi = Image.fromarray(lut_ndwi[ndwi_norm])
        img_ndwi.thumbnail((384, 384), Image.Resampling.LANCZOS)
        buf_ndwi = io.BytesIO()
        img_ndwi.save(buf_ndwi, format="PNG")
        ndwi_b64 = f"data:image/png;base64,{base64.b64encode(buf_ndwi.getvalue()).decode('utf-8')}"

        return {
            "applicable": True,
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

