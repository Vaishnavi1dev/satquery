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
