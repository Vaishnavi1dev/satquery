import io
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.data.ingestion import ImageMetadataEnvelope, ImageIngestionService
from app.storage.sandbox import StorageSandbox


class EvidenceRenderer:
    """Renders visual overlays, bounding boxes, change maps, and multi-sensor comparisons."""

    def __init__(self, sandbox: StorageSandbox):
        self.sandbox = sandbox
        self.ingestion_svc = ImageIngestionService()

    def _load_pil_image(self, envelope: ImageMetadataEnvelope) -> Image.Image:
        try:
            raw_data, _ = self.ingestion_svc.read_image_data(Path(envelope.filepath))
            if raw_data.ndim == 2:
                norm = np.clip(raw_data, 0, 255).astype(np.uint8)
                return Image.fromarray(norm).convert("RGB")
            elif raw_data.ndim == 3:
                if raw_data.shape[2] >= 3:
                    sub = raw_data[:, :, :3]
                elif raw_data.shape[2] == 2:
                    sub = np.dstack([raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 0]])
                else:
                    sub = np.repeat(raw_data, 3, axis=-1)

                p2 = np.percentile(sub, 2)
                p98 = np.percentile(sub, 98)
                if p98 > p2:
                    norm = np.clip((sub - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                else:
                    norm = np.zeros_like(sub, dtype=np.uint8)
                return Image.fromarray(norm)
        except Exception:
            return Image.new("RGB", (512, 512), color=(20, 25, 35))

    def render_bounding_boxes(
        self,
        session_id: str,
        envelope: ImageMetadataEnvelope,
        boxes: List[List[int]],
        label: str = "Grounding Target"
    ) -> Path:
        """Renders bounding boxes [x1, y1, x2, y2] onto the source image with neon accent colors."""
        base_img = self._load_pil_image(envelope)
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Neon cyan accent with semi-transparent fill
        box_stroke = (0, 240, 255, 240)
        box_fill = (0, 240, 255, 45)

        for box in boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            # Ensure proper ordering
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_stroke, width=3, fill=box_fill)

            # Label badge
            badge_h = 24
            badge_w = min(180, x_max - x_min)
            draw.rectangle([x_min, max(0, y_min - badge_h), x_min + badge_w, y_min], fill=(0, 180, 200, 220))
            draw.text((x_min + 6, max(2, y_min - badge_h + 4)), label[:22], fill=(10, 15, 25, 255))

        combined = Image.alpha_composite(base_img.convert("RGBA"), overlay)
        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_ground_{envelope.image_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        combined.convert("RGB").save(dest_path, format="PNG")
        return dest_path

    def render_bi_temporal_change(
        self,
        session_id: str,
        env_t1: ImageMetadataEnvelope,
        env_t2: ImageMetadataEnvelope,
        change_boxes: Optional[List[List[int]]] = None
    ) -> Path:
        """Renders side-by-side bi-temporal comparison with highlighted change areas."""
        img1 = self._load_pil_image(env_t1)
        img2 = self._load_pil_image(env_t2)

        # Harmonize height
        target_h = 512
        w1 = int(img1.width * (target_h / img1.height))
        w2 = int(img2.width * (target_h / img2.height))

        r1 = img1.resize((w1, target_h), Image.Resampling.BILINEAR)
        r2 = img2.resize((w2, target_h), Image.Resampling.BILINEAR)

        total_w = w1 + w2 + 16
        canvas = Image.new("RGB", (total_w, target_h + 40), color=(15, 20, 30))
        canvas.paste(r1, (0, 40))
        canvas.paste(r2, (w1 + 16, 40))

        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"Observation T1 (Baseline) - {env_t1.filename}", fill=(200, 220, 255))
        draw.text((w1 + 32, 12), f"Observation T2 (Follow-up) - {env_t2.filename}", fill=(255, 180, 50))

        # Highlight change boxes on T2
        if change_boxes:
            scale_x = w2 / float(env_t2.width)
            scale_y = target_h / float(env_t2.height)

            for box in change_boxes:
                if len(box) == 4:
                    x1 = int(box[0] * scale_x) + w1 + 16
                    y1 = int(box[1] * scale_y) + 40
                    x2 = int(box[2] * scale_x) + w1 + 16
                    y2 = int(box[3] * scale_y) + 40
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
                    draw.text((x1 + 6, y1 + 6), "Detected Change Zone", fill=(255, 80, 80))

        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_change_{env_t1.image_id}__{env_t2.image_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_path, format="PNG")
        return dest_path

    def render_optical_sar_split(
        self,
        session_id: str,
        opt_env: ImageMetadataEnvelope,
        sar_env: ImageMetadataEnvelope
    ) -> Path:
        """Renders side-by-side Optical vs SAR synchronized display."""
        img_opt = self._load_pil_image(opt_env)
        img_sar = self._load_pil_image(sar_env)

        target_h = 512
        w_opt = int(img_opt.width * (target_h / img_opt.height))
        w_sar = int(img_sar.width * (target_h / img_sar.height))

        r_opt = img_opt.resize((w_opt, target_h), Image.Resampling.BILINEAR)
        r_sar = img_sar.resize((w_sar, target_h), Image.Resampling.BILINEAR)

        total_w = w_opt + w_sar + 16
        canvas = Image.new("RGB", (total_w, target_h + 40), color=(12, 16, 24))
        canvas.paste(r_opt, (0, 40))
        canvas.paste(r_sar, (w_opt + 16, 40))

        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"Optical / Multispectral Spectrum ({opt_env.modality.upper()})", fill=(0, 240, 255))
        draw.text((w_opt + 32, 12), f"Synthetic Aperture Radar Backscatter (SAR)", fill=(180, 130, 255))

        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_fusion_{opt_env.image_id}__{sar_env.image_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_path, format="PNG")
        return dest_path

    def render_temporal_sequence_filmstrip(
        self,
        session_id: str,
        envelopes: List[ImageMetadataEnvelope],
        boxes: Optional[List[List[int]]] = None
    ) -> Path:
        """Renders an aligned chronological timeline filmstrip across N >= 3 satellite epochs."""
        images = [self._load_pil_image(env) for env in envelopes]
        target_h = 360
        resized = []
        for img in images:
            w = int(img.width * (target_h / img.height))
            resized.append(img.resize((w, target_h), Image.Resampling.BILINEAR))

        total_w = sum(img.width for img in resized) + (len(resized) - 1) * 16 + 24
        canvas = Image.new("RGB", (total_w, target_h + 60), color=(10, 14, 22))
        draw = ImageDraw.Draw(canvas)

        curr_x = 12
        for idx, (img, env) in enumerate(zip(resized, envelopes)):
            canvas.paste(img, (curr_x, 50))
            epoch_label = f"Epoch T{idx + 1} - {env.filename[:20]}"
            color = (0, 240, 255) if idx == 0 else ((255, 200, 80) if idx == len(envelopes) - 1 else (200, 220, 255))
            draw.text((curr_x + 4, 18), epoch_label, fill=color)

            if idx > 0:
                draw.text((curr_x - 12, 50 + target_h // 2 - 10), "→", fill=(255, 255, 255))

            if boxes and (idx - 1) < len(boxes) and idx > 0:
                box = boxes[idx - 1]
                scale_x = img.width / float(env.width)
                scale_y = target_h / float(env.height)
                bx1 = int(box[0] * scale_x) + curr_x
                by1 = int(box[1] * scale_y) + 50
                bx2 = int(box[2] * scale_x) + curr_x
                by2 = int(box[3] * scale_y) + 50
                draw.rectangle([bx1, by1, bx2, by2], outline=(255, 80, 80), width=3)
                draw.text((bx1 + 4, by1 + 4), f"Δ Event T{idx}→T{idx+1}", fill=(255, 100, 100))

            curr_x += img.width + 16

        seq_id = f"{envelopes[0].image_id}__to__{envelopes[-1].image_id}"
        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_sequence_{seq_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_path, format="PNG")
        return dest_path
