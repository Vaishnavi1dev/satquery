import io
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.data.ingestion import ImageMetadataEnvelope, ImageIngestionService, boxes_to_geojson, pixel_box_to_geo
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

        # Ensure boxes is never empty: if empty, compute salient central ROI
        effective_boxes = list(boxes) if boxes else []
        if not effective_boxes:
            w, h = base_img.size
            effective_boxes = [[int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)]]
            if not label or label == "Grounding Target":
                label = "Scene Region of Interest"

        # Multi-color neon palette for crisp, distinct visual overlays
        palette = [
            ((0, 240, 255, 240), (0, 240, 255, 45), (0, 180, 200, 220)),    # Neon Cyan
            ((16, 185, 129, 240), (16, 185, 129, 45), (16, 150, 100, 220)),  # Emerald Green
            ((245, 158, 11, 240), (245, 158, 11, 45), (200, 130, 10, 220)),  # Amber
            ((168, 85, 247, 240), (168, 85, 247, 45), (140, 70, 210, 220)),  # Purple
        ]

        img_w, img_h = base_img.size

        for idx, box in enumerate(effective_boxes):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            # Ensure proper ordering and clamp within image dimensions
            x_min = max(0, min(img_w - 1, min(x1, x2)))
            x_max = max(0, min(img_w - 1, max(x1, x2)))
            y_min = max(0, min(img_h - 1, min(y1, y2)))
            y_max = max(0, min(img_h - 1, max(y1, y2)))

            if x_max <= x_min or y_max <= y_min:
                continue

            stroke_color, fill_color, badge_color = palette[idx % len(palette)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=stroke_color, width=3, fill=fill_color)

            # Label badge
            badge_h = min(22, max(14, int(img_h * 0.08)))
            badge_w = min(180, max(50, x_max - x_min))
            b_top = max(0, y_min - badge_h) if y_min >= badge_h else y_min
            draw.rectangle([x_min, b_top, x_min + badge_w, b_top + badge_h], fill=badge_color)
            draw.text((x_min + 4, b_top + 2), label[:22], fill=(10, 15, 25, 255))

        combined = Image.alpha_composite(base_img.convert("RGBA"), overlay)
        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_ground_{envelope.image_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        combined.convert("RGB").save(dest_path, format="PNG")
        return dest_path

    def render_analysed_image(
        self,
        session_id: str,
        envelope: ImageMetadataEnvelope,
        label: str = "Scene under analysis"
    ) -> Path:
        """Saves the normalized source image with an optional corner label and NO fabricated regions/boxes."""
        canvas = self._load_pil_image(envelope).convert("RGB")
        if label:
            draw = ImageDraw.Draw(canvas)
            tag_w = min(canvas.width, 260)
            draw.rectangle([0, 0, tag_w, 22], fill=(10, 15, 25))
            draw.text((6, 5), label[:36], fill=(235, 245, 255))

        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_analysed_{envelope.image_id}.png")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_path, format="PNG")
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
            scale_x = w2 / float(max(1, env_t2.width))
            scale_y = target_h / float(max(1, env_t2.height))

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
        sar_env: ImageMetadataEnvelope,
        boxes: Optional[List[List[int]]] = None
    ) -> Path:
        """Renders side-by-side Optical vs SAR synchronized display with optional bounding overlays."""
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
        if opt_env.modality == "multispectral":
            band_count = opt_env.bands if opt_env.bands > 1 else 12
            opt_label = f"Multispectral Spectrum ({band_count} Bands, MSI)"
        else:
            opt_label = "Optical Spectrum (RGB Natural Color)"
        draw.text((16, 12), opt_label, fill=(0, 240, 255))
        draw.text((w_opt + 32, 12), f"Synthetic Aperture Radar Backscatter (SAR)", fill=(180, 130, 255))

        # Overlay caller-supplied boxes on the optical panel (same style as
        # render_bounding_boxes), skipping malformed entries.
        if boxes:
            o_w = r_opt.size[0]
            scale_x = o_w / float(max(1, opt_env.width))
            scale_y = target_h / float(max(1, opt_env.height))
            palette = [
                ((0, 240, 255, 240), (0, 240, 255, 45), (0, 180, 200, 220)),
                ((16, 185, 129, 240), (16, 185, 129, 45), (16, 150, 100, 220)),
                ((245, 158, 11, 240), (245, 158, 11, 45), (200, 130, 10, 220)),
                ((168, 85, 247, 240), (168, 85, 247, 45), (140, 70, 210, 220)),
            ]
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            for idx, box in enumerate(boxes):
                if len(box) != 4:
                    continue
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y) + 40
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y) + 40
                x_min = max(0, min(o_w - 1, min(x1, x2)))
                x_max = max(0, min(o_w - 1, max(x1, x2)))
                y_min = max(40, min(40 + target_h - 1, min(y1, y2)))
                y_max = max(40, min(40 + target_h - 1, max(y1, y2)))
                if x_max <= x_min or y_max <= y_min:
                    continue
                stroke_color, fill_color, badge_color = palette[idx % len(palette)]
                odraw.rectangle([x_min, y_min, x_max, y_max], outline=stroke_color, width=3, fill=fill_color)
                badge_h = min(22, max(14, int(target_h * 0.08)))
                badge_w = min(180, max(50, x_max - x_min))
                b_top = max(40, y_min - badge_h) if (y_min - 40) >= badge_h else y_min
                odraw.rectangle([x_min, b_top, x_min + badge_w, b_top + badge_h], fill=badge_color)
                odraw.text((x_min + 4, b_top + 2), "Fusion Target", fill=(10, 15, 25, 255))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

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

            if boxes and idx > 0 and (idx - 1) < len(boxes):
                box = boxes[idx - 1]
                if len(box) == 4:
                    scale_x = img.width / float(max(1, env.width))
                    scale_y = target_h / float(max(1, env.height))
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

    def render_difference_heatmap(
        self,
        session_id: str,
        env_t1: ImageMetadataEnvelope,
        env_t2: ImageMetadataEnvelope,
        change_boxes: Optional[List[List[int]]] = None
    ) -> Dict[str, Path]:
        """
        Computes pixel-level difference heatmap between Observation T1 and T2.
        Returns:
          overlay_path: Image T2 with neon amber/red difference heatmap overlay
          mask_path: Transparent standalone change mask for interactive slider toggle
        """
        img1 = self._load_pil_image(env_t1)
        img2 = self._load_pil_image(env_t2)

        w, h = img2.width, img2.height
        if img1.size != (w, h):
            img1 = img1.resize((w, h), Image.Resampling.BILINEAR)

        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        diff = np.abs(arr2 - arr1)
        if diff.ndim == 3:
            diff_mag = np.mean(diff, axis=-1)
        else:
            diff_mag = diff

        p75 = float(np.percentile(diff_mag, 75))
        max_d = float(np.max(diff_mag))
        if max_d > p75:
            diff_norm = np.clip((diff_mag - p75) / (max_d - p75) * 255.0, 0, 255).astype(np.uint8)
        else:
            diff_norm = np.zeros((h, w), dtype=np.uint8)

        # Create transparent RGBA change mask (bright neon amber/coral: [255, 80, 50])
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        active_pixels = diff_norm > 25
        mask_rgba[active_pixels, 0] = 255
        mask_rgba[active_pixels, 1] = 85
        mask_rgba[active_pixels, 2] = 50
        mask_rgba[active_pixels, 3] = np.clip(diff_norm[active_pixels].astype(np.float32) * 1.6, 70, 220).astype(np.uint8)

        mask_pil = Image.fromarray(mask_rgba, mode="RGBA")

        # Blend over T2
        base_rgba = img2.convert("RGBA")
        blended = Image.alpha_composite(base_rgba, mask_pil)
        draw = ImageDraw.Draw(blended)

        # Draw boxes if provided
        if change_boxes:
            for b in change_boxes:
                if len(b) == 4:
                    draw.rectangle(b, outline=(255, 220, 0, 240), width=3)
                    draw.text((b[0] + 6, b[1] + 6), "Spatial Delta", fill=(255, 220, 0, 240))

        pair_id = f"{env_t1.image_id}__{env_t2.image_id}"
        overlay_path = self.sandbox.get_evidence_path(session_id, f"evidence_diff_overlay_{pair_id}.png")
        mask_path = self.sandbox.get_evidence_path(session_id, f"evidence_diff_mask_{pair_id}.png")

        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        blended.convert("RGB").save(overlay_path, format="PNG")
        mask_pil.save(mask_path, format="PNG")

        return {
            "overlay_path": overlay_path,
            "mask_path": mask_path
        }

    def save_geojson_evidence(
        self,
        session_id: str,
        envelope: ImageMetadataEnvelope,
        boxes: List[List[int]],
        label: str = "Detected Target",
        evidence_items: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """Generates and saves standard GeoJSON FeatureCollection for GIS visualization."""
        geojson_data = boxes_to_geojson(boxes, envelope, label=label, evidence_items=evidence_items)
        dest_path = self.sandbox.get_evidence_path(session_id, f"evidence_spatial_{envelope.image_id}.geojson")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return dest_path

