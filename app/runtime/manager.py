import logging
import os
import json
import gc
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.config import get_base_dir, load_app_config

logger = logging.getLogger(__name__)


class ModelExecutionError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ModelRuntimeManager:
    """
    Manages model loading, device placement, load group eviction,
    and execution between Model A (EarthDial Grounding), Model B (DOFA Hypernetwork), and Model C (EarthDial-4B Multi-Modal).
    """

    def __init__(self):
        self.config = load_app_config()
        self.device = self._resolve_device()
        self.active_models: Dict[str, Any] = {}
        self.load_errors: Dict[str, str] = {}
        self.current_load_group: Optional[str] = None
        self.mode = self.config.runtime.mode  # "simulation" or "real"

    def _resolve_device(self) -> str:
        pref = self.config.runtime.device.lower()
        if pref == "cpu":
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available() and pref in ("auto", "cuda"):
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and pref in ("auto", "mps"):
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def ensure_model_loaded(self, model_key: str, load_group: str):
        """Load a checkpoint-backed model when possible, otherwise retain simulation mode.

        Loading is deliberately lazy because the two adapted models are large.  The
        runtime never downloads weights implicitly; a base model must already exist
        locally unless ``allow_model_downloads`` is explicitly enabled.
        """
        if model_key in self.active_models:
            return self.active_models[model_key].get("handle")

        # Evict conflicting load group if memory is constrained
        if load_group == "llm_secondary" and "llm_primary" in self.active_models:
            logger.info("Evicting llm_primary to accommodate llm_secondary under VRAM budget.")
            self.unload_group("llm_primary")
        elif load_group == "llm_primary" and "llm_secondary" in self.active_models:
            logger.info("Evicting llm_secondary to accommodate llm_primary under VRAM budget.")
            self.unload_group("llm_secondary")

        logger.info(f"Loading {model_key} (Group: {load_group}, Device: {self.device}, Mode: {self.mode})...")
        handle = None
        error = None
        if self.mode == "real":
            try:
                handle = self._load_real_model(model_key)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.load_errors[model_key] = error
                logger.warning("Checkpoint load failed for %s; using simulation fallback: %s", model_key, error)

        self.active_models[model_key] = {
            "loaded_at": time.time(),
            "load_group": load_group,
            "device": self.device,
            "handle": handle,
            "backend": "checkpoint" if handle is not None else "simulation",
            "error": error,
        }
        return handle

    def _resolve_path(self, value: Optional[str]) -> Path:
        path = Path(value or "")
        return path if path.is_absolute() else get_base_dir() / path

    def _artifact_dir(self, model_key: str) -> Optional[Path]:
        store = self.config.model_store
        if model_key == "dofa-fusion":
            roots = [store.dofa_model_path]
            filename = "fusion_head.pt"
        else:
            roots = [store.earthdial_bigearthnet_model_path, store.earthdial_model_path]
            filename = "adapter_model.safetensors"

        candidates: List[Path] = []
        for root in roots:
            root_path = self._resolve_path(root)
            if not root_path.exists():
                continue
            if root_path.is_file() and root_path.name == filename:
                candidates.append(root_path.parent)
            else:
                candidates.extend(p.parent for p in root_path.rglob(filename))

        if not candidates:
            return None

        def checkpoint_rank(path: Path):
            manifest_path = path / "adapter_manifest.json"
            rank = 0
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    rank = int(manifest.get("step", 0)) * 1_000_000 + int(manifest.get("epoch", 0))
                    if model_key == "dofa-fusion":
                        rank = max(rank, int(manifest.get("epochs", 0)))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            for part in path.parts:
                if part.startswith("epoch_"):
                    try:
                        rank = max(rank, int(part.split("_", 1)[1]) * 1_000_000)
                    except ValueError:
                        pass
            if path.name == "ckpt_latest":
                rank += 500_000_000
            return rank, (path / filename).stat().st_mtime

        return max(candidates, key=checkpoint_rank)

    def get_checkpoint_dir(self, model_key: str) -> Optional[Path]:
        """Return the highest-step local artifact, including nested HF checkpoints."""
        return self._artifact_dir(model_key)

    def _base_model_name(self, adapter_dir: Path) -> str:
        configured = self.config.runtime.base_model_path or self.config.model_store.earthdial_base_model_path
        if configured:
            return str(self._resolve_path(configured))
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return str(data.get("base_model_name_or_path", "OpenGVLab/InternVL2-4B"))
        return "OpenGVLab/InternVL2-4B"

    def _load_real_model(self, model_key: str):
        import torch

        artifact_dir = self._artifact_dir(model_key)
        if artifact_dir is None:
            raise FileNotFoundError(f"No checkpoint artifact found for '{model_key}'")

        if model_key == "dofa-fusion":
            from training.dofa.train_dofa_fusion import ModelBDOFAFusion

            checkpoint = artifact_dir / "fusion_head.pt"
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            vocab_size = int(state.get("lm_head.weight", torch.empty(32000, 768)).shape[0])
            model = ModelBDOFAFusion(vocab_size=vocab_size)
            model.fusion_head.load_state_dict(state, strict=True)
            model.eval()
            if self.device != "cpu":
                model.to(self.device)
            return {"kind": "dofa", "model": model, "checkpoint_dir": str(artifact_dir)}

        import transformers
        from peft import PeftModel

        base_name = self._base_model_name(artifact_dir)
        local_only = not self.config.runtime.allow_model_downloads
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(artifact_dir), trust_remote_code=True, use_fast=False, local_files_only=True
        )
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_only,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        if self.device == "cuda" and self.config.runtime.quantization in ("int4", "4bit"):
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except (ImportError, Exception) as exc:
                logger.warning("4-bit quantization unavailable; attempting CPU offload: %s", exc)
        model_cls = getattr(transformers, "AutoModel", transformers.AutoModelForCausalLM)
        model = model_cls.from_pretrained(base_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(artifact_dir), is_trainable=False)
        model.eval()
        if self.device != "cpu":
            model.to(self.device)
        return {
            "kind": "earthdial",
            "model": model,
            "tokenizer": tokenizer,
            "checkpoint_dir": str(artifact_dir),
            "base_model": base_name,
        }

    def get_model_handle(self, model_key: str) -> Optional[Dict[str, Any]]:
        record = self.active_models.get(model_key)
        return record.get("handle") if record else None

    def _image_tensor(self, filepath: str, size: int = 512):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(filepath) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def run_earthdial(self, query: str, image_paths: List[str], parameters: Dict[str, Any]):
        handle = self.get_model_handle("earthdial-4b")
        if not handle or handle.get("kind") != "earthdial":
            return None

        model = handle["model"]
        tokenizer = handle["tokenizer"]
        valid_paths = [path for path in image_paths if path and Path(path).exists()]
        tensors = [self._image_tensor(path) for path in valid_paths]
        if not tensors or not hasattr(model, "chat"):
            return None

        import torch
        pixel_values = torch.cat(tensors, dim=0)
        prompt = "\n".join(["<image>"] * len(tensors)) + "\n" + query
        generation_config = {
            "max_new_tokens": parameters.get("max_new_tokens", 256),
            "temperature": parameters.get("temperature", 0.0),
            "do_sample": parameters.get("do_sample", False),
        }
        with torch.inference_mode():
            chat_kwargs = {
                "tokenizer": tokenizer,
                "pixel_values": pixel_values,
                "question": prompt,
                "generation_config": generation_config,
            }
            # InternVL accepts this argument for multi-image conversations;
            # retry without it for compatible single-image/custom wrappers.
            if len(tensors) > 1:
                chat_kwargs["num_patches_list"] = [1] * len(tensors)
            try:
                response = model.chat(**chat_kwargs)
            except TypeError:
                chat_kwargs.pop("num_patches_list", None)
                response = model.chat(**chat_kwargs)
        if isinstance(response, tuple):
            response = response[0]
        return {"text": str(response).strip(), "checkpoint_dir": handle["checkpoint_dir"]}

    def run_dofa_fusion(self, optical_path: str, sar_path: str, optical_wavelengths: List[float], sar_wavelengths: List[float]):
        handle = self.get_model_handle("dofa-fusion")
        if not handle or handle.get("kind") != "dofa":
            return None

        import torch
        opt = self._image_tensor(optical_path, size=224)
        sar = self._image_tensor(sar_path, size=224)
        opt_wls = torch.tensor([optical_wavelengths[:3]], dtype=torch.float32, device=self.device)
        sar_values = (sar_wavelengths + [sar_wavelengths[-1]])[:3]
        sar_wls = torch.tensor([sar_values], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            output = handle["model"](opt, sar, opt_wls, sar_wls)
            logits = output["logits"]
            token_confidence = torch.softmax(logits.mean(dim=1), dim=-1).amax(dim=-1).item()
        return {
            "checkpoint_dir": handle["checkpoint_dir"],
            "logits_shape": list(logits.shape),
            "token_confidence": round(float(token_confidence), 6),
        }

    def get_artifact_status(self) -> Dict[str, Any]:
        status = {}
        for key in ("earthdial-4b", "dofa-fusion"):
            path = self._artifact_dir(key)
            record = self.active_models.get(key, {})
            status[key] = {
                "checkpoint_dir": str(path) if path else None,
                "available": path is not None,
                "backend": record.get("backend", "not_loaded"),
                "load_error": self.load_errors.get(key),
            }
        return status

    def unload_group(self, load_group: str):
        to_remove = [k for k, v in self.active_models.items() if v.get("load_group") == load_group]
        for k in to_remove:
            del self.active_models[k]
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_runtime_status(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "mode": self.mode,
            "vram_budget_gb": self.config.runtime.vram_budget_gb,
            "quantization": self.config.runtime.quantization,
            "active_models": list(self.active_models.keys()),
            "deterministic": self.config.runtime.deterministic,
            "artifacts": self.get_artifact_status(),
        }
