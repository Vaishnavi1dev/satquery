import logging
import os
import json
import gc
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.config import get_base_dir, load_app_config

logger = logging.getLogger(__name__)

EARTHDIAL_REMOTE_CODE_FILES = (
    "configuration_internvl_chat.py",
    "configuration_intern_vit.py",
    "configuration_phi3.py",
    "conversation.py",
    "modeling_internvl_chat.py",
    "modeling_intern_vit.py",
    "modeling_phi3.py",
    "preprocessor_config.json",
)


EARTHDIAL_MODEL_KEYS = ("earthdial-4b", "earthdial-original", "earthdial-ms")


def earthdial_model_key_for(modalities, default_key: str) -> str:
    """Route MS/SAR inputs to the multispectral EarthDial checkpoint.

    Returns ``"earthdial-ms"`` when any supplied modality is ``"sar"`` or
    ``"multispectral"``; otherwise returns ``default_key`` so optical inputs keep
    their existing single-image (``earthdial-4b``) and multi-temporal
    (``earthdial-original``) checkpoints.
    """
    normalized = {str(m).lower() for m in modalities if m}
    if "sar" in normalized or "multispectral" in normalized:
        return "earthdial-ms"
    return default_key


class ModelExecutionError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _install_legacy_cache_compat() -> None:
    """Restore cache accessors the vendored remote Phi3 code still expects.

    transformers 4.57 dropped ``Cache.get_usable_length``/``get_max_length``.
    Adding thin aliases back onto the cache classes keeps the remote attention
    layers working without editing the trust_remote_code files. Best effort only.
    """
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return

    def _get_usable_length(self, new_seq_length, layer_idx=0):
        get_max = getattr(self, "get_max_cache_shape", None)
        try:
            max_length = get_max() if callable(get_max) else None
        except Exception:
            max_length = None
        try:
            previous_seq_length = self.get_seq_length(layer_idx)
        except TypeError:
            previous_seq_length = self.get_seq_length()
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def _get_max_length(self):
        get_max = getattr(self, "get_max_cache_shape", None)
        try:
            return get_max() if callable(get_max) else None
        except Exception:
            return None

    for cls in (Cache, DynamicCache):
        try:
            if not hasattr(cls, "get_usable_length"):
                cls.get_usable_length = _get_usable_length
            if not hasattr(cls, "get_max_length"):
                cls.get_max_length = _get_max_length
        except Exception:
            pass


def _earthdial_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    """Version-compatible replacement for the bundled remote InternVL/Phi3 helper.

    The vendored remote code still reads ``past_key_values.seen_tokens``, which was
    removed from the transformers ``Cache`` API. This mirrors the original behaviour
    (using the current accessors) and keeps ``inputs_embeds`` for the first,
    not-yet-cached step, which is how InternVL feeds its ``<image>`` embeddings.
    """
    try:
        from transformers.cache_utils import Cache
    except Exception:
        Cache = ()

    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
            cache_length = past_length
            get_max = getattr(past_key_values, "get_max_cache_shape", None)
            max_cache_length = get_max() if callable(get_max) else None
        else:
            past_length = cache_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    if inputs_embeds is not None and past_length == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


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
        self._lock = threading.RLock()
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

    @staticmethod
    def earthdial_model_key_for(modalities, default_key: str) -> str:
        """Select the EarthDial checkpoint for the given input modalities."""
        return earthdial_model_key_for(modalities, default_key)

    def ensure_model_loaded(self, model_key: str, load_group: str):
        """Load a checkpoint-backed model when possible, otherwise retain simulation mode.

        Loading is deliberately lazy because the adapted models are large. The runtime
        never downloads weights implicitly; a base model must already exist locally
        unless ``allow_model_downloads`` is explicitly enabled.

        The whole membership check / eviction / load / insert sequence runs under a
        re-entrant lock so concurrent requests cannot each load a 4B checkpoint and
        leave two EarthDial variants resident at once.
        """
        if model_key in self.active_models:
            record = self.active_models[model_key]
            if record.get("handle") is not None or self.mode != "real":
                return record.get("handle")

        with self._lock:
            # Double-checked: a concurrent loader may have populated this key while we
            # were waiting for the lock. A stale real-mode failure record is dropped so
            # the requested load is retried rather than cached forever.
            if model_key in self.active_models:
                record = self.active_models[model_key]
                if record.get("handle") is not None or self.mode != "real":
                    return record.get("handle")
                del self.active_models[model_key]

            # Evict the conflicting EarthDial load group before loading a new one so the
            # fine-tuned adapter and the original weights never coexist on the GPU.
            def _group_active(group: str) -> bool:
                with self._lock:
                    return any(rec.get("load_group") == group for rec in self.active_models.values())

            if load_group == "llm_secondary" and _group_active("llm_primary"):
                logger.info("Evicting llm_primary to accommodate llm_secondary under VRAM budget.")
                self.unload_group("llm_primary")
            elif load_group == "llm_primary" and _group_active("llm_secondary"):
                logger.info("Evicting llm_secondary to accommodate llm_primary under VRAM budget.")
                self.unload_group("llm_secondary")

            # All EarthDial checkpoints are ~4B and cannot co-reside in the 8 GB
            # budget, even when they share a load group (single-image optical vs SAR).
            # Release the other EarthDial variants first; never touch DOFA.
            if model_key in EARTHDIAL_MODEL_KEYS:
                conflicting = [
                    key for key in EARTHDIAL_MODEL_KEYS
                    if key != model_key and key in self.active_models
                ]
                if conflicting:
                    logger.info(
                        "Evicting EarthDial variant(s) %s before loading %s (VRAM budget).",
                        ", ".join(conflicting),
                        model_key,
                    )
                    self._unload_models(conflicting)

            artifact_dir = self._artifact_dir(model_key)

            logger.info(f"Loading {model_key} (Group: {load_group}, Device: {self.device}, Mode: {self.mode})...")
            handle = None
            error = None
            if self.mode == "real":
                if artifact_dir is None:
                    error = f"No checkpoint artifact found for '{model_key}'"
                else:
                    try:
                        handle = self._load_real_model(model_key)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        logger.warning("Checkpoint load failed for %s; using simulation fallback: %s", model_key, error)

            if handle is not None:
                self.load_errors.pop(model_key, None)
                self.active_models[model_key] = {
                    "loaded_at": time.time(),
                    "load_group": load_group,
                    "device": self.device,
                    "handle": handle,
                    "backend": "checkpoint",
                    "error": None,
                }
                return handle

            # Same-family fallback only: ``earthdial-original`` is the multi-image RGB
            # checkpoint and may alias the fine-tuned RGB adapter. ``earthdial-ms`` is a
            # different modality family and must never be served by an RGB handle.
            if model_key == "earthdial-original" and self.mode == "real":
                self._free_memory()
                try:
                    fallback_handle = self.ensure_model_loaded("earthdial-4b", "llm_primary")
                except Exception as exc:
                    logger.warning("Fallback load of earthdial-4b failed: %s", exc)
                    fallback_handle = None
                if fallback_handle is not None:
                    fallback_record = self.active_models.get("earthdial-4b", {})
                    reason = error or f"{model_key} weights unavailable"
                    logger.info(
                        "Aliasing the fine-tuned earthdial-4b handle for %s (%s).", model_key, reason
                    )
                    self.active_models[model_key] = {
                        "loaded_at": time.time(),
                        "load_group": load_group,
                        "device": fallback_record.get("device", self.device),
                        "handle": fallback_handle,
                        "backend": fallback_record.get("backend", "checkpoint"),
                        "error": (
                            f"{model_key} load failed ({reason}); "
                            "using fine-tuned earthdial-4b adapter handle"
                        ),
                    }
                    return fallback_handle

            if self.mode == "real":
                # Failures are never terminal: no cache entry is written so the next
                # request retries the load, and the reason is surfaced for status.
                if error:
                    self.load_errors[model_key] = error
                return None

            # Simulation mode: keep a truthful non-resident record.
            self.active_models[model_key] = {
                "loaded_at": time.time(),
                "load_group": load_group,
                "device": self.device,
                "handle": None,
                "backend": "simulation",
                "error": error,
            }
            return None

    def _resolve_path(self, value: Optional[str]) -> Path:
        path = Path(value or "")
        return path if path.is_absolute() else get_base_dir() / path

    def _read_json(self, path: Path) -> Optional[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _has_weight_file(path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "fusion_head.pt").is_file():
            return True
        if (path / "adapter_model.safetensors").is_file():
            return True
        return any(p.is_file() for p in path.glob("*.safetensors"))

    def _ready_checkpoint_dir(self, path: Path, adapter: bool = False) -> Optional[Path]:
        """Return ``path`` only when it is a complete, loadable checkpoint directory.

        A partially downloaded artifact (config only, or weights only) must not be
        reported as available. Full checkpoints need a parseable ``config.json`` plus a
        weight file; PEFT adapters need a parseable ``adapter_config.json`` (or
        ``config.json``) plus their adapter weights.
        """
        if not path.is_dir():
            return None
        config_names = ("adapter_config.json", "config.json") if adapter else ("config.json",)
        if not any(self._read_json(path / name) is not None for name in config_names):
            return None
        if not self._has_weight_file(path):
            return None
        return path

    def _artifact_dir(self, model_key: str) -> Optional[Path]:
        store = self.config.model_store
        if model_key == "dofa-fusion":
            roots = [store.dofa_model_path]
            filename = "fusion_head.pt"
        elif model_key == "earthdial-original":
            return self._ready_checkpoint_dir(self._resolve_path(store.earthdial_model_path))
        elif model_key == "earthdial-ms":
            return self._ready_checkpoint_dir(self._resolve_path(store.earthdial_ms_model_path))
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

        best = max(candidates, key=checkpoint_rank)
        if model_key == "dofa-fusion":
            return best if (best / filename).is_file() else None
        return self._ready_checkpoint_dir(best, adapter=True)

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

    def _ensure_earthdial_remote_code(self, artifact_dir: Path) -> None:
        """Best-effort copy of the InternVL2 remote-code files into an artifact dir.

        A fresh clone only ships the fine-tuned EarthDial weights; the
        ``trust_remote_code`` modules live in the configured base-model dir. Copy
        any missing file, never overwriting existing ones, and never raise so the
        eventual model load can surface its own error.
        """
        try:
            configured = (
                self.config.model_store.earthdial_base_model_path
                or self.config.runtime.base_model_path
                or "models/InternVL2-4B"
            )
            base_dir = self._resolve_path(configured)
        except Exception as exc:
            logger.warning("Could not resolve EarthDial base-model dir: %s", exc)
            return

        for filename in EARTHDIAL_REMOTE_CODE_FILES:
            destination = artifact_dir / filename
            if destination.exists():
                continue
            source = base_dir / filename
            if not source.exists():
                logger.warning(
                    "EarthDial remote-code file %s missing from base model dir %s; skipping.",
                    filename,
                    base_dir,
                )
                continue
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(destination))
                logger.info(
                    "Provisioned EarthDial remote-code file %s from %s.", filename, base_dir
                )
            except Exception as exc:
                logger.warning(
                    "Could not provision EarthDial remote-code file %s: %s", filename, exc
                )

    @staticmethod
    def _free_memory() -> None:
        """Release Python and CUDA allocator memory before (re)loading a large model."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_earthdial_tokenizer(self, artifact_dir: Path, local_only: bool):
        """Load the EarthDial tokenizer, retrying with the fast implementation.

        ``local_files_only`` follows ``allow_model_downloads`` instead of being pinned
        to ``True``. ``use_fast=False`` is preferred (the InternVL tokenizer ships a
        sentencepiece model), with an automatic ``use_fast=True`` retry on failure.
        """
        import transformers

        last_exc = None
        for use_fast in (False, True):
            try:
                return transformers.AutoTokenizer.from_pretrained(
                    str(artifact_dir),
                    trust_remote_code=True,
                    use_fast=use_fast,
                    local_files_only=local_only,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Tokenizer load (use_fast=%s, local_files_only=%s) failed for %s: %s",
                    use_fast,
                    local_only,
                    artifact_dir,
                    exc,
                )
        raise last_exc

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

        if model_key in ("earthdial-original", "earthdial-ms"):
            self._ensure_earthdial_remote_code(artifact_dir)
            import transformers

            local_only = not self.config.runtime.allow_model_downloads
            tokenizer = self._load_earthdial_tokenizer(artifact_dir, local_only)
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
            model = model_cls.from_pretrained(str(artifact_dir), **model_kwargs)
            model.eval()
            if self.device != "cpu":
                model.to(self.device)
            self._ensure_generation_mixin(model)
            return {
                "kind": "earthdial",
                "model": model,
                "tokenizer": tokenizer,
                "checkpoint_dir": str(artifact_dir),
                "base_model": str(artifact_dir),
                "variant": "original" if model_key == "earthdial-original" else "multispectral",
            }

        import transformers
        from peft import PeftModel

        base_name = self._base_model_name(artifact_dir)
        local_only = not self.config.runtime.allow_model_downloads
        tokenizer = self._load_earthdial_tokenizer(artifact_dir, local_only)
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
        self._ensure_generation_mixin(model)
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

    def _earthdial_preprocessor(self):
        """Best-effort InternVL preprocessing settings (normalization + tile size)."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = 448
        base = self.config.runtime.base_model_path or "models/InternVL2-4B"
        candidates = [
            self._resolve_path(base) / "preprocessor_config.json",
            self._resolve_path("models/InternVL2-4B/preprocessor_config.json"),
        ]
        for path in candidates:
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            image_mean = data.get("image_mean")
            image_std = data.get("image_std")
            if isinstance(image_mean, (list, tuple)) and len(image_mean) == 3:
                try:
                    mean = [float(v) for v in image_mean]
                except (TypeError, ValueError):
                    pass
            if isinstance(image_std, (list, tuple)) and len(image_std) == 3:
                try:
                    std = [float(v) for v in image_std]
                except (TypeError, ValueError):
                    pass
            tile = data.get("crop_size") or data.get("size")
            if isinstance(tile, int) and tile > 0:
                size = tile
            break
        return mean, std, size

    def _earthdial_image_tensor(self, filepath: str, size: int = 448):
        """InternVL-style normalized [3, S, S] float32 pixel tensor (unbatched)."""
        import numpy as np
        import torch
        from PIL import Image

        mean, std, tile = self._earthdial_preprocessor()
        if not isinstance(size, int) or size <= 0:
            size = tile
        with Image.open(filepath) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32)
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
        return tensor.to(self.device)

    def _earthdial_model_dtype(self, model):
        import torch
        try:
            dtype = next(model.parameters()).dtype
            if dtype in (torch.float16, torch.bfloat16, torch.float32):
                return dtype
        except Exception:
            pass
        try:
            for param in model.parameters():
                if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    return param.dtype
        except Exception:
            pass
        return torch.float16

    def _find_remote_language_model(self, model):
        candidates = []
        try:
            candidates.append(getattr(model, "language_model", None))
        except Exception:
            pass
        try:
            base = getattr(model, "base_model", None)
        except Exception:
            base = None
        for obj in (base, getattr(base, "model", None) if base is not None else None):
            if obj is None:
                continue
            try:
                candidates.append(getattr(obj, "language_model", None))
            except Exception:
                pass
        for lm in candidates:
            if lm is not None and hasattr(lm, "get_input_embeddings"):
                return lm
        return None

    def _ensure_generation_mixin(self, model) -> bool:
        """Make the remote Phi3 language model able to run HF generation.

        transformers >= 4.50 no longer puts GenerationMixin in PreTrainedModel, so
        the trust_remote_code phi3 class loses ``generate``. Splice the mixin into
        the class bases after the remote class, install a cache-API compatible
        ``prepare_inputs_for_generation`` (the vendored copy predates the current
        ``Cache`` interface), and ensure a generation config exists because it was
        initialised to ``None`` before the splice.
        """
        try:
            from transformers import GenerationConfig
            from transformers.generation import GenerationMixin
        except Exception as exc:
            logger.warning("GenerationMixin unavailable for EarthDial: %s", exc)
            return False
        _install_legacy_cache_compat()
        try:
            lm = self._find_remote_language_model(model)
            if lm is None:
                return False
            lm_cls = type(lm)
            if not hasattr(lm, "generate"):
                try:
                    if GenerationMixin not in lm_cls.__mro__:
                        lm_cls.__bases__ = lm_cls.__bases__ + (GenerationMixin,)
                except Exception as exc:
                    logger.warning("Could not add GenerationMixin to %s bases: %s", lm_cls.__name__, exc)
                if not hasattr(lm_cls, "generate"):
                    lm_cls.generate = GenerationMixin.generate
            try:
                lm_cls.prepare_inputs_for_generation = _earthdial_prepare_inputs_for_generation
            except Exception as exc:
                logger.warning("Could not install compatible prepare_inputs_for_generation: %s", exc)
            try:
                if getattr(lm, "generation_config", None) is None:
                    lm.generation_config = GenerationConfig.from_model_config(lm.config)
            except Exception as exc:
                logger.warning("Could not initialise remote generation_config: %s", exc)
            return hasattr(lm, "generate")
        except Exception as exc:
            logger.warning("Failed to prepare remote generation for EarthDial: %s", exc)
            return False

    def run_earthdial(
        self,
        query: str,
        image_paths: List[str],
        parameters: Dict[str, Any],
        model_key: Optional[str] = None,
    ):
        handle = None
        requested_key = model_key or (parameters or {}).get("model_key")
        if requested_key:
            # Honor the explicitly requested checkpoint (e.g. the multispectral handle
            # for SAR/MS inputs); never silently run a different modality variant.
            candidate = self.get_model_handle(requested_key)
            if candidate and candidate.get("kind") == "earthdial":
                handle = candidate
        else:
            # Load-group eviction keeps only one EarthDial variant resident at a time.
            for key in ("earthdial-original", "earthdial-4b", "earthdial-ms"):
                candidate = self.get_model_handle(key)
                if candidate and candidate.get("kind") == "earthdial":
                    handle = candidate
                    break

        if not handle:
            return None

        model = handle["model"]
        tokenizer = handle["tokenizer"]
        valid_paths = [path for path in image_paths if path and Path(path).exists()]
        tensors = [self._earthdial_image_tensor(path) for path in valid_paths]
        if not tensors or not hasattr(model, "chat"):
            return None

        import torch
        self._ensure_generation_mixin(model)
        num_images = len(tensors)
        pixel_values = torch.stack(tensors, dim=0)
        try:
            model_dtype = self._earthdial_model_dtype(model)
        except Exception:
            model_dtype = torch.float16
        pixel_values = pixel_values.to(dtype=model_dtype).to(self.device)

        # Present the query in an explicit Q/A frame so the adapted model answers the
        # question instead of always emitting a generic land-cover caption.
        prompt = "\n".join(["<image>"] * num_images) + "\n" + query + "\nAnswer:"
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
                "num_patches_list": [1] * num_images,
            }
            # InternVL accepts this argument for multi-image conversations;
            # retry without it for compatible single-image/custom wrappers.
            try:
                response = model.chat(**chat_kwargs)
            except TypeError:
                chat_kwargs.pop("num_patches_list", None)
                response = model.chat(**chat_kwargs)
        if isinstance(response, tuple):
            response = response[0]
        text = "" if response is None else str(response).strip()
        return {"text": text, "checkpoint_dir": handle["checkpoint_dir"]}

    def run_dofa_fusion(self, optical_path: str, sar_path: str, optical_wavelengths: List[float], sar_wavelengths: List[float]):
        handle = self.get_model_handle("dofa-fusion")
        if not handle or handle.get("kind") != "dofa":
            return None

        import torch
        opt = self._image_tensor(optical_path, size=224)
        sar = self._image_tensor(sar_path, size=224)
        opt_values = list(optical_wavelengths or [])[:3] or [0.665]
        sar_source = list(sar_wavelengths or []) or [0.056]
        sar_values = (sar_source + [sar_source[-1]])[:3]
        opt_wls = torch.tensor([opt_values], dtype=torch.float32, device=self.device)
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
        with self._lock:
            active_snapshot = dict(self.active_models)
            error_snapshot = dict(self.load_errors)
        for key in ("earthdial-4b", "earthdial-original", "earthdial-ms", "dofa-fusion"):
            path = self._artifact_dir(key)
            record = active_snapshot.get(key, {})
            status[key] = {
                "checkpoint_dir": str(path) if path else None,
                "available": path is not None,
                "backend": record.get("backend", "not_loaded"),
                "load_error": error_snapshot.get(key),
            }
        return status

    def _unload_models(self, model_keys):
        """Drop specific model records and release their VRAM (never other groups)."""
        removed = False
        with self._lock:
            for key in list(model_keys):
                if key in self.active_models:
                    del self.active_models[key]
                    removed = True
        if not removed:
            return
        self._free_memory()

    def unload_group(self, load_group: str):
        with self._lock:
            to_remove = [k for k, v in self.active_models.items() if v.get("load_group") == load_group]
            for k in to_remove:
                del self.active_models[k]
        self._free_memory()

    def get_runtime_status(self) -> Dict[str, Any]:
        with self._lock:
            active_keys = list(self.active_models.keys())
        gpu_name = None
        total_vram = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception:
            pass

        return {
            "device": self.device,
            "device_name": gpu_name,
            "total_vram_gb": total_vram,
            "mode": self.mode,
            "vram_budget_gb": self.config.runtime.vram_budget_gb,
            "quantization": self.config.runtime.quantization,
            "active_models": active_keys,
            "deterministic": self.config.runtime.deterministic,
            "artifacts": self.get_artifact_status(),
        }
