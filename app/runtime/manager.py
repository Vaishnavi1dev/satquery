import logging
import os
import time
from typing import Dict, Any, Optional, List
from app.config import load_app_config

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
    and execution between Model A (EarthDial), Model B (DOFA), and Model C (DeltaVLM).
    """

    def __init__(self):
        self.config = load_app_config()
        self.device = self._resolve_device()
        self.active_models: Dict[str, Any] = {}
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
        """Manages lazy loading and eviction between llm_primary and llm_secondary."""
        if model_key in self.active_models:
            return

        # Evict conflicting load group if memory is constrained
        if load_group == "llm_secondary" and "llm_primary" in self.active_models:
            logger.info("Evicting llm_primary to accommodate llm_secondary under VRAM budget.")
            self.unload_group("llm_primary")
        elif load_group == "llm_primary" and "llm_secondary" in self.active_models:
            logger.info("Evicting llm_secondary to accommodate llm_primary under VRAM budget.")
            self.unload_group("llm_secondary")

        logger.info(f"Loading {model_key} (Group: {load_group}, Device: {self.device}, Mode: {self.mode})...")
        self.active_models[model_key] = {
            "loaded_at": time.time(),
            "load_group": load_group,
            "device": self.device
        }

    def unload_group(self, load_group: str):
        to_remove = [k for k, v in self.active_models.items() if v.get("load_group") == load_group]
        for k in to_remove:
            del self.active_models[k]
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
            "deterministic": self.config.runtime.deterministic
        }
