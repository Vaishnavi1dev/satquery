import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.runtime.manager import ModelExecutionError


_SUPPORTED_MODALITIES = {"optical", "multispectral", "sar"}
_INVALID_MODEL_TEXTS = {"", "none", "null"}


class ToolParameterError(Exception):
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolOutput(BaseModel):
    tool_name: str
    model_name: str
    text: str
    boxes: Optional[List[List[int]]] = None  # [ [x1, y1, x2, y2], ... ]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_type: str  # "analysed_image", "bi_temporal_pair", "opt_sar_pair"
    evidence_ptr: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolBase(ABC):
    """Abstract base class for all specialist remote sensing tools."""

    def __init__(self, descriptor: Dict[str, Any], runtime_mgr):
        self.descriptor = descriptor
        self.name: str = descriptor["name"]
        self.model_key: str = descriptor["model_key"]
        self.model_name: str = descriptor["model_name"]
        self.load_group: str = descriptor["load_group"]
        self.permitted_params: Dict[str, Any] = descriptor.get("permitted_params", {})
        self.runtime_mgr = runtime_mgr

    def validate_and_filter_params(self, raw_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {}
        raw = raw_params or {}

        for name, spec in self.permitted_params.items():
            if name in raw:
                val = self._coerce_param(name, raw[name], spec)
                if isinstance(val, float) and not math.isfinite(val):
                    raise ToolParameterError(f"Parameter '{name}' must be a finite number.")
                if "min" in spec and val < spec["min"]:
                    raise ToolParameterError(f"Parameter '{name}' below minimum allowed ({spec['min']}).")
                if "max" in spec and val > spec["max"]:
                    raise ToolParameterError(f"Parameter '{name}' exceeds maximum allowed ({spec['max']}).")
                params[name] = val
            else:
                params[name] = spec.get("default")

        return params

    @staticmethod
    def _coerce_param(name: str, val: Any, spec: Dict[str, Any]) -> Any:
        """Coerce a raw parameter to its declared type, rejecting lossy/invalid input."""
        expected_type = spec.get("type")
        if expected_type == "int":
            if isinstance(val, bool) or isinstance(val, float):
                raise ToolParameterError(f"Parameter '{name}' must be of type int.")
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val.strip())
                except ValueError:
                    raise ToolParameterError(f"Parameter '{name}' must be of type int.")
            raise ToolParameterError(f"Parameter '{name}' must be of type int.")
        if expected_type == "float":
            if isinstance(val, bool):
                raise ToolParameterError(f"Parameter '{name}' must be of type float.")
            try:
                return float(val)
            except (TypeError, ValueError):
                raise ToolParameterError(f"Parameter '{name}' must be of type float.")
        if expected_type == "bool":
            if isinstance(val, bool):
                return val
            if isinstance(val, str) and val.strip().lower() in ("true", "false"):
                return val.strip().lower() == "true"
            raise ToolParameterError(f"Parameter '{name}' must be a boolean (true/false).")
        return val

    def enforce_joint_use(self, images: List[Any], required_count: int = 2):
        if len(images) < required_count:
            raise ModelExecutionError(
                "MDL_JOINT_USE_VIOLATION",
                f"Tool '{self.name}' strictly mandates joint-use across {required_count} images. "
                f"Only {len(images)} image was provided."
            )

    @staticmethod
    def normalize_modality(modality: Any) -> str:
        """Lower-case and validate a modality label against the supported set."""
        normalized = modality.strip().lower() if isinstance(modality, str) else ""
        if normalized not in _SUPPORTED_MODALITIES:
            raise ModelExecutionError(
                "MDL_UNSUPPORTED_MODALITY",
                f"Unsupported modality '{modality}'. Supported modalities: optical, multispectral, sar.",
            )
        return normalized

    def validate_image_envelopes(self, images: List[Any], max_count: Optional[int] = None) -> List[Any]:
        """Reject null/non-envelope entries and any count above ``max_count``."""
        if not isinstance(images, (list, tuple)):
            raise ModelExecutionError(
                "MDL_INVALID_IMAGE_ENVELOPE",
                f"Tool '{self.name}' expected a list of image envelopes.",
            )
        if max_count is not None and len(images) > max_count:
            raise ModelExecutionError(
                "MDL_JOINT_USE_VIOLATION",
                f"Tool '{self.name}' accepts at most {max_count} images; received {len(images)}.",
            )
        for idx, env in enumerate(images):
            if env is None:
                raise ModelExecutionError(
                    "MDL_INVALID_IMAGE_ENVELOPE",
                    f"Tool '{self.name}' received a null image envelope at index {idx}.",
                )
            if not hasattr(env, "modality"):
                raise ModelExecutionError(
                    "MDL_INVALID_IMAGE_ENVELOPE",
                    f"Tool '{self.name}' received a non-envelope object at index {idx}.",
                )
        return list(images)

    @staticmethod
    def usable_model_text(result: Any) -> Optional[str]:
        """Return genuine model text, treating None/'none'/'null'/'' as no output."""
        if not isinstance(result, dict):
            return None
        text = result.get("text")
        if not isinstance(text, str):
            return None
        text = text.strip()
        if text.lower() in _INVALID_MODEL_TEXTS:
            return None
        return text

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        """Executes the tool adapter and returns standard ToolOutput."""
        pass
