from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.runtime.manager import ModelExecutionError


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
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_type: str  # "analysed_image", "bi_temporal_pair", "opt_sar_pair"
    evidence_ptr: Optional[str] = None
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
                val = raw[name]
                expected_type = spec.get("type")
                try:
                    if expected_type == "int":
                        val = int(val)
                    elif expected_type == "float":
                        val = float(val)
                    elif expected_type == "bool":
                        val = bool(val)
                except (ValueError, TypeError):
                    raise ToolParameterError(f"Parameter '{name}' must be of type {expected_type}.")

                if "min" in spec and val < spec["min"]:
                    raise ToolParameterError(f"Parameter '{name}' below minimum allowed ({spec['min']}).")
                if "max" in spec and val > spec["max"]:
                    raise ToolParameterError(f"Parameter '{name}' exceeds maximum allowed ({spec['max']}).")
                params[name] = val
            else:
                params[name] = spec.get("default")

        return params

    def enforce_joint_use(self, images: List[Any], required_count: int = 2):
        if len(images) < required_count:
            raise ModelExecutionError(
                "MDL_JOINT_USE_VIOLATION",
                f"Tool '{self.name}' strictly mandates joint-use across {required_count} images. "
                f"Only {len(images)} image was provided."
            )

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> ToolOutput:
        """Executes the tool adapter and returns standard ToolOutput."""
        pass
