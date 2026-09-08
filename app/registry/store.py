import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.config import get_base_dir
from app.runtime.manager import ModelRuntimeManager
from app.tools.base import ToolBase
from app.tools.rs_vqa import SingleImageVQATool
from app.tools.rs_caption import SingleImageCaptionTool
from app.tools.rs_ground import TextGuidedGroundingTool
from app.tools.change_vqa import BiTemporalChangeVQATool
from app.tools.opt_sar_fusion import OpticalSARFusionTool


class ToolRegistryStore:
    """
    Central tool registry reading config/registry.json.
    Enforces registry-only invocation with declarative, versioned descriptors.
    """

    TOOL_CLASS_MAP = {
        "rs-vqa": SingleImageVQATool,
        "rs-caption": SingleImageCaptionTool,
        "rs-ground": TextGuidedGroundingTool,
        "change-vqa": BiTemporalChangeVQATool,
        "opt-sar-fusion": OpticalSARFusionTool,
    }

    def __init__(self, registry_path: Optional[Path] = None, runtime_mgr: Optional[ModelRuntimeManager] = None):
        base_dir = get_base_dir()
        self.registry_path = registry_path or (base_dir / "config" / "registry.json")
        self.runtime_mgr = runtime_mgr or ModelRuntimeManager()
        self.descriptors: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, ToolBase] = {}
        self.version: str = "1.0.0"
        self._load_registry()

    def _load_registry(self):
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry config not found at: {self.registry_path}")

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("registry_version", "1.0.0")
        for desc in data.get("tools", []):
            name = desc["name"]
            self.descriptors[name] = desc

            tool_cls = self.TOOL_CLASS_MAP.get(name)
            if tool_cls:
                self.tools[name] = tool_cls(desc, self.runtime_mgr)

    def list_descriptors(self) -> List[Dict[str, Any]]:
        return list(self.descriptors.values())

    def get_descriptor(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.descriptors.get(tool_name)

    def get_tool(self, tool_name: str) -> Optional[ToolBase]:
        return self.tools.get(tool_name)

    def find_tool_for_task(self, task: str, input_count: int = 1) -> Optional[ToolBase]:
        """Finds the enabled specialist tool matching task label and input count."""
        for desc in self.descriptors.values():
            if not desc.get("enabled", True):
                continue
            if desc.get("task") == task and desc.get("input_count") == input_count:
                return self.tools.get(desc["name"])
        return None
