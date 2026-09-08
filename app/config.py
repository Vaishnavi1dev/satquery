from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    name: str = "SatQuery AI"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"


class RuntimeConfig(BaseModel):
    mode: str = "simulation"  # "simulation" or "real"
    device: str = "auto"
    vram_budget_gb: float = 16.0
    quantization: str = "int8"
    fallback_to_cpu: bool = True
    deterministic: bool = True
    random_seed: int = 42
    timeout_seconds: int = 60


class StorageConfig(BaseModel):
    root_dir: str = "sessions"
    max_sessions: int = 100
    max_storage_gb: float = 10.0
    artifact_cleanup_hours: int = 24


class ModelStoreConfig(BaseModel):
    weights_dir: str = "models"
    earthdial_model_path: str = "models/earthdial"
    dofa_model_path: str = "models/dofa"
    deltavlm_model_path: str = "models/deltavlm"


class ReportsConfig(BaseModel):
    output_format: str = "html"
    include_overlays: bool = True
    include_trace: bool = True


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    model_store: ModelStoreConfig = Field(default_factory=ModelStoreConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)


class SensorProfile(BaseModel):
    name: str
    modality: str
    bands: List[str]
    wavelengths_um: List[float]
    gsd_m: Any
    composites: Dict[str, List[str]]
    scaling: str = "linear"
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None


# Cached config instances
_CONFIG_CACHE: Optional[AppConfig] = None
_SENSOR_PROFILES_CACHE: Optional[Dict[str, SensorProfile]] = None


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    base_dir = get_base_dir()
    path = Path(config_path) if config_path else base_dir / "config" / "app_config.yaml"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = AppConfig(**data)
    else:
        config = AppConfig()

    # Environment variable overrides
    if "SATQUERY_MODE" in os.environ:
        config.runtime.mode = os.environ["SATQUERY_MODE"]
    if "SATQUERY_DEVICE" in os.environ:
        config.runtime.device = os.environ["SATQUERY_DEVICE"]

    _CONFIG_CACHE = config
    return config


def load_sensor_profiles(profiles_path: Optional[str] = None) -> Dict[str, SensorProfile]:
    global _SENSOR_PROFILES_CACHE
    if _SENSOR_PROFILES_CACHE is not None:
        return _SENSOR_PROFILES_CACHE

    base_dir = get_base_dir()
    path = Path(profiles_path) if profiles_path else base_dir / "config" / "sensor_profiles.yaml"

    profiles = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            sensors = raw.get("sensors", {})
            for key, val in sensors.items():
                profiles[key] = SensorProfile(**val)

    _SENSOR_PROFILES_CACHE = profiles
    return profiles
