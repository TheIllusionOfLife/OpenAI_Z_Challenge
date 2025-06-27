"""
Configuration management for the archaeological site discovery system.

This module handles all configuration settings, environment variables,
and project-specific parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseSettings, validator
import json
import yaml
from dotenv import load_dotenv


# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class DataPaths:
    """Configuration for data paths."""

    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.outputs_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class GeospatialConfig:
    """Configuration for geospatial processing."""

    default_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:3857"  # Web Mercator for calculations
    default_pixel_size: float = 30.0  # meters
    nodata_value: float = -9999.0
    resampling_method: str = "bilinear"

    # Amazon region bounds (approximate)
    amazon_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "min_lat": -10.0,
            "max_lat": 5.0,
            "min_lon": -75.0,
            "max_lon": -45.0,
        }
    )


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = "gpt-4.1"
    max_retries: int = 3
    timeout: int = 60
    max_tokens: int = 4000
    temperature: float = 0.1  # Low temperature for consistent results

    # Model preferences for different tasks
    model_preferences: Dict[str, str] = field(
        default_factory=lambda: {
            "coordinate_extraction": "gpt-4.1",
            "text_analysis": "gpt-4.1",
            "simple_extraction": "o3-mini",
            "complex_analysis": "gpt-4.1",
            "report_generation": "gpt-4.1",
            "validation": "o4-mini",
        }
    )


@dataclass
class ArchaeologyConfig:
    """Configuration for archaeological analysis."""

    # Site detection parameters
    min_confidence_threshold: float = 0.7
    cluster_distance_threshold: float = 100.0  # meters
    min_cluster_size: int = 2

    # Terrain analysis parameters
    slope_threshold: float = 15.0  # degrees
    elevation_range: Dict[str, float] = field(
        default_factory=lambda: {"min": 50.0, "max": 500.0}
    )

    # Vegetation analysis parameters
    ndvi_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "bare_soil": 0.1,
            "sparse": 0.3,
            "moderate": 0.5,
            "dense": 0.7,
        }
    )

    # Known archaeological indicators
    site_indicators: List[str] = field(
        default_factory=lambda: [
            "platform",
            "plaza",
            "mound",
            "earthwork",
            "terrace",
            "canal",
            "causeway",
            "circular_feature",
            "linear_feature",
        ]
    )


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    max_file_size: int = 10_000_000  # 10MB
    backup_count: int = 5


class Settings(BaseSettings):
    """Main settings class using Pydantic for validation."""

    # Environment
    environment: str = "development"
    debug: bool = False

    # Project info
    project_name: str = "OpenAI to Z Challenge"
    version: str = "0.1.0"

    # API Keys and secrets
    openai_api_key: Optional[str] = None

    # Database (if needed in future)
    database_url: Optional[str] = None

    # Processing settings
    max_workers: int = 4
    batch_size: int = 10

    # Competition specific
    competition_name: str = "OpenAI to Z Challenge"
    submission_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("openai_api_key")
    def validate_openai_key(cls, v):
        if not v:
            # Try to get from environment
            v = os.getenv("OPENAI_API_KEY")
        return v


class ConfigManager:
    """Manages all configuration for the project."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_file = config_file or Path("config.yaml")
        self.settings = Settings()
        self.data_paths = DataPaths()
        self.geospatial = GeospatialConfig()
        self.openai = OpenAIConfig()
        self.archaeology = ArchaeologyConfig()
        self.logging = LoggingConfig()

        # Load custom config if file exists
        if self.config_file.exists():
            self.load_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                if self.config_file.suffix.lower() == ".yaml":
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Update configurations with loaded data
            self._update_config_from_dict(config_data)

        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")

    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            "settings": self._dataclass_to_dict(self.settings),
            "data_paths": self._dataclass_to_dict(self.data_paths),
            "geospatial": self._dataclass_to_dict(self.geospatial),
            "openai": self._dataclass_to_dict(self.openai),
            "archaeology": self._dataclass_to_dict(self.archaeology),
            "logging": self._dataclass_to_dict(self.logging),
        }

        try:
            with open(self.config_file, "w") as f:
                if self.config_file.suffix.lower() == ".yaml":
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save config file {self.config_file}: {e}")

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration objects from dictionary."""
        if "geospatial" in config_data:
            self._update_dataclass(self.geospatial, config_data["geospatial"])
        if "openai" in config_data:
            self._update_dataclass(self.openai, config_data["openai"])
        if "archaeology" in config_data:
            self._update_dataclass(self.archaeology, config_data["archaeology"])
        if "logging" in config_data:
            self._update_dataclass(self.logging, config_data["logging"])

    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass object with dictionary data."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, Path):
                    result[key] = str(value)
                elif isinstance(value, dict):
                    result[key] = value
                elif isinstance(value, (list, tuple)):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result
        return {}

    def get_data_path(self, data_type: str) -> Path:
        """Get path for specific data type."""
        path_mapping = {
            "raw": self.data_paths.raw_data_dir,
            "processed": self.data_paths.processed_data_dir,
            "models": self.data_paths.models_dir,
            "outputs": self.data_paths.outputs_dir,
            "logs": self.data_paths.logs_dir,
        }
        return path_mapping.get(data_type, self.data_paths.raw_data_dir)

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration as dictionary."""
        return {
            "api_key": self.openai.api_key,
            "default_model": self.openai.default_model,
            "max_retries": self.openai.max_retries,
            "timeout": self.openai.timeout,
            "max_tokens": self.openai.max_tokens,
            "temperature": self.openai.temperature,
            "model_preferences": self.openai.model_preferences,
        }

    def get_geospatial_config(self) -> Dict[str, Any]:
        """Get geospatial configuration as dictionary."""
        return {
            "default_crs": self.geospatial.default_crs,
            "target_crs": self.geospatial.target_crs,
            "default_pixel_size": self.geospatial.default_pixel_size,
            "nodata_value": self.geospatial.nodata_value,
            "resampling_method": self.geospatial.resampling_method,
            "amazon_bounds": self.geospatial.amazon_bounds,
        }

    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues."""
        issues = {"errors": [], "warnings": []}

        # Check OpenAI API key
        if not self.openai.api_key:
            issues["errors"].append("OpenAI API key is not set")

        # Check data directories
        for path_name, path in [
            ("raw_data", self.data_paths.raw_data_dir),
            ("processed_data", self.data_paths.processed_data_dir),
            ("models", self.data_paths.models_dir),
            ("outputs", self.data_paths.outputs_dir),
        ]:
            if not path.exists():
                issues["warnings"].append(
                    f"{path_name} directory does not exist: {path}"
                )

        # Check geospatial bounds
        bounds = self.geospatial.amazon_bounds
        if bounds["min_lat"] >= bounds["max_lat"]:
            issues["errors"].append("Invalid latitude bounds in geospatial config")
        if bounds["min_lon"] >= bounds["max_lon"]:
            issues["errors"].append("Invalid longitude bounds in geospatial config")

        # Check archaeology thresholds
        thresholds = self.archaeology.ndvi_thresholds
        threshold_values = [
            thresholds["bare_soil"],
            thresholds["sparse"],
            thresholds["moderate"],
            thresholds["dense"],
        ]
        if threshold_values != sorted(threshold_values):
            issues["errors"].append("NDVI thresholds are not in ascending order")

        return issues

    def create_example_config(self) -> str:
        """Create an example configuration file content."""
        example_config = {
            "geospatial": {
                "default_crs": "EPSG:4326",
                "target_crs": "EPSG:3857",
                "default_pixel_size": 30.0,
                "amazon_bounds": {
                    "min_lat": -10.0,
                    "max_lat": 5.0,
                    "min_lon": -75.0,
                    "max_lon": -45.0,
                },
            },
            "openai": {
                "default_model": "gpt-4.1",
                "max_retries": 3,
                "timeout": 60,
                "temperature": 0.1,
            },
            "archaeology": {
                "min_confidence_threshold": 0.7,
                "cluster_distance_threshold": 100.0,
                "slope_threshold": 15.0,
            },
            "logging": {"level": "INFO", "file_handler": True, "console_handler": True},
        }

        return yaml.dump(example_config, default_flow_style=False)


# Global configuration instance
config = ConfigManager()
