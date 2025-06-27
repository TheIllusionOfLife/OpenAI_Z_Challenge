"""
Data loading functionality for archaeological site discovery.

This module provides classes and functions for loading various types of geospatial
and archaeological data including LiDAR, satellite imagery, NDVI, GIS data,
and archaeological literature.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    message: str
    confidence_score: float = 0.0


@dataclass
class QualityAssessment:
    """Data quality assessment result."""

    has_missing_values: bool
    has_infinite_values: bool
    completeness_score: float
    quality_score: float


class LiDARLoader:
    """Loader for LiDAR point cloud data."""

    def __init__(self, file_path: str):
        """Initialize LiDAR loader with file path."""
        self.file_path = file_path
        self.data = None
        self.metadata = None

    def load_data(self) -> np.ndarray:
        """Load LiDAR data from file."""
        with rasterio.open(self.file_path) as src:
            self.data = src.read(1).astype(np.float32)
            self.metadata = src.meta
        return self.data

    def extract_terrain_features(
        self, elevation_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract terrain features from elevation data."""
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation_data)

        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope = np.degrees(slope_rad)

        # Calculate aspect in degrees
        aspect = np.degrees(np.arctan2(-grad_x, grad_y))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # Calculate curvature (simplified)
        grad_xx = np.gradient(grad_x, axis=1)
        grad_yy = np.gradient(grad_y, axis=0)
        curvature = grad_xx + grad_yy

        return {"slope": slope, "aspect": aspect, "curvature": curvature}

    def validate_data_format(self, data: np.ndarray, metadata: Dict) -> bool:
        """Validate LiDAR data format and metadata."""
        if data.ndim != 2:
            return False
        if data.size == 0:
            return False
        if not isinstance(metadata, dict):
            return False
        if "crs" not in metadata:
            return False
        return True


class SatelliteImageLoader:
    """Loader for satellite imagery data."""

    def __init__(self, file_path: str):
        """Initialize satellite image loader."""
        self.file_path = file_path
        self.bands = None
        self.metadata = None

    def load_data(self) -> np.ndarray:
        """Load multispectral satellite data."""
        with rasterio.open(self.file_path) as src:
            self.bands = src.read()
            self.metadata = src.meta
        return self.bands

    def calculate_vegetation_indices(
        self, red_band: np.ndarray, nir_band: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate vegetation indices from satellite bands."""
        # Ensure float32 for calculations
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)

        # NDVI calculation
        numerator = nir - red
        denominator = nir + red
        ndvi = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator != 0,
        )

        # EVI calculation (Enhanced Vegetation Index)
        # EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
        # Simplified without blue band
        evi = 2.5 * np.divide(
            numerator,
            nir + 6 * red + 1,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=(nir + 6 * red + 1) != 0,
        )

        return {"ndvi": ndvi, "evi": evi}


class NDVILoader:
    """Loader for NDVI data."""

    def __init__(self, file_path: str):
        """Initialize NDVI loader."""
        self.file_path = file_path
        self.data = None
        self.metadata = None

    def load_data(self) -> np.ndarray:
        """Load NDVI data and validate range."""
        with rasterio.open(self.file_path) as src:
            self.data = src.read(1).astype(np.float32)
            self.metadata = src.meta
        return self.data

    def detect_anomalies(
        self, ndvi_data: np.ndarray, threshold: float = 0.3
    ) -> np.ndarray:
        """Detect vegetation anomalies in NDVI data."""
        # Simple anomaly detection based on local variation
        from scipy import ndimage

        # Calculate local mean with a sliding window
        kernel_size = 3
        local_mean = ndimage.uniform_filter(ndvi_data, size=kernel_size)

        # Calculate absolute difference from local mean
        anomaly_score = np.abs(ndvi_data - local_mean)

        # Threshold for anomaly detection
        anomalies = anomaly_score > threshold

        return anomalies


class GISDataLoader:
    """Loader for GIS vector data."""

    def __init__(self, file_path: str):
        """Initialize GIS data loader."""
        self.file_path = file_path
        self.data = None

    def load_data(self) -> gpd.GeoDataFrame:
        """Load vector GIS data."""
        self.data = gpd.read_file(self.file_path)
        return self.data

    def reproject(self, gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
        """Reproject GIS data to target CRS."""
        return gdf.to_crs(target_crs)


class ArchaeologicalLiteratureLoader:
    """Loader for archaeological literature data."""

    def __init__(self, file_path: str):
        """Initialize literature loader."""
        self.file_path = file_path
        self.data = None

    def load_data(self) -> Dict[str, Any]:
        """Load archaeological literature from JSON."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def extract_coordinates(self, literature_data: Dict[str, Any]) -> List[List[float]]:
        """Extract site coordinates from literature."""
        coordinates = []
        if "sites" in literature_data:
            for site in literature_data["sites"]:
                if "coordinates" in site:
                    coordinates.append(site["coordinates"])
        return coordinates

    def validate_doi_format(self, doi: str) -> bool:
        """Validate DOI format."""
        # Basic DOI format validation
        doi_pattern = r"^10\.\d{4,}/.+$"
        return bool(re.match(doi_pattern, doi))


class DatasetValidator:
    """Validator for dataset consistency and quality."""

    def __init__(self):
        """Initialize dataset validator."""
        self.validation_rules = {
            "crs_consistency": True,
            "spatial_overlap": True,
            "data_quality": True,
        }

    def validate_crs_consistency(
        self, datasets: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate CRS consistency across datasets."""
        if not datasets:
            return ValidationResult(False, "No datasets provided")

        reference_crs = datasets[0].get("crs")
        for dataset in datasets[1:]:
            if dataset.get("crs") != reference_crs:
                return ValidationResult(
                    False, f"CRS mismatch: {reference_crs} vs {dataset.get('crs')}"
                )

        return ValidationResult(True, "CRS consistent across all datasets")

    def validate_spatial_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Validate spatial extent overlap between two bounding boxes."""
        # bbox format: [minx, miny, maxx, maxy]
        overlap_x = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        overlap_y = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

        return overlap_x > 0 and overlap_y > 0

    def assess_data_quality(self, data: np.ndarray) -> QualityAssessment:
        """Assess data quality for missing/invalid values."""
        # Check for missing values (NaN)
        has_missing = np.isnan(data).any()
        missing_ratio = np.isnan(data).sum() / data.size

        # Check for infinite values
        has_infinite = np.isinf(data).any()
        infinite_ratio = np.isinf(data).sum() / data.size

        # Calculate completeness score
        completeness_score = 1.0 - missing_ratio - infinite_ratio

        # Overall quality score
        quality_score = completeness_score * 0.7 + (0.3 if not has_infinite else 0.0)

        return QualityAssessment(
            has_missing_values=has_missing,
            has_infinite_values=has_infinite,
            completeness_score=completeness_score,
            quality_score=quality_score,
        )
