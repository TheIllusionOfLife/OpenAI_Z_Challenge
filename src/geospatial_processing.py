"""
Geospatial processing functionality for archaeological site discovery.

This module provides classes and functions for processing geospatial data including
coordinate transformations, raster operations, terrain analysis, vegetation analysis,
and archaeological site detection.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings


@dataclass 
class SiteDetection:
    """Represents a detected archaeological site."""
    coordinates: Tuple[float, float]
    confidence: float
    features: Dict[str, Any]


@dataclass
class ClusterResult:
    """Result of site clustering."""
    representative_coordinates: Tuple[float, float]
    member_count: int
    average_confidence: float


class CoordinateTransformer:
    """Handles coordinate transformations between different CRS."""
    
    def __init__(self, source_crs: str, target_crs: str):
        """Initialize coordinate transformer."""
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    
    def transform_coordinates(self, lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinate arrays."""
        x_transformed, y_transformed = self.transformer.transform(lons, lats)
        return np.array(x_transformed), np.array(y_transformed)
    
    def transform_bounds(self, bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Transform bounding box."""
        minx, miny, maxx, maxy = bounds
        
        # Transform corner points
        corners_x = [minx, minx, maxx, maxx]
        corners_y = [miny, maxy, miny, maxy]
        
        trans_x, trans_y = self.transformer.transform(corners_x, corners_y)
        
        return (min(trans_x), min(trans_y), max(trans_x), max(trans_y))
    
    def inverse_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform coordinates back to source CRS."""
        lons, lats = self.inverse_transformer.transform(x, y)
        return np.array(lons), np.array(lats)


class RasterProcessor:
    """Handles raster data processing operations."""
    
    def __init__(self, nodata_value: Optional[float] = None, resampling_method: str = 'bilinear'):
        """Initialize raster processor."""
        self.nodata_value = nodata_value
        self.resampling_method = resampling_method
        self._resampling_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic
        }
    
    def resample_raster(self, input_data: np.ndarray, input_transform: Any, 
                       target_resolution: float) -> Tuple[np.ndarray, Any]:
        """Resample raster to different resolution."""
        # Calculate new dimensions
        pixel_size_x = abs(input_transform.a)
        pixel_size_y = abs(input_transform.e)
        
        scale_x = pixel_size_x / target_resolution
        scale_y = pixel_size_y / target_resolution
        
        new_width = int(input_data.shape[1] * scale_x)
        new_height = int(input_data.shape[0] * scale_y)
        
        # Create new transform
        new_transform = input_transform * input_transform.scale(
            input_data.shape[1] / new_width,
            input_data.shape[0] / new_height
        )
        
        # Resample using scipy for simplicity
        from scipy.ndimage import zoom
        zoom_factors = (new_height / input_data.shape[0], new_width / input_data.shape[1])
        resampled_data = zoom(input_data, zoom_factors, order=1)  # bilinear
        
        return resampled_data.astype(input_data.dtype), new_transform
    
    def align_rasters(self, rasters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Align multiple rasters to common grid."""
        if not rasters:
            return []
        
        # Find common bounds and resolution
        all_bounds = []
        all_resolutions = []
        
        for raster in rasters:
            transform = raster['transform']
            data = raster['data']
            
            # Calculate bounds
            bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
            all_bounds.append(bounds)
            
            # Get resolution
            res = abs(transform.a)
            all_resolutions.append(res)
        
        # Common bounds (intersection)
        min_bounds = [max(b[0] for b in all_bounds),  # left
                     max(b[1] for b in all_bounds),   # bottom
                     min(b[2] for b in all_bounds),   # right
                     min(b[3] for b in all_bounds)]   # top
        
        # Use finest resolution
        target_resolution = min(all_resolutions)
        
        aligned_rasters = []
        for raster in rasters:
            # For simplicity, just resample to target resolution
            resampled_data, new_transform = self.resample_raster(
                raster['data'], raster['transform'], target_resolution
            )
            
            aligned_raster = {
                'data': resampled_data,
                'transform': new_transform,
                'crs': raster['crs']
            }
            aligned_rasters.append(aligned_raster)
        
        return aligned_rasters
    
    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate raster statistics."""
        # Mask invalid values
        valid_data = data[~np.isnan(data) & ~np.isinf(data)]
        
        if len(valid_data) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data))
        }
    
    def mask_nodata_values(self, data: np.ndarray) -> np.ma.MaskedArray:
        """Mask nodata values in raster."""
        if self.nodata_value is not None:
            mask = (data == self.nodata_value) | np.isnan(data) | np.isinf(data)
        else:
            mask = np.isnan(data) | np.isinf(data)
        
        return np.ma.masked_array(data, mask=mask)


class TerrainAnalyzer:
    """Analyzes terrain features from elevation data."""
    
    def __init__(self):
        """Initialize terrain analyzer."""
        self.elevation_data = None
    
    def calculate_slope(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate slope from elevation data."""
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation, pixel_size)
        
        # Calculate slope in radians, then convert to degrees
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def calculate_aspect(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate aspect from elevation data."""
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation, pixel_size)
        
        # Calculate aspect in radians, then convert to degrees
        aspect_rad = np.arctan2(-grad_x, grad_y)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to 0-360 range
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
        
        return aspect_deg
    
    def calculate_curvature(self, elevation: np.ndarray, pixel_size: float) -> Dict[str, np.ndarray]:
        """Calculate curvature from elevation data."""
        # Calculate first derivatives
        grad_y, grad_x = np.gradient(elevation, pixel_size)
        
        # Calculate second derivatives
        grad_xx = np.gradient(grad_x, pixel_size, axis=1)
        grad_yy = np.gradient(grad_y, pixel_size, axis=0) 
        grad_xy = np.gradient(grad_x, pixel_size, axis=0)
        
        # Profile curvature (curvature in the direction of slope)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.where(grad_mag == 0, 1e-8, grad_mag)  # Avoid division by zero
        
        profile_curvature = (grad_xx * grad_x**2 + 2 * grad_xy * grad_x * grad_y + grad_yy * grad_y**2) / grad_mag**3
        
        # Plan curvature (curvature perpendicular to slope direction)
        plan_curvature = (grad_xx * grad_y**2 - 2 * grad_xy * grad_x * grad_y + grad_yy * grad_x**2) / grad_mag**2
        
        return {
            'profile_curvature': profile_curvature,
            'plan_curvature': plan_curvature
        }
    
    def identify_terrain_features(self, elevation: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """Identify terrain features like peaks, valleys, ridges."""
        # Simple feature detection using local maxima/minima
        from scipy.ndimage import maximum_filter, minimum_filter
        
        # Local maxima (peaks)
        local_maxima = maximum_filter(elevation, size=3)
        peaks_mask = (elevation == local_maxima) & (elevation > np.percentile(elevation, 95))
        peaks = list(zip(*np.where(peaks_mask)))
        
        # Local minima (valleys)
        local_minima = minimum_filter(elevation, size=3)
        valleys_mask = (elevation == local_minima) & (elevation < np.percentile(elevation, 5))
        valleys = list(zip(*np.where(valleys_mask)))
        
        # Ridges (simplified - high curvature areas)
        curvature = self.calculate_curvature(elevation, 1.0)
        ridge_mask = curvature['profile_curvature'] > np.percentile(curvature['profile_curvature'], 90)
        ridges = list(zip(*np.where(ridge_mask)))
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'ridges': ridges
        }


class VegetationAnalyzer:
    """Analyzes vegetation patterns from NDVI and other indices."""
    
    def __init__(self):
        """Initialize vegetation analyzer."""
        self.ndvi_data = None
        self.threshold_values = {
            'bare_soil': 0.1,
            'sparse': 0.3,
            'moderate': 0.5,
            'dense': 0.7
        }
    
    def classify_vegetation_density(self, ndvi_data: np.ndarray) -> np.ndarray:
        """Classify vegetation density from NDVI."""
        classes = np.zeros_like(ndvi_data, dtype=int)
        
        # Classify based on thresholds
        classes = np.where(ndvi_data < self.threshold_values['bare_soil'], 0, classes)  # Bare soil
        classes = np.where((ndvi_data >= self.threshold_values['bare_soil']) & 
                          (ndvi_data < self.threshold_values['sparse']), 1, classes)  # Sparse
        classes = np.where((ndvi_data >= self.threshold_values['sparse']) & 
                          (ndvi_data < self.threshold_values['moderate']), 2, classes)  # Moderate
        classes = np.where((ndvi_data >= self.threshold_values['moderate']) & 
                          (ndvi_data < self.threshold_values['dense']), 3, classes)  # Dense
        classes = np.where(ndvi_data >= self.threshold_values['dense'], 4, classes)  # Very dense
        
        return classes
    
    def detect_vegetation_anomalies(self, ndvi_data: np.ndarray, window_size: int = 3, 
                                   threshold: float = 0.5) -> np.ndarray:
        """Detect vegetation anomalies."""
        # Calculate local mean
        local_mean = ndimage.uniform_filter(ndvi_data, size=window_size)
        
        # Calculate deviation from local mean
        deviation = np.abs(ndvi_data - local_mean)
        
        # Identify anomalies
        anomalies = deviation > threshold
        
        return anomalies
    
    def calculate_vegetation_indices(self, red_band: np.ndarray, nir_band: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate additional vegetation indices."""
        # Ensure float32 for calculations
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # NDVI
        ndvi_num = nir - red
        ndvi_den = nir + red
        ndvi = np.divide(ndvi_num, ndvi_den, out=np.zeros_like(ndvi_num), where=ndvi_den!=0)
        
        # SAVI (Soil Adjusted Vegetation Index)
        L = 0.5  # Soil brightness correction factor
        savi_num = (nir - red) * (1 + L)
        savi_den = nir + red + L
        savi = np.divide(savi_num, savi_den, out=np.zeros_like(savi_num), where=savi_den!=0)
        
        # EVI (Enhanced Vegetation Index) - simplified without blue band
        evi_num = 2.5 * (nir - red)
        evi_den = nir + 6 * red + 1
        evi = np.divide(evi_num, evi_den, out=np.zeros_like(evi_num), where=evi_den!=0)
        
        return {
            'ndvi': ndvi,
            'savi': savi,
            'evi': evi
        }


class SpatialFeatureExtractor:
    """Extracts spatial features for machine learning."""
    
    def __init__(self):
        """Initialize spatial feature extractor."""
        self.feature_types = ['textural', 'morphological', 'distance']
    
    def extract_textural_features(self, image: np.ndarray, window_size: int = 7) -> Dict[str, np.ndarray]:
        """Extract textural features using GLCM (simplified)."""
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Calculate GLCM for different directions
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Initialize output arrays
        contrast = np.zeros(image.shape, dtype=np.float32)
        dissimilarity = np.zeros(image.shape, dtype=np.float32)
        homogeneity = np.zeros(image.shape, dtype=np.float32)
        energy = np.zeros(image.shape, dtype=np.float32)
        
        # Calculate features in sliding windows
        half_window = window_size // 2
        
        for i in range(half_window, image.shape[0] - half_window):
            for j in range(half_window, image.shape[1] - half_window):
                window = image_uint8[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                
                try:
                    glcm = graycomatrix(window, distances, angles, levels=256, symmetric=True, normed=True)
                    
                    contrast[i, j] = np.mean(graycoprops(glcm, 'contrast'))
                    dissimilarity[i, j] = np.mean(graycoprops(glcm, 'dissimilarity'))
                    homogeneity[i, j] = np.mean(graycoprops(glcm, 'homogeneity'))
                    energy[i, j] = np.mean(graycoprops(glcm, 'energy'))
                except:
                    # Handle edge cases
                    continue
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy
        }
    
    def calculate_distance_features(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                                   reference_points: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """Calculate distance-based features."""
        # Flatten coordinate arrays for distance calculation
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        ref_points = np.array(reference_points)
        
        # Calculate distances to all reference points
        distances = cdist(coords, ref_points)
        
        # Calculate features
        min_distance = np.min(distances, axis=1).reshape(x_coords.shape)
        mean_distance = np.mean(distances, axis=1).reshape(x_coords.shape)
        
        return {
            'min_distance': min_distance,
            'mean_distance': mean_distance
        }
    
    def extract_morphological_features(self, binary_image: np.ndarray) -> Dict[str, float]:
        """Extract morphological features from binary image."""
        # Label connected components
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image)
        
        if not regions:
            return {'area': 0, 'perimeter': 0, 'compactness': 0, 'eccentricity': 0}
        
        # Calculate features for largest region
        largest_region = max(regions, key=lambda r: r.area)
        
        area = largest_region.area
        perimeter = largest_region.perimeter
        compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        eccentricity = largest_region.eccentricity
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'eccentricity': float(eccentricity)
        }


class ArchaeologicalSiteDetector:
    """Detects potential archaeological sites."""
    
    def __init__(self):
        """Initialize archaeological site detector."""
        self.detection_parameters = {
            'min_confidence': 0.5,
            'cluster_distance': 100,  # meters
            'min_cluster_size': 2
        }
        self.trained_model = None
    
    def identify_potential_sites(self, features: np.ndarray, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify potential archaeological sites."""
        # Simplified detection - in reality would use trained ML model
        potential_sites = []
        
        # For demonstration, use simple threshold on feature variance
        feature_variance = np.var(features, axis=2)
        high_variance_mask = feature_variance > np.percentile(feature_variance, 95)
        
        # Find coordinates of potential sites
        y_coords, x_coords = np.where(high_variance_mask)
        
        for i, (y, x) in enumerate(zip(y_coords, x_coords)):
            # Calculate a mock confidence score
            confidence = min(1.0, feature_variance[y, x] / np.max(feature_variance))
            
            if confidence >= confidence_threshold:
                site = {
                    'coordinates': (x, y),
                    'confidence': float(confidence),
                    'features': {
                        'feature_variance': float(feature_variance[y, x]),
                        'neighborhood_features': features[y, x, :].tolist()
                    }
                }
                potential_sites.append(site)
        
        return potential_sites
    
    def validate_site_characteristics(self, site_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate site characteristics against known patterns."""
        # Simple validation rules
        elevation = site_data.get('elevation', 0)
        slope = site_data.get('slope', 0)
        distance_to_water = site_data.get('distance_to_water', float('inf'))
        
        # Archaeological site preferences (simplified)
        elevation_valid = 50 <= elevation <= 500  # Moderate elevation
        slope_valid = slope <= 15  # Not too steep
        water_proximity_valid = distance_to_water <= 1000  # Within 1km of water
        
        # Calculate confidence score
        validation_checks = [elevation_valid, slope_valid, water_proximity_valid]
        confidence_score = sum(validation_checks) / len(validation_checks)
        
        is_valid = confidence_score >= 0.6
        
        reasons = []
        if not elevation_valid:
            reasons.append("Elevation outside typical range")
        if not slope_valid:
            reasons.append("Slope too steep")
        if not water_proximity_valid:
            reasons.append("Too far from water sources")
        
        return {
            'is_valid': is_valid,
            'confidence_score': confidence_score,
            'reasons': reasons
        }
    
    def cluster_nearby_detections(self, detections: List[Dict[str, Any]], max_distance: float = 10) -> List[ClusterResult]:
        """Cluster nearby site detections."""
        if not detections:
            return []
        
        # Extract coordinates
        coords = np.array([d['coordinates'] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])
        
        # Perform clustering
        clustering = DBSCAN(eps=max_distance, min_samples=1).fit(coords)
        labels = clustering.labels_
        
        # Create cluster results
        clustered_sites = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_confidences = confidences[cluster_mask]
            
            # Calculate representative coordinates (centroid)
            representative_coords = tuple(np.mean(cluster_coords, axis=0))
            
            cluster_result = ClusterResult(
                representative_coordinates=representative_coords,
                member_count=int(np.sum(cluster_mask)),
                average_confidence=float(np.mean(cluster_confidences))
            )
            clustered_sites.append(cluster_result)
        
        return clustered_sites
    
    def generate_site_report(self, site: Dict[str, Any]) -> Dict[str, Any]:
        """Generate site discovery report."""
        coordinates = site['coordinates']
        confidence = site['confidence']
        features = site.get('features', {})
        
        # Generate site ID
        site_id = f"SITE_{int(abs(coordinates[0]))}_{int(abs(coordinates[1]))}"
        
        # Generate description
        terrain_features = features.get('terrain_features', [])
        feature_desc = ", ".join(terrain_features) if terrain_features else "Unknown features"
        
        description = f"Potential archaeological site identified with {feature_desc}. " \
                     f"Location shows anomalous patterns in geospatial data analysis."
        
        # Confidence assessment
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Recommended validation steps
        validation_steps = [
            "High-resolution satellite imagery review",
            "LiDAR data verification",
            "Archaeological literature cross-reference"
        ]
        
        if confidence >= 0.8:
            validation_steps.append("Field survey recommendation")
        
        return {
            'site_id': site_id,
            'coordinates': coordinates,
            'description': description,
            'confidence_assessment': {
                'score': confidence,
                'level': confidence_level
            },
            'recommended_validation': validation_steps
        }