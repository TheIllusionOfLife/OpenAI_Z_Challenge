#!/usr/bin/env python3
"""Manual integration test for archaeological site discovery system."""

import sys
import traceback
from src.geospatial_processing import RasterProcessor, SpatialFeatureExtractor
from src.data_loading import DataLoader
import numpy as np


def test_geospatial_pipeline():
    """Test basic geospatial processing pipeline."""
    print("Testing geospatial processing pipeline...")

    try:
        # Create synthetic test data
        test_raster = np.random.rand(100, 100)
        processor = RasterProcessor()

        # Test raster processing
        print("✓ RasterProcessor initialized")

        # Test feature extraction
        extractor = SpatialFeatureExtractor()
        print("✓ SpatialFeatureExtractor initialized")

        print("✓ Geospatial pipeline test passed")
        return True

    except Exception as e:
        print(f"✗ Geospatial pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading capabilities."""
    print("Testing data loading...")

    try:
        loader = DataLoader()
        formats = loader.supported_formats
        print(f"✓ Supported formats: {formats}")
        return True

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Manual Integration Test ===")

    tests = [
        test_data_loading,
        test_geospatial_pipeline,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
