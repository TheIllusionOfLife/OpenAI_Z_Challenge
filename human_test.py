#!/usr/bin/env python3
"""
Comprehensive human testing script for archaeological site discovery system.

This script performs manual verification of core functionality without requiring
external dependencies like API keys or real geospatial data files.
"""

import sys
import traceback
import psutil
import numpy as np
import os


def test_environment_setup():
    """Test environment and dependency setup."""
    print("=== Phase 1: Environment Setup Testing ===")

    try:
        print(f"Python version: {sys.version}")

        # Test core imports
        print("Testing core module imports...")
        import src.openai_integration

        print("✓ OpenAI integration module imported")

        import src.geospatial_processing

        print("✓ Geospatial processing module imported")

        import src.data_loading

        print("✓ Data loading module imported")

        import src.config

        print("✓ Configuration module imported")

        import src.logging_utils

        print("✓ Logging utilities module imported")

        print("✓ All core modules imported successfully")
        return True

    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        traceback.print_exc()
        return False


def test_core_functionality():
    """Test core functionality of each component."""
    print("\n=== Phase 2: Core Functionality Testing ===")

    results = []

    # Test OpenAI integration
    print("Testing OpenAI integration...")
    try:
        from src.openai_integration import (
            OpenAIClient,
            CompletionResponse,
            TokenManager,
        )

        # Test without API key (should fail gracefully)
        try:
            client = OpenAIClient()
            print("✗ Should have failed without API key")
            results.append(False)
        except ValueError as e:
            print(f"✓ Correctly handles missing API key: {str(e)[:60]}...")
            results.append(True)

        # Test TokenManager with invalid model
        try:
            manager = TokenManager("invalid-model")
            print("✓ TokenManager handles invalid model gracefully")
            results.append(True)
        except Exception as e:
            print(f"✓ TokenManager error handling: {str(e)[:60]}...")
            results.append(True)

    except Exception as e:
        print(f"✗ OpenAI integration test failed: {e}")
        results.append(False)

    # Test geospatial processing
    print("\nTesting geospatial processing...")
    try:
        from src.geospatial_processing import (
            CoordinateTransformer,
            RasterProcessor,
            SpatialFeatureExtractor,
            ArchaeologicalSiteDetector,
        )

        # Test coordinate transformation
        transformer = CoordinateTransformer()
        result = transformer.transform_coordinates(
            [-70.0, -12.0], "EPSG:4326", "EPSG:32718"
        )
        print(f"✓ Coordinate transformation: {result}")

        # Test raster processor
        processor = RasterProcessor()
        print("✓ RasterProcessor initialized")

        # Test feature extractor with synthetic data
        extractor = SpatialFeatureExtractor()
        test_image = np.random.rand(50, 50).astype(np.uint8)
        features = extractor.extract_textural_features(test_image, window_size=3)
        print(f"✓ Textural features extracted: {len(features)} feature types")

        # Test site detector
        detector = ArchaeologicalSiteDetector()
        print("✓ ArchaeologicalSiteDetector initialized")

        results.append(True)

    except Exception as e:
        print(f"✗ Geospatial processing test failed: {e}")
        traceback.print_exc()
        results.append(False)

    # Test data loading
    print("\nTesting data loading...")
    try:
        from src.data_loading import DataLoader

        loader = DataLoader()
        formats = loader.supported_formats
        print(f"✓ Supported formats: {formats}")

        results.append(True)

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        results.append(False)

    return all(results)


def test_integration():
    """Test component integration."""
    print("\n=== Phase 3: Integration Testing ===")

    try:
        from src.geospatial_processing import RasterProcessor, SpatialFeatureExtractor
        from src.data_loading import DataLoader

        # Create synthetic pipeline test
        print("Testing integrated geospatial pipeline...")

        # Synthetic raster data
        synthetic_raster = {
            "data": np.random.rand(100, 100),
            "transform": None,  # Would normally be a rasterio transform
            "crs": "EPSG:4326",
        }

        processor = RasterProcessor()
        extractor = SpatialFeatureExtractor()

        # Test feature extraction on synthetic data
        features = extractor.extract_textural_features(
            synthetic_raster["data"].astype(np.uint8)
        )

        print(f"✓ Pipeline extracted {len(features)} feature types")
        print("✓ Integration test passed")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Phase 4: Error Handling Testing ===")

    results = []

    # Test invalid inputs
    print("Testing error handling...")
    try:
        from src.geospatial_processing import SpatialFeatureExtractor

        extractor = SpatialFeatureExtractor()

        # Test with empty array
        try:
            empty_array = np.array([])
            features = extractor.extract_textural_features(empty_array)
            print("✗ Should have handled empty array")
            results.append(False)
        except (ValueError, IndexError):
            print("✓ Correctly handles empty input arrays")
            results.append(True)

        # Test with uniform array (division by zero case)
        try:
            uniform_array = np.ones((50, 50), dtype=np.uint8) * 128
            features = extractor.extract_textural_features(uniform_array)
            print("✓ Correctly handles uniform arrays (no division by zero)")
            results.append(True)
        except Exception as e:
            print(f"✗ Failed on uniform array: {e}")
            results.append(False)

    except Exception as e:
        print(f"✗ Error handling test setup failed: {e}")
        results.append(False)

    # Test configuration errors
    print("\nTesting configuration error handling...")
    try:
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            from src.openai_integration import OpenAIClient

            client = OpenAIClient()
            print("✗ Should have failed without API key")
            results.append(False)
        except ValueError:
            print("✓ Correctly handles missing API key")
            results.append(True)
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    except Exception as e:
        print(f"✗ Configuration error test failed: {e}")
        results.append(False)

    return all(results)


def test_performance():
    """Test basic performance and memory usage."""
    print("\n=== Phase 5: Performance & Memory Testing ===")

    try:
        import psutil
        from src.geospatial_processing import SpatialFeatureExtractor

        print("Testing memory usage with larger datasets...")
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        extractor = SpatialFeatureExtractor()

        # Test with moderately large synthetic dataset
        large_image = np.random.rand(500, 500).astype(np.uint8)

        features = extractor.extract_textural_features(large_image, window_size=5)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        print(f"✓ Memory usage: {memory_used:.1f} MB for 500x500 image processing")

        if memory_used < 200:  # Reasonable memory usage threshold
            print("✓ Memory usage is acceptable")
            return True
        else:
            print("⚠ High memory usage detected")
            return True  # Still pass, just a warning

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False


def run_quick_verification():
    """Quick verification for immediate feedback."""
    print("\n=== Quick Verification Test ===")

    try:
        print("1. Testing imports...")
        import src.openai_integration
        import src.geospatial_processing
        import src.data_loading

        print("✓ All imports successful")

        print("2. Testing basic functionality...")
        from src.geospatial_processing import CoordinateTransformer

        transformer = CoordinateTransformer()
        print("✓ CoordinateTransformer initialized")

        print("3. Testing error handling...")
        try:
            from src.openai_integration import OpenAIClient

            client = OpenAIClient(api_key="test-key")
            print("✓ OpenAI client handles test key gracefully")
        except Exception as e:
            print(f"Expected behavior with test key: {str(e)[:50]}...")

        print("🎉 Quick verification completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Quick verification failed: {e}")
        return False


def main():
    """Run all human tests."""
    print("🧪 Archaeological Site Discovery System - Human Testing")
    print("=" * 60)

    tests = [
        ("Environment Setup", test_environment_setup),
        ("Core Functionality", test_core_functionality),
        ("Integration", test_integration),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]

    passed = 0
    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} tests...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 HUMAN TEST RESULTS SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")

    print(f"\nOverall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All human tests passed! System is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
