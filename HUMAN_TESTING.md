# Human Testing Guide

This guide provides comprehensive instructions for manually testing the archaeological site discovery system to verify it works correctly from a user perspective.

## Quick Start

For immediate verification, run the quick test:

```bash
source .venv/bin/activate
python quick_test.py
```

## Comprehensive Testing

For full system verification, run the comprehensive test suite:

```bash
source .venv/bin/activate
python human_test.py
```

## Manual Testing Phases

### Phase 1: Environment Setup (5-10 minutes)

**Goal**: Verify the system can be set up and dependencies work

#### 1.1 Environment Activation
```bash
source .venv/bin/activate
python --version  # Should show Python 3.9+
```

#### 1.2 Core Module Import Test
```bash
python -c "
print('Testing core imports...')
import src.openai_integration; print('✓ OpenAI integration')
import src.geospatial_processing; print('✓ Geospatial processing')
import src.data_loading; print('✓ Data loading')
import src.config; print('✓ Configuration')
import src.logging_utils; print('✓ Logging utilities')
print('✅ All core modules imported successfully')
"
```

#### 1.3 Dependency Verification
```bash
python -c "
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import openai
import tiktoken
print('✅ All major dependencies available')
"
```

### Phase 2: Core Functionality Testing (15-20 minutes)

#### 2.1 OpenAI Integration Testing
```bash
# Test graceful API key handling
python -c "
from src.openai_integration import OpenAIClient, TokenManager

# Should fail gracefully without API key
try:
    client = OpenAIClient()
    print('✗ Should have failed without API key')
except ValueError as e:
    print(f'✓ Correctly handles missing API key: {str(e)[:50]}...')

# Test token manager with invalid model
try:
    manager = TokenManager('invalid-model')
    print('✓ TokenManager handles invalid model gracefully')
except Exception as e:
    print(f'✓ Expected token manager behavior: {str(e)[:50]}...')
"
```

#### 2.2 Geospatial Processing Testing
```bash
# Test coordinate transformation
python -c "
from src.geospatial_processing import CoordinateTransformer
transformer = CoordinateTransformer()

# Test basic coordinate conversion (Peru coordinates)
result = transformer.transform_coordinates([-70.0, -12.0], 'EPSG:4326', 'EPSG:32718')
print(f'✓ Coordinate transformation: {result}')

# Test with multiple points
points = [[-70.0, -12.0], [-71.0, -13.0]]
results = transformer.transform_coordinates(points, 'EPSG:4326', 'EPSG:32718')
print(f'✓ Multiple coordinate transformation: {len(results)} points converted')
"
```

```bash
# Test raster processing
python -c "
import numpy as np
from src.geospatial_processing import RasterProcessor, SpatialFeatureExtractor

# Create synthetic test data
test_raster = np.random.rand(100, 100).astype(np.uint8)

processor = RasterProcessor()
print('✓ RasterProcessor initialized')

extractor = SpatialFeatureExtractor()
features = extractor.extract_textural_features(test_raster, window_size=5)
print(f'✓ Extracted {len(features)} textural feature types')

# Test with uniform data (edge case)
uniform_data = np.ones((50, 50), dtype=np.uint8) * 128
uniform_features = extractor.extract_textural_features(uniform_data)
print('✓ Handles uniform data without division by zero')
"
```

#### 2.3 Data Loading Testing
```bash
# Test data loader capabilities
python -c "
from src.data_loading import DataLoader

loader = DataLoader()
formats = loader.supported_formats
print(f'✓ Supported formats: {formats}')

# Test file validation
test_files = ['test.tif', 'test.shp', 'test.csv', 'test.json']
for file in test_files:
    format_type = loader.detect_file_format(file)
    print(f'✓ {file} -> {format_type}')
"
```

### Phase 3: Integration Testing (10-15 minutes)

#### 3.1 End-to-End Pipeline Test
```bash
# Run the manual integration test
python manual_integration_test.py
```

#### 3.2 Custom Integration Test
```bash
# Test components working together
python -c "
import numpy as np
from src.geospatial_processing import RasterProcessor, SpatialFeatureExtractor, ArchaeologicalSiteDetector
from src.data_loading import DataLoader

print('Testing integrated archaeological pipeline...')

# Synthetic workflow
synthetic_raster = {
    'data': np.random.rand(200, 200),
    'transform': None,
    'crs': 'EPSG:4326'
}

# Initialize components
processor = RasterProcessor()
extractor = SpatialFeatureExtractor()
detector = ArchaeologicalSiteDetector()

print('✓ All components initialized')

# Extract features
features = extractor.extract_textural_features(
    synthetic_raster['data'].astype(np.uint8)
)
print(f'✓ Features extracted: {list(features.keys())}')

print('✅ Integration pipeline test completed')
"
```

### Phase 4: Error Handling Testing (10 minutes)

#### 4.1 Input Validation Testing
```bash
# Test error handling with invalid inputs
python -c "
import numpy as np
from src.geospatial_processing import SpatialFeatureExtractor

extractor = SpatialFeatureExtractor()

# Test empty array
try:
    empty_array = np.array([])
    features = extractor.extract_textural_features(empty_array)
    print('✗ Should have handled empty array')
except (ValueError, IndexError):
    print('✓ Correctly handles empty input arrays')

# Test 1D array
try:
    onedim_array = np.array([1, 2, 3, 4, 5])
    features = extractor.extract_textural_features(onedim_array)
    print('✗ Should have handled 1D array')
except (ValueError, IndexError):
    print('✓ Correctly rejects 1D arrays')

print('✅ Input validation tests passed')
"
```

#### 4.2 Configuration Error Testing
```bash
# Test configuration error handling
python -c "
import os
from src.openai_integration import OpenAIClient

# Save original environment
original_key = os.environ.get('OPENAI_API_KEY')

# Test without API key
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

try:
    client = OpenAIClient()
    print('✗ Should have failed without API key')
except ValueError:
    print('✓ Correctly handles missing API key')

# Restore environment
if original_key:
    os.environ['OPENAI_API_KEY'] = original_key

print('✅ Configuration error tests passed')
"
```

### Phase 5: Performance Testing (5 minutes)

#### 5.1 Memory Usage Test
```bash
# Test memory usage with larger datasets
python -c "
import psutil
import numpy as np
from src.geospatial_processing import SpatialFeatureExtractor

print('Testing memory usage...')
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

extractor = SpatialFeatureExtractor()
large_image = np.random.rand(1000, 1000).astype(np.uint8)

features = extractor.extract_textural_features(large_image, window_size=7)

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f'✓ Memory usage: {memory_used:.1f} MB for 1000x1000 image')

if memory_used < 500:
    print('✓ Memory usage is acceptable')
else:
    print('⚠ High memory usage detected - consider optimization')
"
```

#### 5.2 Processing Speed Test
```bash
# Test processing speed
python -c "
import time
import numpy as np
from src.geospatial_processing import SpatialFeatureExtractor

extractor = SpatialFeatureExtractor()
test_image = np.random.rand(500, 500).astype(np.uint8)

start_time = time.time()
features = extractor.extract_textural_features(test_image)
end_time = time.time()

processing_time = end_time - start_time
print(f'✓ Processing time: {processing_time:.2f} seconds for 500x500 image')

if processing_time < 30:
    print('✓ Processing speed is acceptable')
else:
    print('⚠ Slow processing detected - consider optimization')
"
```

## Success Criteria Checklist

### ✅ Environment Setup
- [ ] Virtual environment activated successfully
- [ ] Python 3.9+ running
- [ ] All core modules import without errors
- [ ] All major dependencies available

### ✅ Core Functionality
- [ ] OpenAI integration handles missing API keys gracefully
- [ ] Coordinate transformations work correctly
- [ ] Geospatial processing handles synthetic data
- [ ] Feature extraction produces expected output types
- [ ] Data loading supports expected file formats

### ✅ Integration
- [ ] Components work together without conflicts
- [ ] End-to-end pipeline processes synthetic data
- [ ] No import or initialization conflicts

### ✅ Error Handling
- [ ] Empty/invalid inputs handled gracefully
- [ ] Missing configurations produce clear errors
- [ ] Edge cases (uniform data) don't cause crashes
- [ ] System provides helpful error messages

### ✅ Performance
- [ ] Memory usage reasonable for test datasets (< 500MB for 1000x1000)
- [ ] Processing speed acceptable (< 30s for 500x500)
- [ ] No memory leaks in repeated operations

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If modules can't be imported, check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Ensure you're in the project root and virtual environment is active
pwd  # Should show project root
which python  # Should show .venv/bin/python
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt

# Check specific missing packages
python -c "import [package_name]"
```

#### Permission Errors
```bash
# Ensure files are executable
chmod +x *.py
```

### Expected Behaviors

- **Missing API Key**: Should raise `ValueError` with clear message
- **Invalid Coordinates**: Should handle gracefully or raise informative error
- **Empty Data**: Should validate input and raise appropriate exceptions
- **Memory Usage**: Should be proportional to input size, not excessive

## Integration with Development Workflow

This human testing should be performed:

1. **After major code changes** - Run `python quick_test.py`
2. **Before creating PRs** - Run `python human_test.py`
3. **After bug fixes** - Test specific scenarios manually
4. **Before releases** - Full manual verification of all phases

The human testing complements automated tests by verifying real-world usage scenarios and user experience aspects that unit tests might miss.