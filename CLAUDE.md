# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

### Test-Driven Development (TDD)

* As an overall principle, do test-driven development.
* First, write tests based on expected input/output pairs. Avoid creating mock implementations, even for functionality that doesn't exist yet in the codebase.
* Second, run the tests and confirm they fail. Do not write any implementation code at this stage.
* Third, commit the test when you're satisfied with them.
* Then, write code that passes the tests. Do not modify the tests. Keep going until all tests pass.
* Finally, commit the code once only when you're satisfied with the changes.

## Project Overview

This is the "OpenAI to Z Challenge" Kaggle competition project for discovering archaeological sites in the Amazon rainforest using AI models. The goal is to identify unknown archaeological sites using LiDAR data, satellite imagery, NDVI data, GIS data, and archaeological literature.

The project has completed Phase 1 (Production Infrastructure) and Phase 2 (Competition Implementation) with comprehensive TDD implementation, CI/CD pipeline, human testing framework, and complete Jupyter notebook workflow demonstrations. Phase 3 focuses on real competition data integration and final submission.

## Core Development Commands

### Environment Setup
```bash
# Set up virtual environment and dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Testing Commands
```bash
# Quick functionality verification
python quick_test.py

# Comprehensive manual testing
python human_test.py

# Full automated test suite with coverage
pytest

# Run specific test file
pytest tests/test_geospatial_processing.py -v

# Run tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run only unit tests (exclude slow integration tests)
pytest -m "not slow"

# Run integration tests only
pytest -m "integration"

# Run performance benchmarks
pytest -m "benchmark"

# Run tests with specific markers
pytest -m "unit and not slow"
```

### Code Quality Commands
```bash
# Format code (must run before commits)
black .
isort .

# Lint code
flake8 src/ tests/

# Run all quality checks
black . && isort . && flake8 src/ tests/ && pytest

# Complete quality pipeline (recommended before PR)
black . && isort . && flake8 src/ tests/ && pytest --cov=src --cov-report=term-missing
```

### Integration Testing
```bash
# Run integration tests manually
python manual_integration_test.py
```

## Code Architecture

### Core Module Structure

The codebase follows a modular architecture with these key components:

1. **`src/config.py`**: Centralized configuration management with dataclasses for different subsystems (OpenAI, geospatial, archaeology). Uses Pydantic for validation and supports both YAML/JSON config files.

2. **`src/openai_integration.py`**: OpenAI API client with support for o3/o4 mini and GPT-4.1 models. Includes token counting, rate limiting, coordinate extraction from archaeological literature, and asynchronous processing.

3. **`src/geospatial_processing.py`**: Comprehensive geospatial analysis including coordinate transformations, raster operations, terrain analysis (slope, curvature), vegetation analysis (NDVI), and archaeological site detection algorithms.

4. **`src/data_loading.py`**: Multi-format geospatial data loading with validation for GeoTIFF, CSV, Shapefile formats and coordinate reference system handling.

5. **`src/logging_utils.py`**: Centralized logging configuration with file and console handlers, log rotation, and performance timing decorators.

### Key Design Patterns

- **Configuration Management**: Single `ConfigManager` class handles all subsystem configurations with validation and file persistence
- **Data Classes**: Extensive use of `@dataclass` for structured data representation (coordinates, detections, responses)
- **Error Handling**: Comprehensive try-catch blocks with proper logging and graceful degradation
- **Async Support**: OpenAI client supports both sync and async operations for batch processing
- **Modular Processing**: Each module handles a specific domain (AI, geospatial, data) with clear interfaces

### Testing Architecture

- **Unit Tests**: Isolated testing of individual functions with mocked dependencies
- **Integration Tests**: End-to-end testing with real data processing workflows
- **Performance Tests**: Memory usage and processing speed benchmarks
- **Human Testing**: Manual verification scripts for comprehensive system validation
- **CI/CD Pipeline**: Automated testing across Python 3.9-3.13 with coverage reporting

### Data Processing Pipeline

1. **Configuration Loading**: `config.py` loads environment variables and validation rules
2. **Data Ingestion**: `data_loading.py` handles multi-format geospatial data with validation
3. **Geospatial Analysis**: `geospatial_processing.py` performs terrain and vegetation analysis
4. **AI Analysis**: `openai_integration.py` processes archaeological literature for site extraction
5. **Results Integration**: Combine AI-extracted coordinates with geospatial analysis

### Test Suite Architecture

The project uses a comprehensive multi-level testing approach:

**Test Structure:**
* `tests/test_config.py` - Configuration validation and environment handling
* `tests/test_data_loading.py` - Multi-format data loading and validation
* `tests/test_geospatial_processing.py` - Core geospatial algorithms and performance
* `tests/test_openai_integration.py` - AI model integration and coordinate extraction
* `tests/test_logging_utils.py` - Logging configuration and error handling

**Test Markers & Coverage:**
* `pytest.ini` enforces 80% minimum coverage
* Test markers: `slow`, `integration`, `unit`, `benchmark`
* Coverage reports: HTML (htmlcov/) and terminal output
* Performance benchmarks for memory (<200MB) and speed (<30s) targets

**Key Test Patterns:**
* **Mock External Dependencies**: OpenAI API, file system operations, network requests
* **Synthetic Data Generation**: Amazon rainforest signatures for geospatial testing
* **Property-Based Testing**: Coordinate transformations and mathematical operations
* **Error Scenario Testing**: Network failures, invalid data, missing files

## Environment Configuration

### Required Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_api_key_here
```

### System Dependencies
The project requires GDAL for geospatial processing. On Ubuntu/CI:
```bash
sudo apt-get install -y gdal-bin libgdal-dev libproj-dev libgeos-dev
```

## Performance Targets

- Memory usage: <200MB for 500x500 image processing
- Processing speed: <30s for 500x500 textural feature extraction
- Test suite completion: <60s across all Python versions
- Code coverage: >80% (enforced by pytest configuration)

## Competition Implementation Status

### Phase 2 (COMPLETED ✅)
Complete Jupyter notebook implementation located in `notebooks/`:
- `01_archaeological_site_discovery_workflow.ipynb` - End-to-end site discovery workflow
- `02_machine_learning_models.ipynb` - ML/DL models (Random Forest, XGBoost, CNN)
- `03_data_integration_pipeline.ipynb` - Multi-source data integration pipeline

Key achievements:
- Correct NDVI-to-NIR formulas for accurate vegetation analysis
- Complete synthetic data generation with realistic Amazon rainforest signatures
- Full machine learning pipeline with archaeological site detection
- Comprehensive error handling and code quality improvements
- 89% of code review feedback addressed systematically

### Phase 3 (NEXT PRIORITY)
Real competition data integration and final submission:
- Connect to actual Kaggle competition dataset
- Fine-tune models on real archaeological data
- Generate competition site predictions
- Validate results and prepare submission

## Jupyter Notebook Architecture

The `notebooks/` directory contains comprehensive Phase 2 implementation:

**`01_archaeological_site_discovery_workflow.ipynb`**
* End-to-end site discovery pipeline with synthetic Amazon rainforest data
* LiDAR processing, satellite imagery analysis, and NDVI vegetation mapping
* Archaeological feature detection using terrain analysis and clustering
* Coordinate extraction from literature using OpenAI models

**`02_machine_learning_models.ipynb`**
* Random Forest and XGBoost models for site classification
* CNN implementation for image-based site detection
* Feature engineering from geospatial data (slope, curvature, NDVI)
* Model evaluation and performance benchmarking

**`03_data_integration_pipeline.ipynb`**
* Multi-source data integration (LiDAR, satellite, literature)
* Data validation and quality assurance workflows
* Coordinate reference system transformations
* Export pipelines for competition submission

## CI/CD Pipeline Architecture

**GitHub Actions Workflows:**
* `ci.yml`: Main CI pipeline with Python 3.9-3.13 matrix testing
* `claude_code.yml` & `claude_code_login.yml`: Claude Code integration

**Quality Gates:**
* Code formatting (Black, isort) with 88-character line length
* Linting (flake8) with complexity and style checks
* Security scanning (bandit, safety) for vulnerability detection
* Test coverage enforcement (>80% required)
* Performance benchmarks for geospatial processing

## Competition Links

- Competition Page: https://www.kaggle.com/competitions/openai-to-z-challenge/
- Rules: https://www.kaggle.com/competitions/openai-to-z-challenge/rules
- Discussion: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion