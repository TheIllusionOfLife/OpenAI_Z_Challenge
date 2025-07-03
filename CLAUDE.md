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

The project has completed Phase 1 (Production Infrastructure) with comprehensive TDD implementation, CI/CD pipeline, and human testing framework. Phase 2 focuses on competition implementation with Jupyter notebooks and real data integration.

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
- **CI/CD Pipeline**: Automated testing across Python 3.9-3.11 with coverage reporting

### Data Processing Pipeline

1. **Configuration Loading**: `config.py` loads environment variables and validation rules
2. **Data Ingestion**: `data_loading.py` handles multi-format geospatial data with validation
3. **Geospatial Analysis**: `geospatial_processing.py` performs terrain and vegetation analysis
4. **AI Analysis**: `openai_integration.py` processes archaeological literature for site extraction
5. **Results Integration**: Combine AI-extracted coordinates with geospatial analysis

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

## Competition Implementation (Phase 2)

The project is currently focused on Phase 2 implementation with Jupyter notebooks located in `notebooks/`:
- `01_archaeological_site_discovery_workflow.ipynb`
- `02_machine_learning_models.ipynb`
- `03_data_integration_pipeline.ipynb`

## Competition Links

- Competition Page: https://www.kaggle.com/competitions/openai-to-z-challenge/
- Rules: https://www.kaggle.com/competitions/openai-to-z-challenge/rules
- Discussion: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion