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

## Project Architecture

The competition follows a structured workflow with four main phases:

1. **Data Understanding and Preprocessing**: Handle LiDAR point cloud data, satellite images, NDVI data, GIS data, and archaeological literature in formats like GeoTIFF, CSV, and JSON
2. **Feature Engineering**: Extract terrain features from LiDAR, vegetation indices from satellite imagery, and land use information from GIS data
3. **Model Building and Evaluation**: Implement ML models (Random Forest, XGBoost) and deep learning models (CNN, Transformer) for site detection
4. **Discovery Verification**: Validate discovered sites against archaeological literature and expert review

## Key Technologies

- **OpenAI Models**: o3/o4 mini and GPT-4.1 for natural language processing of archaeological literature
- **Geospatial Libraries**: GeoPandas, Rasterio for spatial data processing
- **ML/DL Frameworks**: scikit-learn, XGBoost for traditional ML; CNN/Transformer architectures for deep learning
- **Data Processing**: Pandas, NumPy for data manipulation

## Data Types

The project handles multiple geospatial data formats:
- **LiDAR**: Point cloud data in formats like .las/.laz or GeoTIFF
- **Satellite Imagery**: High-resolution images, typically GeoTIFF
- **NDVI Data**: Vegetation index data
- **GIS Data**: Terrain and land use information
- **Archaeological Literature**: Research papers and metadata with DOIs

## Evaluation Criteria

Submissions are evaluated on:
1. Number of discovered archaeological sites
2. Reliability backed by credible sources (LiDAR tile IDs, DOIs, satellite scene IDs)
3. Transparency and reproducibility of notebooks
4. Archaeological expert review quality

## Submission Requirements

- **Jupyter Notebooks**: Complete analysis and model building process
- **Documentation**: Site locations, features, and archaeological background
- **GitHub Repository**: Public code and data sharing
- **200-word Summary**: Concise project description

## Competition Links

- Competition Page: https://www.kaggle.com/competitions/openai-to-z-challenge/
- Rules: https://www.kaggle.com/competitions/openai-to-z-challenge/rules
- Discussion: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion