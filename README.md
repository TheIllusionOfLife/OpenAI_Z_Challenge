# OpenAI to Z Challenge - Amazon Rainforest

## Project Overview

This project aims to address the OpenAI to Z Challenge on Kaggle. The goal is to discover unknown archaeological sites in the Amazon rainforest using various data sources and machine learning models.

## Competition Details

- **Title**: OpenAI to Z Challenge
- **Objective**: Discover unknown archaeological sites in the Amazon rainforest.
- **Models**: OpenAI o3/o4 mini and GPT-4.1 models.
- **Organizer**: Kaggle
- **Competition Page**: [https://www.kaggle.com/competitions/openai-to-z-challenge/](https://www.kaggle.com/competitions/openai-to-z-challenge/)

## Workflow

This project will follow a phased approach as outlined in the competition details:

1.  **Phase 1: Data Understanding and Preprocessing**
2.  **Phase 2: Model Building and Evaluation**
3.  **Phase 3: Discovery Validation and Archaeological Evaluation**
4.  **Phase 4: Results Documentation and Submission**

## Project Structure

-   `data/`: Directory for raw, processed, and external data.
    -   `data/raw/`: Raw data as provided or downloaded.
    -   `data/processed/`: Cleaned and transformed data ready for modeling.
-   `docs/`: Documentation files, including the final `discovery_report.md`.
-   `notebooks/`: Jupyter notebooks for different phases of the project.
    -   `notebooks/phase1_data_preprocessing/01_initial_data_exploration.ipynb`: Data loading, initial inspection, and preprocessing.
    -   `notebooks/phase2_model_building/02_model_building_and_evaluation.ipynb`: Model training, tuning, and evaluation.
    -   `notebooks/phase3_validation/03_discovery_validation.ipynb`: Validation of model predictions against literature and other sources.
    -   `notebooks/phase4_submission/04_submission_preparation.ipynb`: Preparing the final report and submission files.
-   `src/`: Source code for reusable functions, classes, or scripts.
-   `models/`: Saved trained models. (This directory will be created when models are saved)
-   `requirements.txt`: Python package dependencies. (To be generated)
-   `LICENSE`: Project license. (To be added)

## Usage

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL_HERE]
    cd openai-to-z-challenge
    ```
2.  **Set up the environment:**
    (It's recommended to use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Download Data:**
    (Instructions for obtaining data - e.g., from Kaggle - should be placed here. For now, assume data is placed in `data/raw/`)

4.  **Run Notebooks:**
    Navigate to the `notebooks/` directory and run the Jupyter notebooks in sequence:
    -   `phase1_data_preprocessing/01_initial_data_exploration.ipynb`
    -   `phase2_model_building/02_model_building_and_evaluation.ipynb`
    -   `phase3_validation/03_discovery_validation.ipynb`
    -   `phase4_submission/04_submission_preparation.ipynb`

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeatureName`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some YourFeatureName'`).
5.  Push to the branch (`git push origin feature/YourFeatureName`).
6.  Open a Pull Request.

Please ensure your code adheres to standard Python style guides (e.g., PEP 8).
