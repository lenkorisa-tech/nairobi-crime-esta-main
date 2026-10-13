&#x20;  # Methodological Framework: Nairobi Crime Estimation \& Predictive Policing Readiness



&#x20;  ## 1. Overview

&#x20;  This repository contains the code, data processing pipelines, and analytical outputs for assessing publicly available policing data in Nairobi City County, Kenya. The primary objective is to evaluate machine learning-based predictive policing readiness, focusing on urban crime hotspot prediction, spatial-temporal analysis, and ethical AI considerations.



&#x20;  ## 2. Data Landscape \& Processing

&#x20;  - \*\*Data Sources\*\*: Aggregated from publicly available Kenyan national crime statistics (KNBS), spatial OpenStreetMap (OSM) data, and local typology classifications.

&#x20;  - \*\*Processing Pipeline\*\*: Handled via `src/data\_processing.py` and Jupyter notebooks (`01\_data\_landscape\_audit.ipynb`, `02\_eda\_visualizations.ipynb`). Includes missing value imputation, temporal alignment, and spatial joining.



&#x20;  ## 3. Spatial-Temporal Analysis

&#x20;  - \*\*Temporal Trends\*\*: Analyzed year-over-year crime count fluctuations (2018–2024).

&#x20;  - \*\*Spatial Mapping\*\*: Utilized Local Indicators of Spatial Association (LISA) to identify statistically significant crime hotspots and cold spots across Nairobi sub-counties (`03\_spatial\_mapping.ipynb`).



&#x20;  ## 4. Ethical AI \& Fairness Audit

&#x20;  - Recognizing the high risk of algorithmic bias in predictive policing, a representational equity audit was conducted (`05\_fairness\_equity\_audit.py`).

&#x20;  - \*\*Metrics\*\*: Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) evaluated across spatial groupings (`Sub\_County`).

&#x20;  - \*\*Output\*\*: Generates `representational\_bias\_report.md` and visual charts to ensure transparency and accountability.



&#x20;  ## 5. Reproducibility

&#x20;  This project adheres to strict reproducibility standards. All dependencies are listed in `requirements.txt`, and the environment can be replicated using the provided Python virtual environment (`venv`).

