# Nairobi Crime ESTA: Bayesian Spatial-Temporal Predictive Modeling

## 📖 Project Overview
This repository contains the complete data pipeline and source code for a Master's thesis research project. The study develops an **Ensemble Spatio-Temporal Algorithm (ESTA)** to predict and analyze urban crime hotspots within Nairobi City County, Kenya. 

By integrating publicly available demographic data from the Kenya National Bureau of Statistics (KNBS) and crime microdata from the National Crime Research Centre (NCRC), this project utilizes Bayesian spatial modeling to identify high-risk sub-counties and evaluate the socio-environmental covariates driving crime distribution.

## 🗂️ Project Structure
The repository is organized according to professional data science standards to ensure reproducibility:

```text
nairobi-crime-esta/
├── data/
│   ├── raw/             # Original, immutable data drops (Ignored by Git for security)
│   └── processed/       # Cleaned, transformed data ready for modeling
├── notebooks/           
│   └── Spatial/         # Jupyter notebooks for EDA, mapping, and Bayesian modeling
├── src/                 # Modular Python scripts for the ESTA pipeline
│   └── data_processing.py
├── outputs/             # Generated visualizations and reports (Ignored by Git)
├── Dockerfile           # Container configuration for reproducible environments
├── requirements.txt     # Exact Python dependencies
└── README.md            # You are here!
