# TRIPOD-AI Transparency Checklist

This document details the project's adherence to the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis - Artificial Intelligence (TRIPOD-AI) guidelines, adapted for spatial predictive modeling.

## 1. Title & Abstract
- [x] **Exploratory Nature**: The title and abstract clearly state the exploratory and evaluative nature of the study regarding predictive policing readiness, rather than claiming a deployable operational tool.

## 2. Methods
- [x] **Data Sources**: Details the use of aggregated Kenya National Bureau of Statistics (KNBS) and National Crime Research Centre (NCRC) data, explicitly noting the absence of incident-level, geo-tagged microdata.
- [x] **Modeling Approach**: Methods detail the use of spatial block cross-validation (CV) and Bayesian Conditional Autoregressive (CAR) priors to account for spatial autocorrelation.
- [x] **Missing Data**: Methods and results present Missing Not At Random (MNAR) patterns, acknowledging structural data gaps in specific regions.

## 3. Results
- [x] **Performance & Equity**: Results present both predictive performance metrics and formal equity audit outcomes. 
- [x] **Fairness Metrics**: Reports a Demographic Parity Difference (DPD) of ~0.504, highlighting a severe representational bias and data completeness gap between formal/mixed wards and informal/high-density settlements (e.g., Mathare, Kibra, Makadara).

## 4. Discussion
- [x] **Limitations**: Critically addresses the Modifiable Areal Unit Problem (MAUP) and Small-N constraints inherent in aggregated sub-county data.
- [x] **Ethical Implications**: Explicitly states that deploying standard ML predictive policing models on this aggregated data would inherit structural biases, systematically mischaracterizing marginalized communities. 

## 5. Computational Pipeline & Reproducibility Artifacts
- [x] **Code Repository**: Publicly available at [https://github.com/lenkorisa-tech/nairobi-crime-esta-main](https://github.com/lenkorisa-tech/nairobi-crime-esta-main) (MIT License).
- [x] **Repository Contents**: Includes `data/` (anonymized CSV/GeoJSON), `notebooks/` (Jupyter EDA workflows), `src/` (Modular Python scripts for ESTA & fairness audits), `requirements.txt`, and `Dockerfile`.
- [x] **Environment Pinning**: Global random seed (`random_state=42`) is set for all stochastic operations to ensure exact reproducibility.