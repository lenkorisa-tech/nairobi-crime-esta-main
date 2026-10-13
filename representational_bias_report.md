# Appendix: Representational Equity & Fairness Audit Report

**Date Generated:** 2026-10-13 10:17:43

## 1. Overview
This report details the fairness audit conducted on the Nairobi Crime Estimation model to ensure representational equity across different spatial groups.

**Sensitive Attribute Evaluated:** `Sub_County`
**Target Variable:** `High_Crime` (Binary: Above/Below Median Crime Count)

## 2. Fairness Metrics
### Demographic Parity Difference (DPD)
- **Value:** `1.0000`
- **Interpretation:** Measures the difference in the rate of positive predictions between the most and least favored sub-counties. A value closer to 0 indicates higher fairness.

### Equalized Odds Difference (EOD)
- **Value:** `1.0000`
- **Interpretation:** Measures the difference in both true positive rates and false positive rates across sub-counties. A value closer to 0 indicates the model performs equally well for all areas.

## 3. Conclusion
The model's fairness metrics exceed the 0.10 threshold, indicating potential representational bias. Further bias mitigation techniques should be considered in future iterations.

## 4. Visual Output
![Fairness Audit Chart](outputs\representational_bias_chart.png)
