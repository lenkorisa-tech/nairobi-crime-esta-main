# 05_fairness_equity_audit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = 'data/processed/nairobi_panel_dataset.csv'
SENSITIVE_FEATURE = 'Sub_County'  # Spatial grouping for fairness check
TARGET_FEATURE = 'High_Crime'     # We will create this binary target for the audit

# Output paths
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Drop rows with missing values in key columns
df = df.dropna(subset=[SENSITIVE_FEATURE, 'Crime_Count'])

print("Preparing features and target...")
# Create a binary target: 1 if Crime_Count is above median, 0 otherwise
median_crime = df['Crime_Count'].median()
df[TARGET_FEATURE] = (df['Crime_Count'] > median_crime).astype(int)

# Encode categorical features for the model
le_sub = LabelEncoder()
le_settle = LabelEncoder()

df['Sub_County_Encoded'] = le_sub.fit_transform(df['Sub_County'])
df['Settlement_Typology_Encoded'] = le_settle.fit_transform(df['Settlement_Typology'])

# Features for the model
X = df[['Sub_County_Encoded', 'Year', 'Settlement_Typology_Encoded']]
y = df[TARGET_FEATURE]
sensitive_data = df[SENSITIVE_FEATURE]

# Train-test split
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_data, test_size=0.2, random_state=42, stratify=y
)

print("Training a baseline model for audit purposes...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("Generating predictions...")
y_pred = model.predict(X_test)

print("Calculating fairness metrics...")
# Calculate Demographic Parity Difference (DPD)
dpd = demographic_parity_difference(
    y_true=y_test, 
    y_pred=y_pred, 
    sensitive_features=sens_test
)

# Calculate Equalized Odds Difference (EOD)
eod = equalized_odds_difference(
    y_true=y_test, 
    y_pred=y_pred, 
    sensitive_features=sens_test
)

print(f"Demographic Parity Difference (DPD): {dpd:.4f}")
print(f"Equalized Odds Difference (EOD): {eod:.4f}")

# ==========================================
# VISUALIZATION
# ==========================================
print("Generating high-resolution bar chart...")
metrics = ['Demographic Parity\nDifference (DPD)', 'Equalized Odds\nDifference (EOD)']
values = [dpd, eod]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Representational Equity & Fairness Audit\nNairobi Crime Estimation Model', fontsize=14, fontweight='bold')
plt.ylabel('Difference Metric (Lower is Better)', fontsize=12)
plt.ylim(0, max(values) + 0.1 if max(values) > 0.1 else 0.2)
plt.axhline(y=0.1, color='r', linestyle='--', label='Acceptable Threshold (0.10)')
plt.legend()
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'representational_bias_chart.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Chart saved to {chart_path}")

# ==========================================
# REPORT GENERATION
# ==========================================
print("Generating representational_bias_report.md...")
report_path = 'representational_bias_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Appendix: Representational Equity & Fairness Audit Report\n\n")
    f.write(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 1. Overview\n")
    f.write("This report details the fairness audit conducted on the Nairobi Crime Estimation model to ensure representational equity across different spatial groups.\n\n")
    f.write(f"**Sensitive Attribute Evaluated:** `{SENSITIVE_FEATURE}`\n")
    f.write(f"**Target Variable:** `{TARGET_FEATURE}` (Binary: Above/Below Median Crime Count)\n\n")
    f.write("## 2. Fairness Metrics\n")
    f.write("### Demographic Parity Difference (DPD)\n")
    f.write(f"- **Value:** `{dpd:.4f}`\n")
    f.write("- **Interpretation:** Measures the difference in the rate of positive predictions between the most and least favored sub-counties. A value closer to 0 indicates higher fairness.\n\n")
    f.write("### Equalized Odds Difference (EOD)\n")
    f.write(f"- **Value:** `{eod:.4f}`\n")
    f.write("- **Interpretation:** Measures the difference in both true positive rates and false positive rates across sub-counties. A value closer to 0 indicates the model performs equally well for all areas.\n\n")
    f.write("## 3. Conclusion\n")
    if max(dpd, eod) <= 0.1:
        f.write("The model's fairness metrics fall within the generally acceptable threshold (≤ 0.10), indicating minimal representational bias across the evaluated sub-counties.\n")
    else:
        f.write("The model's fairness metrics exceed the 0.10 threshold, indicating potential representational bias. Further bias mitigation techniques should be considered in future iterations.\n")
    f.write("\n## 4. Visual Output\n")
    f.write(f"![Fairness Audit Chart]({chart_path})\n")

print(f"Report saved to {report_path}")
print("\n✅ Phase 5 Complete! Fairness audit successfully executed.")