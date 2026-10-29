import pandas as pd
import numpy as np
from scipy import stats

# 1. Load the panel data
df = pd.read_csv("data/processed/proportional_allocation_panel.csv")

# 2. Create a binary missingness indicator (1 if Reported_Count is NaN, 0 otherwise)
df['is_missing'] = df['Reported_Count'].isna().astype(int)

# 3. Test 1: Does Population differ between missing and observed rows?
missing_pop = df[df['is_missing'] == 1]['Population']
observed_pop = df[df['is_missing'] == 0]['Population']
t_stat_pop, p_val_pop = stats.ttest_ind(missing_pop, observed_pop, equal_var=False)

# 4. Test 2: Does Settlement Typology differ? (Informal = 1, Formal = 0)
df['is_informal'] = (df['Settlement_Typology'] == 'Informal Settlement').astype(int)
missing_informal = df[df['is_missing'] == 1]['is_informal']
observed_informal = df[df['is_missing'] == 0]['is_informal']
t_stat_typo, p_val_typo = stats.ttest_ind(missing_informal, observed_informal, equal_var=False)

# 5. Output Results for Chapter 4
print("="*75)
print("📊 MISSINGNESS MECHANISM TEST (Proving MAR/MNAR)")
print("="*75)
print(f"Total Observations: {len(df)}")
print(f"Missing Observations: {df['is_missing'].sum()} ({df['is_missing'].mean()*100:.1f}%)")
print("-" * 75)

print("1. Population Difference Test (Welch's t-test):")
print(f"   Mean Population (Missing rows):    {missing_pop.mean():,.0f}")
print(f"   Mean Population (Observed rows):   {observed_pop.mean():,.0f}")
print(f"   P-Value: {p_val_pop:.4e}")
print("-" * 75)

print("2. Settlement Typology Difference Test:")
print(f"   % Informal Settlements (Missing):  {missing_informal.mean()*100:.1f}%")
print(f"   % Informal Settlements (Observed): {observed_informal.mean()*100:.1f}%")
print(f"   P-Value: {p_val_typo:.4e}")
print("="*75)

alpha = 0.05
if p_val_pop < alpha or p_val_typo < alpha:
    print("\n✅ CONCLUSION: Reject the MCAR assumption (p < 0.05).")
    print("The probability of missing data is significantly associated with sub-county")
    print("Population and Settlement Typology. This confirms the data is Missing At")
    print("Random (MAR) or Missing Not At Random (MNAR), validating the need for")
    print("bias-aware imputation strategies in the predictive policing model.")
else:
    print("\n⚠️ CONCLUSION: No significant difference found (p >= 0.05).")
print("="*75 + "\n")