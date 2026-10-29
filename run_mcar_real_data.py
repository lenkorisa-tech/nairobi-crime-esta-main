import pandas as pd
import numpy as np
from scipy.stats import chi2

def little_mcar_test(X):
    # Keep only numeric columns and drop columns that are 100% NaN
    dataset = X.select_dtypes(include=[np.number]).dropna(axis=1, how='all').copy()
    vars_list = dataset.columns.tolist()
    n_var = len(vars_list)
    
    if n_var == 0:
        raise ValueError("No numeric columns with data found for MCAR test.")

    gmean = dataset.mean()
    gcov = dataset.cov()
    
    r = dataset.isnull().astype(int)
    mdp = r.dot(2 ** np.arange(n_var))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    
    dataset["mdp"] = [sorted_mdp.index(x) for x in mdp]
    
    pj = 0
    d2 = 0.0
    
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset["mdp"] == i, vars_list]
        observed_mask = ~dataset_temp.isnull().any()
        pj += np.sum(observed_mask)
        
        obs_vars = dataset_temp.columns[observed_mask].tolist()
        if len(obs_vars) == 0:
            continue
            
        means_diff = dataset_temp[obs_vars].mean() - gmean[obs_vars]
        sub_cov = gcov.loc[obs_vars, obs_vars]
        mj = len(dataset_temp)
        
        try:
            cov_inv = np.linalg.inv(sub_cov)
            parta = np.dot(means_diff.T, cov_inv)
            d2 += mj * np.dot(parta, means_diff)
        except np.linalg.LinAlgError:
            continue
            
    df_degrees = int(pj - n_var)
    
    if df_degrees > 0 and d2 >= 0:
        p_value = float(1 - chi2.cdf(d2, df_degrees))
    else:
        p_value = 1.0
        d2 = 0.0
        
    return float(d2), float(p_value), df_degrees

# 1. Load your REAL KEBS Excel data
file_path = "KEBSCrimedata.xlsx"

try:
    print(f"Loading real data from: {file_path}...")
    real_df = pd.read_excel(file_path)
    print(f"✅ Successfully loaded data with shape: {real_df.shape}")
    
    # 2. Run Little's MCAR Test
    print("\n⏳ Running Little's MCAR Test on real KEBS data...")
    chi_square, p_value, df_degrees = little_mcar_test(real_df)
    
    # 3. Output Results for Chapter 4
    print("\n" + "="*70)
    print("📊 LITTLE'S MCAR TEST RESULTS (Real KEBS Data)")
    print("="*70)
    print(f"Chi-Square Statistic: {chi_square:.4f}")
    print(f"Degrees of Freedom:   {df_degrees}")
    print(f"P-Value:              {p_value:.4e}")
    print("="*70)
    
    alpha = 0.05
    if p_value < alpha:
        print("\n⚠️ CONCLUSION: Reject the null hypothesis (p < 0.05).")
        print("The data is NOT Missing Completely At Random (MCAR).")
        print("\n📝 Suggested Chapter 4 phrasing:")
        print(f"   'Little’s MCAR test conducted on the real KEBS data yielded a chi-square statistic of {chi_square:.2f} (df = {df_degrees}, p = {p_value:.4e}). This result leads to the rejection of the MCAR assumption, indicating that the missingness is likely systematic (MAR or MNAR). This validates the need for ethical, bias-aware imputation strategies, as missing data may disproportionately reflect under-reporting in specific sub-counties.'")
    else:
        print("\n✅ CONCLUSION: Fail to reject the null hypothesis (p >= 0.05).")
        print("The data may be Missing Completely At Random (MCAR).")
        print("\n📝 Suggested Chapter 4 phrasing:")
        print(f"   'Little’s MCAR test conducted on the real KEBS data yielded a chi-square statistic of {chi_square:.2f} (df = {df_degrees}, p = {p_value:.4e}). This result indicates no significant evidence against the MCAR assumption, supporting the use of standard imputation techniques for the predictive policing model.'")
    print("="*70 + "\n")

except FileNotFoundError:
    print(f"❌ Error: Could not find '{file_path}'.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")