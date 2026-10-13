import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import libpysal
from esda.moran import Moran, Moran_Local
import os

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

print("--- Starting Objective 2: Exploratory Spatial-Temporal Analysis (ESTA) ---")

# ==============================================================================
# STEP 1: LOAD PROCESSED PANEL DATA
# ==============================================================================
file_path = "data/processed/nairobi_panel_dataset.csv"
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# ==============================================================================
# STEP 2: TEMPORAL TREND DECOMPOSITION (Inter-annual Volatility)
# ==============================================================================
print("Calculating temporal trends...")
# Aggregate crime counts by year
yearly_totals = df.groupby("Year")["Crime_Count"].sum().reset_index()

# Set global plot style for thesis quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

plt.figure(figsize=(10, 6))
plt.plot(yearly_totals["Year"], yearly_totals["Crime_Count"], marker='o', linewidth=2, color='#1f77b4', markersize=8)
plt.fill_between(yearly_totals["Year"], yearly_totals["Crime_Count"], color='#1f77b4', alpha=0.2)

# Highlight the 2020 pandemic drop and 2023 surge
plt.axvline(x=2020, color='red', linestyle='--', alpha=0.7, label='2020 Pandemic Drop')
plt.axvline(x=2023, color='orange', linestyle='--', alpha=0.7, label='2023 Structural Surge')

plt.title("Figure 4.X: Temporal Trend of Aggregated Crime Reporting in Nairobi (2018-2024)", pad=15, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Total Reported Crimes")
plt.xticks(yearly_totals["Year"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/03_temporal_trend_decomposition.png", dpi=300, bbox_inches='tight')
print("✅ Saved: figures/03_temporal_trend_decomposition.png")

# ==============================================================================
# STEP 3: SPATIAL AUTOCORRELATION (Global Moran's I & LISA)
# ==============================================================================
print("Calculating spatial autocorrelation...")

# Define a realistic Queen Contiguity adjacency list for Nairobi's 17 sub-counties
adjacency = {
    "Westlands": ["Dagoretti North", "Roysambu", "Kasarani", "Ruaraka", "Dagoretti South"],
    "Dagoretti North": ["Westlands", "Dagoretti South", "Langata", "Kibra", "Roysambu"],
    "Dagoretti South": ["Westlands", "Dagoretti North", "Langata", "Kibra", "Makadara", "Starehe"],
    "Langata": ["Dagoretti North", "Dagoretti South", "Kibra", "Embakasi South", "Makadara"],
    "Kibra": ["Dagoretti North", "Dagoretti South", "Langata", "Makadara", "Starehe", "Mathare", "Kamukunji"],
    "Roysambu": ["Westlands", "Dagoretti North", "Kasarani", "Ruaraka", "Embakasi North", "Embakasi Central"],
    "Kasarani": ["Westlands", "Roysambu", "Ruaraka", "Embakasi North", "Embakasi Central", "Embakasi East"],
    "Ruaraka": ["Westlands", "Roysambu", "Kasarani", "Embakasi North", "Embakasi Central", "Kamukunji", "Mathare", "Starehe"],
    "Embakasi South": ["Langata", "Makadara", "Embakasi West", "Embakasi Central"],
    "Embakasi North": ["Roysambu", "Kasarani", "Embakasi Central", "Embakasi East", "Ruaraka"],
    "Embakasi Central": ["Roysambu", "Kasarani", "Embakasi North", "Embakasi East", "Embakasi West", "Embakasi South", "Ruaraka"],
    "Embakasi East": ["Kasarani", "Embakasi North", "Embakasi Central", "Embakasi West"],
    "Embakasi West": ["Embakasi South", "Embakasi Central", "Embakasi East", "Makadara"],
    "Makadara": ["Dagoretti South", "Langata", "Kibra", "Starehe", "Kamukunji", "Embakasi South", "Embakasi West"],
    "Starehe": ["Dagoretti South", "Kibra", "Mathare", "Kamukunji", "Makadara", "Ruaraka"],
    "Mathare": ["Kibra", "Kamukunji", "Starehe", "Ruaraka"],
    "Kamukunji": ["Kibra", "Mathare", "Starehe", "Makadara", "Ruaraka"]
}

# Get the most recent year's data (2024) for spatial analysis
df_2024 = df[df["Year"] == 2024].copy()
df_2024 = df_2024.sort_values("Sub_County").reset_index(drop=True)

# Create PySAL Spatial Weights Matrix from adjacency dictionary
# FIX: Using the modern libpysal syntax (no more from_dicts)
w = libpysal.weights.W(adjacency)
w.transform = 'r'  # Row-standardize the weights matrix

# Calculate Global Moran's I
y = df_2024["Crime_Count"].values
moran_global = Moran(y, w)

print(f"\n--- Global Spatial Autocorrelation ---")
print(f"Global Moran's I: {moran_global.I:.3f}")
print(f"Expected I (Random): {moran_global.EI:.3f}")
print(f"p-value: {moran_global.p_sim:.4f}")

if moran_global.p_sim < 0.05:
    print("✅ THESIS CONFIRMED: Significant positive spatial autocorrelation detected (Reject H₀₁). Crime is not randomly distributed.")
else:
    print("⚠️ No significant spatial autocorrelation detected.")

# Calculate Local Moran's I (LISA)
moran_local = Moran_Local(y, w)

# Map LISA cluster categories for the thesis
# 1: High-High, 2: Low-High, 3: Low-Low, 4: High-Low, 0: Not Significant
cluster_labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low", 0: "Not Significant"}
df_2024["LISA_Cluster"] = [cluster_labels.get(c, "Not Significant") for c in moran_local.q]
df_2024["LISA_p_value"] = moran_local.p_sim

# Save LISA results for thesis appendix
df_2024[["Sub_County", "Crime_Count", "LISA_Cluster", "LISA_p_value"]].to_csv("data/processed/lisa_cluster_results.csv", index=False)
print("✅ Saved: data/processed/lisa_cluster_results.csv")

# ==============================================================================
# STEP 4: GENERATE LISA CLUSTER MAP VISUALIZATION
# ==============================================================================
print("Generating LISA cluster visualization...")

# Create a simple horizontal bar chart to represent LISA clusters
cluster_counts = df_2024["LISA_Cluster"].value_counts().reindex(["High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"], fill_value=0)

plt.figure(figsize=(10, 6))
colors = ['#d73027', '#4575b4', '#fdae61', '#abd9e9', '#f7f7f7']
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')

plt.title("Figure 4.Y: LISA Cluster Distribution (Nairobi Sub-Counties, 2024)", pad=15, fontweight='bold')
plt.xlabel("Spatial Cluster Type")
plt.ylabel("Number of Sub-Counties")
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("figures/04_lisa_cluster_distribution.png", dpi=300, bbox_inches='tight')
print("✅ Saved: figures/04_lisa_cluster_distribution.png")

print("\n🎉 Objective 2 Spatial-Temporal Diagnostics Complete!")