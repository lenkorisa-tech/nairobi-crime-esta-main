import matplotlib.pyplot as plt
import numpy as np

# Data from our test results
categories = ['Observed Records', 'Missing Records (NaNs)']
informal_pct = [15.5, 75.0]  # Informal Settlements
formal_pct = [84.5, 25.0]    # Formal/Mixed-Use

x = np.arange(len(categories))
width = 0.35

# Create the chart
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, informal_pct, width, label='Informal Settlement', color='#d62728') 
rects2 = ax.bar(x + width/2, formal_pct, width, label='Formal/Mixed-Use', color='#2ca02c') 

# Formatting for academic publication
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Disproportionate Missingness in Informal Settlements', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Saves a high-resolution image to your data folder
plt.savefig('data/processed/Figure_4_Missingness_Typology.png', dpi=300)
print("Chart saved to data/processed/Figure_4_Missingness_Typology.png")
plt.show()