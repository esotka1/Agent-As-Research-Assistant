import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('resources/input/data/all_years_combined.csv')

# Filter and aggregate relevant data
df_filtered = df[['year', 'item01001', 'item01002', 'item01003']].dropna()
df_grouped = df_filtered.groupby('year').sum().reset_index()

# Separate the National COA data
df_coa = df[['year', 'item040001']].dropna()
df_coa_grouped = df_coa.groupby('year').sum().reset_index()

# Plot
plt.figure(figsize=(8, 6))
plt.plot(df_grouped['year'], df_grouped['item01001'] / 1e6, linestyle=':', color='green', label='0% of part-owned')
plt.plot(df_grouped['year'], df_grouped['item01002'] / 1e6, linestyle='-', color='blue', label='Imputed part ownership')
plt.plot(df_grouped['year'], df_grouped['item01003'] / 1e6, linestyle='--', color='red', label='100% of part-owned')
plt.plot(df_coa_grouped['year'], df_coa_grouped['item040001'] / 1e6, linestyle='-.', color='orange', label='National COA data')

# Formatting
plt.xlabel('Year', fontsize=12)
plt.ylabel('Black-owned land (millions of acres)', fontsize=12)
plt.title('Figure 1. Black-Owned Farm Acreage by Part Ownership, 1910–2012', fontsize=12, pad=20, weight='bold')
plt.legend(loc='upper right', fontsize=10, frameon=False, title='')

# Adjust grid and axes
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1910, 2020, 10), fontsize=10)
plt.yticks(range(0, 16, 5), fontsize=10)
plt.xlim(1910, 2012)
plt.ylim(0, 15)

# Customize axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_bounds(0, 15)
plt.gca().spines['bottom'].set_bounds(1910, 2010)

# Show the plot
plt.tight_layout()
plt.show()