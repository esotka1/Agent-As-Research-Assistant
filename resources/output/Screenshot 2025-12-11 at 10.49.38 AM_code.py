import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'resources/input/data/all_years_combined.csv'
data = pd.read_csv(file_path)

# Filter the relevant columns
data = data[['year', 'var6', 'var7', 'var178', 'var179', 'var180']]
data.dropna(inplace=True)

# Plotting
plt.figure(figsize=(10, 6))

# Correspond variables to colors
plt.scatter(data['year'], data['var6'], color='blue', label='Average value of farms/acre (operator)', s=10)
plt.scatter(data['year'], data['var179'], color='green', label='Average value of land and buildings/acre (part owners)', s=10)
plt.scatter(data['year'], data['var178'], color='orange', label='Average value of land and buildings/acre (full owners)', s=10)
plt.scatter(data['year'], data['var180'], color='red', label='Average value of land and buildings/acre', s=10)

# Set y-axis scale and limits
plt.ylim(0, 12000)
plt.xlim(1920, 2002)

# Formatting
plt.xlabel('Year', fontsize=12)
plt.ylabel('$/acre', fontsize=12)
plt.title('Figure 2. Land Values, 1920–2002', fontsize=14)
plt.xticks(ticks=range(1920, 2011, 20), fontsize=10)
plt.yticks(fontsize=10)

# Correct legend formatting
plt.legend(loc='upper left', fontsize=9, frameon=False)

plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()