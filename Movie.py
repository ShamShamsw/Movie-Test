# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score

# Load the data from the Excel file
df = pd.read_excel('Dataset/MovieFranchises.xlsx')
# Convert Budget and Lifetime Gross columns to numerical values (removing currency symbols and commas)
df['Budget'] = df['Budget'].replace('[\$,]', '', regex=True).astype(float)
df['Lifetime Gross'] = df['Lifetime Gross'].replace('[\$,]', '', regex=True).astype(float)

# Handle missing or invalid data
df.dropna(subset=['Budget', 'Lifetime Gross'], inplace=True)

# Create a histogram of Budget
bin_width = 50_000_000  # $50 million per bin
max_budget = df['Budget'].max()
bins = np.arange(0, max_budget + bin_width, bin_width)

plt.figure(figsize=(10, 6))
plt.hist(df['Budget'], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Budget ($ Millions)')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Budgets')
plt.xticks(bins, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('budget_histogram.png')  # Save the histogram
plt.show()

# Perform linear regression: Budget vs. Lifetime Gross
X = df[['Budget']].values  # Independent variable
y = df['Lifetime Gross'].values  # Dependent variable

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Sort values for a cleaner regression line
sorted_indices = np.argsort(X.flatten())
X_sorted = X[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plot Budget vs. Lifetime Gross with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Budget ($)')
plt.ylabel('Lifetime Gross ($)')
plt.title('Linear Regression: Budget vs. Lifetime Gross')
plt.legend()
plt.text(X.max() * 0.7, y.max() * 0.8, f'R² = {r2:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.savefig('regression_plot.png')  # Save the regression plot
plt.show()

# Print the R-squared value
print(f'R² value: {r2:.4f}')