# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data from the Excel file
df = pd.read_excel('MovieFranchises.xlsx')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Task 1: Histogram of Budget

# Extract the 'Budget' column (Column H) and remove any non-numeric characters like dollar signs
df['Budget'] = df['Budget'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Create bins for the histogram, ranging from 0 to the maximum budget, with each bin covering $50 million
max_budget = df['Budget'].max()
bins = np.arange(0, max_budget + 50, 50)  # Bins of size 50 million

# Plot the histogram of the budget column
plt.figure(figsize=(10, 6))
plt.hist(df['Budget'], bins=bins, edgecolor='black', color=plt.cm.Paired(np.arange(len(bins)-1)))  # Different color for each bin
plt.title('Distribution of Movie Budgets')
plt.xlabel('Budget in $Millions')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Task 2: Linear Regression Analysis

# Extract the relevant columns for the regression: 'Budget' (Column H) and 'Lifetime Gross' (Column C)
df['Lifetime Gross'] = df['Lifetime Gross'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Remove rows where either Budget or Lifetime Gross is NaN (i.e., missing data)
df_cleaned = df.dropna(subset=['Budget', 'Lifetime Gross'])

# Create the independent (X) and dependent (y) variables
X = df_cleaned[['Budget']]  # Independent variable (Budget)
y = df_cleaned['Lifetime Gross']  # Dependent variable (Lifetime Gross)

# Create a linear regression model
model = LinearRegression()

# Fit the model with the data
model.fit(X, y)

# Get the R-squared value (coefficient of determination) to evaluate the model's fit
r_squared = model.score(X, y)

# Plot the data points and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')  # Scatter plot of data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot regression line

# Add R-squared value as text on the plot
plt.text(0.1, 0.9, f'R-squared = {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=14, color='black')

# Add titles and labels
plt.title('Budget vs Lifetime Gross')
plt.xlabel('Budget in $Millions')
plt.ylabel('Lifetime Gross in $Millions')
plt.legend()
plt.grid(True)
plt.show()