# ================================
# DATA CLEANING & VISUALIZATION PROJECT
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Load Dataset
# --------------------------------
file_path = "raw_data.csv"      

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!\n")
except:
    print("Error: Dataset not found.")
    exit()

# --------------------------------
# Basic Information
# --------------------------------
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------
# Handle Missing Values
# --------------------------------
print("\nHandling Missing Values...")

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --------------------------------
# Remove Duplicate Rows
# --------------------------------
duplicates_before = df.duplicated().sum()

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()

print(f"\nDuplicates Removed: {duplicates_before - duplicates_after}")

# --------------------------------
# Remove Outliers using IQR Method
# --------------------------------
print("\nRemoving Outliers...")

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

# --------------------------------
# Save Cleaned Dataset
# --------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved as cleaned_data.csv")

# --------------------------------
# Descriptive Statistics
# --------------------------------
print("\nStatistical Summary:")
print(df.describe())

# --------------------------------
# Visualization Settings
# --------------------------------
sns.set_style("whitegrid")

# ======================================
# Histogram for Numeric Features
# ======================================
for col in numeric_cols:

    plt.figure(figsize=(8,5))

    sns.histplot(df[col], bins=20, kde=True)

    plt.title(f"Distribution of {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.show()

# ======================================
# Boxplots for Outlier Detection
# ======================================
for col in numeric_cols:

    plt.figure(figsize=(8,4))

    sns.boxplot(x=df[col])

    plt.title(f"Boxplot of {col}")

    plt.show()

# ======================================
# Correlation Heatmap
# ======================================
plt.figure(figsize=(10,8))

correlation_matrix = df[numeric_cols].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ======================================
# Pair Plot
# ======================================
sns.pairplot(df[numeric_cols])

plt.show()

# ======================================
# Count Plots for Categorical Variables
# ======================================
for col in categorical_cols:

    plt.figure(figsize=(10,5))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.title(f"Count Plot of {col}")

    plt.xticks(rotation=45)

    plt.show()

# ======================================
# Generate Insights
# ======================================
print("\n====== Key Insights ======")

for col in numeric_cols:
    print(f"{col}:")
    print(f"Mean = {df[col].mean():.2f}")
    print(f"Median = {df[col].median():.2f}")
    print(f"Maximum = {df[col].max():.2f}")
    print(f"Minimum = {df[col].min():.2f}")
    print("-"*30)

print("\nAnalysis Completed Successfully!")
