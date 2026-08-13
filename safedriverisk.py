# ==============================
# Safe Drive Risk Insurance System
# Libraries
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("smart_city_traffic_stress_dataset.csv")

print("First 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------
# Create Risk Level
# ------------------------------
conditions = [
    (df["stress_index"] < 35),
    (df["stress_index"] >= 35) & (df["stress_index"] < 65),
    (df["stress_index"] >= 65)
]

risk = ["Low", "Medium", "High"]

df["Risk_Level"] = np.select(conditions, risk, default="Medium")

print("\nRisk Level Count")
print(df["Risk_Level"].value_counts())

# ------------------------------
# Insurance Premium
# ------------------------------
premium = []

for i in df["Risk_Level"]:
    if i == "Low":
        premium.append(3000)
    elif i == "Medium":
        premium.append(5000)
    else:
        premium.append(8000)

df["Insurance_Premium"] = premium

print(df.head())

# ------------------------------
# Statistical Summary
# ------------------------------
print(df.describe())

# ------------------------------
# Graph 1
# Stress Index Distribution
# ------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["stress_index"], bins=20, kde=True)
plt.title("Stress Index Distribution")
plt.show()

# ------------------------------
# Graph 2
# Risk Level Count
# ------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Risk_Level")
plt.title("Risk Level")
plt.show()

# ------------------------------
# Graph 3
# Average Speed vs Stress
# ------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(data=df,
                x="avg_speed",
                y="stress_index",
                hue="Risk_Level")
plt.title("Average Speed vs Stress")
plt.show()

# ------------------------------
# Graph 4
# Weather vs Stress
# ------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df,
            x="weather_condition",
            y="stress_index")
plt.title("Weather Condition vs Stress")
plt.xticks(rotation=20)
plt.show()

# ------------------------------
# Graph 5
# Correlation Heatmap
# ------------------------------
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# Save Result
# ------------------------------
df.to_csv("SafeDrive_Insurance_Result.csv", index=False)

print("\nResult Saved Successfully")