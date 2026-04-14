# International-Trade-Forecasting-Risk-Modeling
A machine learning pipeline that leverages regression and classification models to forecast export volumes and assess economic risk levels using key indicators like GDP and trade data.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# 0. Generate  Data 
# ==========================================
np.random.seed(42)
n_samples = 200

# Create correlated features
imports_data = np.random.randint(1000, 5000, n_samples)
gdp_data = imports_data * 12 + np.random.randint(-5000, 5000, n_samples)
exports_data = imports_data * 0.85 + np.random.randint(-300, 300, n_samples)

# Create categorical risk score (High risk if Imports >> Exports)
risk_data = ["High" if (i - e) > 800 else "Low" for i, e in zip(imports_data, exports_data)]

df = pd.DataFrame({
    "imports": imports_data,
    "gdp": gdp_data,
    "exports": exports_data,
    "risk_score": risk_data
})

print("Dataset Preview:")
print(df.head(), "\n")

# ==========================================
# 1. Trade Forecasting (Regression)
# ==========================================
print("--- Trade Forecasting ---")

X_reg = df[["imports", "gdp"]]
y_reg = df["exports"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse:.2f}")

# Plot results (Enhanced with Ideal Prediction Line)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, label="Predictions")

# Add a diagonal line for perfect predictions
min_val = min(y_test_reg.min(), y_pred_reg.min())
max_val = max(y_test_reg.max(), y_pred_reg.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

plt.xlabel("Actual Exports")
plt.ylabel("Predicted Exports")
plt.title("Trade Forecasting: Actual vs Predicted")
plt.legend()
plt.show()

# ==========================================
# 2. Risk Classification
# ==========================================
print("\n--- Risk Classification ---")

X_clf = df[["exports", "imports", "gdp"]]
y_clf = df["risk_score"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

y_pred_clf = clf_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 3. Predict New Country
# ==========================================
print("\n--- New Prediction ---")

# Using the exact same column order as X_clf
new_data = pd.DataFrame({
    "exports": [2500],
    "imports": [2700],
    "gdp": [30000]
})

predicted_risk = clf_model.predict(new_data)
print(f"Features:\n{new_data}")
print(f"\nPredicted Risk Level: {predicted_risk[0]}")

<img width="709" height="470" alt="téléchargement (8)" src="https://github.com/user-attachments/assets/81df721c-b769-4b89-a690-27cf1b897ac3" />
