import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. LOAD DATASET
# ==========================================
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Rename target variable for clarity
df.rename(columns={"MedHouseVal": "Price"}, inplace=True)
print("Dataset Head:")
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING & SPLITTING
# ==========================================
y = df["Price"]

# Simple Linear Regression Setup (Using 'MedInc' - Median Income)
X_simple = df[["MedInc"]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, random_state=42, test_size=0.2
)

# Multiple Linear Regression Setup (Using all features)
X_multi = df.drop(columns=["Price"])
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, random_state=42, test_size=0.2
)

# ==========================================
# 3. SIMPLE LINEAR REGRESSION
# ==========================================
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Predictions
y_pred_s = simple_model.predict(X_test_s)

# Metrics
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\n--- SIMPLE LINEAR REGRESSION ---")
print(f"Intercept: {simple_model.intercept_:.4f}")
print(f"Coefficient (MedInc): {simple_model.coef_[0]:.4f}")
print(f"MAE: {mae_s:.4f}")
print(f"MSE: {mse_s:.4f}")
print(f"R² Score: {r2_s:.4f}")

# ==========================================
# 4. MULTIPLE LINEAR REGRESSION
# ==========================================
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# Predictions
y_pred_m = multi_model.predict(X_test_m)

# Metrics
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("\n--- MULTIPLE LINEAR REGRESSION ---")
print(f"Intercept: {multi_model.intercept_:.4f}")
coef_df = pd.DataFrame(
    {"Feature": X_multi.columns, "Coefficient": multi_model.coef_}
)
print(coef_df)
print(f"MAE: {mae_m:.4f}")
print(f"MSE: {mse_m:.4f}")
print(f"R² Score: {r2_m:.4f}")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Simple Linear Regression Fit
axes[0].scatter(X_test_s, y_test_s, alpha=0.3, color="blue", label="Actual Data")
axes[0].plot(
    X_test_s, y_pred_s, color="red", linewidth=2, label="Regression Line"
)
axes[0].set_xlabel("Median Income (MedInc)")
axes[0].set_ylabel("House Price ($100k)")
axes[0].set_title("Simple Linear Regression Line")
axes[0].legend()

# Plot 2: Actual vs Predicted (Multiple Regression)
axes[1].scatter(y_test_m, y_pred_m, alpha=0.3, color="green")
axes[1].plot(
    [y_test_m.min(), y_test_m.max()],
    [y_test_m.min(), y_test_m.max()],
    "k--",
    lw=2,
)
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title("Multiple Regression: Actual vs. Predicted")

plt.tight_layout()
plt.show()