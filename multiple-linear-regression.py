import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


# Set random seed for reproducibility
np.random.seed(42)
# Step 1: Generate synthetic data
n_samples = 100
X1 = np.random.uniform(1000, 3000, n_samples)  # Predictor 1: e.g., square footage
X2 = np.random.uniform(1, 5, n_samples)       # Predictor 2: e.g., number of bedrooms
X = np.column_stack((X1, X2))                 # Combine into a 2D array
# True relationship: y = 50000 + 100*X1 + 20000*X2 + noise
y = 50000 + 100 * X1 + 20000 * X2 + np.random.normal(0, 10000, n_samples)
# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% train (80 samples), 20% test (20 samples)

# Step 3: Fit the Multiple Linear Regression model on training data
model = LinearRegression()
model.fit(X_train, y_train)
# Step 4: Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Step 5: Evaluate the model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
# Step 6: Print results
print("Model Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient for X1: {model.coef_[0]:.2f}")
print(f"Coefficient for X2: {model.coef_[1]:.2f}")
print(f"R-squared (Training): {r2_train:.4f}")
print(f"R-squared (Testing): {r2_test:.4f}")