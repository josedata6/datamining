# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)
# Step 1: Generate synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 points from 0 to 10
y = 2 + 3 * X - 0.5 * X**2 + np.random.normal(0, 2, (100, 1))  # Add some noise
# Step 2: Transform the data to include polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # Creates [1, X, X^2] as features
# Step 3: Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
# Step 4: Make predictions
y_pred = model.predict(X_poly)
# Step 5: Calculate R-squared
r2 = r2_score(y, y_pred)
print(f"R-squared: {r2:.4f}")
# Step 6: Visualize the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Polynomial fit (degree 2)', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()