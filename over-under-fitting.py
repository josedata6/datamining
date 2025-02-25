import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate Synthetic Data
np.random.seed(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = X**3 - 3*X + np.random.randn(50, 1) * 3 #random noise

# Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Plot Models with Different Degrees
plt.figure(figsize=(12, 5))
degrees = [1, 3, 10]  # Linear, Cubic, Overfitting
titles = ["Underfitting (Linear)", "Good Fit (Cubic)", "Overfitting (Degree=10)"]

for i, degree in enumerate(degrees): #
    poly = PolynomialFeatures(degree=degree) 
    X_train_poly = poly.fit_transform(X_train) 
    X_test_poly = poly.transform(X_test)

    model = LinearRegression().fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)

    # Plot Results
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, color="blue", label="Train Data")
    plt.scatter(X_test, y_test, color="red", label="Test Data")
    X_range = np.linspace(-3, 3, 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(poly.transform(X_range)), color="black", linewidth=2)
    plt.title(f"{titles[i]}\nMSE: {mse:.2f}")
    plt.legend()
plt.tight_layout()
plt.show()