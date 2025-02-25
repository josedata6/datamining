import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
# Generate Synthetic Data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100)
# Target variable with noise
df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 6)]) 
df['Target'] = y
# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Target"]),\
  df["Target"], test_size=0.2, random_state=42)

# Initialize Model
model = LinearRegression()
# Forward Feature Selection (Adding Features Step by Step)
forward_selector = SequentialFeatureSelector(model, n_features_to_select="auto", \
  direction="forward", cv=5)
forward_selector.fit(X_train, y_train)
selected_forward = X_train.columns[forward_selector.get_support()]
print(f"Selected Features (Forward): {list(selected_forward)}")
# Backward Feature Selection (Removing Features Step by Step)
backward_selector = SequentialFeatureSelector(model, n_features_to_select="auto", \
  direction="backward", cv=5)
backward_selector.fit(X_train, y_train)
selected_backward = X_train.columns[backward_selector.get_support()]
print(f"Selected Features (Backward): {list(selected_backward)}")
# Train and Evaluate Models
X_train_forward, X_test_forward = X_train[selected_forward], \
  X_test[selected_forward]
X_train_backward, X_test_backward = X_train[selected_backward], \
  X_test[selected_backward]
# Train forward selection model
model.fit(X_train_forward, y_train)
y_pred_forward = model.predict(X_test_forward)
mse_forward = mean_squared_error(y_test, y_pred_forward)
# Train backward selection model
model.fit(X_train_backward, y_train)
y_pred_backward = model.predict(X_test_backward)
mse_backward = mean_squared_error(y_test, y_pred_backward)
print(f"Forward Selection MSE: {mse_forward:.4f}")
print(f"Backward Selection MSE: {mse_backward:.4f}")