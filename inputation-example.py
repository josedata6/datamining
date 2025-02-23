import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Create DataFrame with missing values
data = {"Feature 1": [1, 2, 3, 4], "Feature 2": [2, 3, np.nan, 5]}
df = pd.DataFrame(data)

# Apply Mean Imputation
imputer = SimpleImputer(strategy="mean")  # Using mean to fill missing values
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Print before and after imputation
print("Original DataFrame:")
print(df)
print("\nDataFrame after Mean Imputation:")
print(df_imputed)