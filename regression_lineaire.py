import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
import joblib  # Import joblib for saving models # type: ignore

# Load the dataset
df = pd.read_csv('cardekho.csv')

# Check that the 'mileage(km/ltr/kg)' column is of type float
if df['mileage(km/ltr/kg)'].dtype != 'float':
    raise ValueError("'mileage(km/ltr/kg)' column should be of type float.")

# Create a pipeline to handle model training and prediction
pipeline = Pipeline([
    ('regressor', LinearRegression())  # Linear regression model
])

# Clean the dataset
df_filtered = df.dropna(subset=['selling_price', 'mileage(km/ltr/kg)'])
df_filtered = df_filtered[(df_filtered['selling_price'] < 3e6) & (df_filtered['mileage(km/ltr/kg)'] < 35)]

# Variables
X = df_filtered[['mileage(km/ltr/kg)']]  # No need to extract float, column is already float
y = df_filtered['selling_price']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Save the pipeline as a single .pkl file
joblib.dump(pipeline, 'model/linear_regression_pipeline.pkl')


print("Linear regression pipeline saved successfully.")