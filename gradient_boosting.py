import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import joblib  # type: ignore

# Load data
df = pd.read_csv('cardekho.csv')

# Clean columns
def extract_float(series):
    return series.str.extract(r'(\d+\.?\d*)').astype(float)

for col in ['mileage(km/ltr/kg)', 'engine', 'max_power']:
    if df[col].dtype == 'object':
        df[col] = extract_float(df[col])

# Drop rows with NaN in required fields
df.dropna(subset=['selling_price', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats'], inplace=True)

# Remove outliers
df = df[(df['selling_price'] < 3e6) & (df['mileage(km/ltr/kg)'] < 35)]

# Select features
features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats'] + \
           [col for col in df.columns if col.startswith(('fuel_', 'seller_type_', 'transmission_', 'owner_'))]
X = df[features]
y = df['selling_price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save full pipeline
joblib.dump(pipeline, 'model/gb_pipeline.pkl')

print("Gradient Boosting pipeline saved successfully.")
