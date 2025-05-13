import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore
import joblib  # type: ignore

# === Load dataset ===
df = pd.read_csv('cardekho.csv')

# Drop non-numeric 'name' column
df.drop(columns=['name'], inplace=True, errors='ignore')

# Ensure 'max_power' is numeric
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

# Separate features and target
X = df.drop(columns=['selling_price'], errors='ignore')
y = df['selling_price']

# Drop rows where target is missing
mask = ~y.isna()
X = X[mask]
y = y[mask]

# === Handle missing values in numerics ===
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())

# === Identify column types ===
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# === Preprocessing pipeline ===
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# === Random Forest pipeline ===
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ))
])

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train model ===
rf_pipeline.fit(X_train, y_train)

# === Evaluate model ===
y_pred = rf_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully.")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# === Save model ===
joblib.dump(rf_pipeline, 'model/randomrf.pkl')
print("Model saved to 'model/randomrf.pkl'")
