import pandas as pd  # type: ignore
from sklearn.cross_decomposition import PLSRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
import joblib  # type: ignore

# Load the dataset
df = pd.read_csv('cardekho.csv')

# Clean numeric text columns
def extract_float(series):
    return series.astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

for col in ['mileage(km/ltr/kg)', 'engine', 'max_power']:
    df[col] = extract_float(df[col])

# Drop rows with missing target
df.dropna(subset=['selling_price'], inplace=True)

# Remove outliers
df = df[(df['selling_price'] < 3e6) & (df['mileage(km/ltr/kg)'] < 35)]

# Define features and target
X = df.drop(columns=['selling_price'])
y = df['selling_price']

# Define feature types
numeric_features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats', 'km_driven']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Final pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pls', PLSRegression(n_components=2))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the full pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Save the complete pipeline
joblib.dump(pipeline, 'model/pls_pipeline.pkl')
