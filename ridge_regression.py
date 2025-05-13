from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import Ridge # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

# Load data
df = pd.read_csv('cardekho.csv')

# Separate features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Align y (in case encoding dropped rows)
y = y.loc[X_encoded.index]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Save the entire pipeline as one job
joblib.dump(pipeline, 'model/ridge_pipeline.pkl')
