import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/KDD_dataset.csv", low_memory=False)

# Columns that need encoding (typical for KDD)
categorical_cols = df.select_dtypes(include=["object"]).columns

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "preprocessors/encoders.pkl")

print("Encoders saved successfully")
print("Encoded columns:", list(encoders.keys()))

