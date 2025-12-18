import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
IMPORTANT_FEATURES = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','count','srv_count',
    'serror_rate','srv_serror_rate',
    'dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate'
]

# Load dataset (disable low memory warning)
df = pd.read_csv("data/KDD_dataset.csv", low_memory=False)

# Convert all possible columns to numeric
df_numeric = df.apply(pd.to_numeric, errors="coerce")

# Drop columns that are completely non-numeric
df_numeric = df_numeric.dropna(axis=1, how="all")

# Fill remaining NaN values
df_numeric = df_numeric.fillna(0)

# Fit scaler
scaler = MinMaxScaler()
scaler.fit(df_numeric)

# Save scaler
joblib.dump(scaler, "preprocessors/scaler.pkl")

print("Scaler saved successfully")
print("Scaled feature count:", df_numeric.shape[1])

