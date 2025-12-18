import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load dataset with robust type inference
data_file = "KDD Dataset.csv"
df = pd.read_csv(data_file, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Preview first few rows to catch any parsing issue
print("🔍 Preview of dataset:")
print(df.head())

# Drop rows where any important feature is missing or has invalid data
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Define important features
important_features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                      'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                      'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate']

# Ensure these features are present
missing_features = [feat for feat in important_features if feat not in df.columns]
if missing_features:
    raise ValueError(f"❌ Missing required features in dataset: {missing_features}")

# Detect label column
label_column = next((col for col in ['label', 'class'] if col in df.columns), None)
if not label_column:
    raise ValueError("❌ ERROR: 'label' or 'class' column not found in dataset!")

# Encode categorical features
categorical_features = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Binary encode target column
df['target'] = (df[label_column].str.strip().astype(str) != 'normal').astype(int)

# Convert all features to numeric, handle conversion errors
for col in important_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs after conversion
df.dropna(subset=important_features, inplace=True)

# Select features and labels
X = df[important_features]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing tools
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build DNN model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("nids_model.keras", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
]

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=2
)

print("✅ Training complete. Model saved as 'nids_model.keras'.")
