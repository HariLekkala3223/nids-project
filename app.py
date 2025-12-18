
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model("models/nids_model.keras")
scaler = joblib.load("preprocessors/scaler.pkl")
label_encoders = joblib.load("preprocessors/encoders.pkl")

# Define important features (same as used in training)
important_features = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate'

]
ALL_FEATURES = [
    "duration","protocol_type","service","flag",
    "src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","num_root",
    "num_file_creations","num_shells","num_access_files",
    "is_guest_login","count","srv_count",
    "serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]


# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=important_features,
        encoders=label_encoders,
        prediction=None,
        input_values={}
    )
def build_full_feature_vector(form, encoders):
    row = {}

    # take only 13 UI inputs
    for feature in important_features:
        value = form[feature]

        if feature in encoders:
            value = encoders[feature].transform([value])[0]
        else:
            value = float(value)

        row[feature] = value

    # fill remaining features with 0
    for feature in ALL_FEATURES:
        if feature not in row:
            row[feature] = 0

    # return in training order (38 features)
    return [row[f] for f in ALL_FEATURES]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = {f: request.form[f] for f in important_features}

        full_features = build_full_feature_vector(
            request.form,
            label_encoders
        )

        X = np.array(full_features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        result = "🚨 Attack Detected!" if prediction > 0.5 else "✅ Network Safe"

    except Exception as e:
        result = f"❌ Error processing input: {e}"
        input_values = {}

    return render_template(
        'index.html',
        prediction=result,
        features=important_features,
        encoders=label_encoders,
        input_values=input_values
    )

if __name__ == '__main__':
    app.run(debug=True)
