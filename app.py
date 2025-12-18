
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

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    input_values = {}

    try:
        for feature in important_features:
            value = request.form[feature]
            input_values[feature] = value

            # Apply label encoding for categorical features
            if feature in label_encoders:
                if value not in label_encoders[feature].classes_:
                    return render_template('index.html', prediction="❌ Error: Invalid input value!", features=important_features, encoders=label_encoders, input_values=input_values)
                value = label_encoders[feature].transform([value])[0]
            else:
                value = float(value)

            input_data.append(value)

        # Normalize input
        input_scaled = scaler.transform([input_data])

        # Make prediction
        prediction = model.predict(input_scaled)
        result = "🚨 Attack Detected!" if prediction > 0.5 else "✅ Network Safe"

    except Exception as e:
        result = f"❌ Error processing input: {e}"

    return render_template('index.html', prediction=result, features=important_features, encoders=label_encoders, input_values=input_values)

if __name__ == '__main__':
    app.run(debug=True)
