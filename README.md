# Network Intrusion Detection System (NIDS)

A Flask-based **Network Intrusion Detection System** that predicts whether a network connection is **Normal** or an **Attack** using a trained deep learning model. The UI collects **13 high-impact features**, while the backend ensures preprocessing is consistent with training.

---

## ✨ Features

* 🔐 Detects malicious network activity
* 🧠 Deep Learning model (Keras/TensorFlow)
* 🧮 MinMax scaling + label encoding
* 🖥️ Simple Flask web interface
* 📦 Clean GitHub repo (dataset excluded via `.gitignore`)

---

## 🧱 Project Structure

```
nids-project/
├── app.py                     # Flask application
├── models/
│   └── nids_model.keras        # Trained model (expects 13 features)
├── preprocessors/
│   ├── create_scaler.py        # Builds scaler using 13 features
│   ├── create_encoders.py      # Builds label encoders
│   └── scaler.pkl              # Saved scaler (13 features)
├── templates/
│   ├── index.html              # Input form (13 features)
│   └── result.html             # Prediction output
├── static/                     # CSS / assets (if any)
├── data/                       # Dataset folder (ignored by git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧩 Input Features (UI – 13)

The UI collects only the most influential features:

* `duration`
* `protocol_type`
* `service`
* `flag`
* `src_bytes`
* `dst_bytes`
* `count`
* `srv_count`
* `serror_rate`
* `srv_serror_rate`
* `dst_host_count`
* `dst_host_srv_count`
* `dst_host_same_srv_rate`

> ℹ️ The **model and scaler are trained on these same 13 features**, ensuring consistent preprocessing and inference.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/HariLekkala3223/nids-project.git
cd nids-project
```

### 2️⃣ Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset is **not included** in this repository (best practice).

### Download

* Use the KDD dataset and place it here:

```
data/KDD_dataset.csv
```

> The `data/` directory is ignored by Git via `.gitignore`.

---

## 🛠️ Build Preprocessors

### Create label encoders

```bash
python preprocessors/create_encoders.py
```

### Create scaler (13 features)

```bash
python preprocessors/create_scaler.py
```

You should see:

```
Scaled feature count: 13
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works (Inference Pipeline)

1. User enters **13 features** in the UI
2. Categorical values are **label-encoded**
3. Numeric values are **scaled** using `MinMaxScaler`
4. The model predicts **Attack / Normal**

---

## 🧪 Output

* ✅ **Network Safe**
* 🚨 **Attack Detected**

---

## 🔍 Troubleshooting

* **Feature mismatch errors**: Ensure the scaler and model are trained on the **same 13 features**.
* **Indentation errors**: Use **4 spaces only**, no tabs.
* **Missing dataset**: Place `KDD_dataset.csv` inside `data/`.

---

## 🧠 Interview Notes

* The UI uses **fewer features** for usability.
* The model and scaler are trained on the **same feature space**.
* Datasets are excluded from GitHub to keep the repo lightweight and reproducible.

---

## 📜 License

This project is for educational and research purposes.

---

## �� Acknowledgements

* KDD Cup Dataset
* Flask, TensorFlow, scikit-learn

