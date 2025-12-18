import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Define column names
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Load dataset
train_df = pd.read_csv("KDDTrain+.csv", names=columns)

# Initialize label encoders
protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

# Fit encoders
train_df["protocol_type"] = protocol_encoder.fit_transform(train_df["protocol_type"])
train_df["service"] = service_encoder.fit_transform(train_df["service"])
train_df["flag"] = flag_encoder.fit_transform(train_df["flag"])

# Save encoders
pickle.dump(protocol_encoder, open("protocol_encoder.pkl", "wb"))
pickle.dump(service_encoder, open("service_encoder.pkl", "wb"))
pickle.dump(flag_encoder, open("flag_encoder.pkl", "wb"))

print("✅ Successfully saved protocol_encoder.pkl, service_encoder.pkl, and flag_encoder.pkl")
