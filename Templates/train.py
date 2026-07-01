import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# Features and target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(random_state=0)

model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/logistic_model.joblib")

# Save scaler
joblib.dump(scaler, "models/scaler.joblib")

# Save pickle model
with open("models/logistic_model.pickle", "wb") as f:
    pickle.dump(model, f)

# Save model info
model_info = {
    "model_type": "Logistic Regression",
    "accuracy": float(accuracy),
    "feature_names": X.columns.tolist(),
    "target_names": ["Not Purchased", "Purchased"]
}

with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

# Save feature ranges
feature_ranges = {
    "Age": {
        "min": int(X["Age"].min()),
        "max": int(X["Age"].max()),
        "default": int(X["Age"].mean())
    },
    "EstimatedSalary": {
        "min": int(X["EstimatedSalary"].min()),
        "max": int(X["EstimatedSalary"].max()),
        "default": int(X["EstimatedSalary"].mean())
    }
}

with open("models/feature_ranges.json", "w") as f:
    json.dump(feature_ranges, f, indent=4)

print("\nFiles Saved Successfully!")
print("✔ logistic_model.joblib")
print("✔ scaler.joblib")
print("✔ logistic_model.pickle")
print("✔ model_info.json")
print("✔ feature_ranges.json")